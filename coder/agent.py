"""Action-Incremental DQN Agent.

Handles the core RL loop with dynamic action-space expansion:
- Detects when the environment has added new actions
- Expands the action embedding table via k-NN interpolation
- Applies progressive unfreezing (Phase 1 -> Phase 2 -> Phase 3)
- Manages optimistic exploration bonuses for new actions
- Maintains target network synchronization

Causal contract (off-policy):
  Q(s,a) = r + gamma * max_{a'} Q'(s', a')
  The TD target naturally includes newly added actions in the max
  over actions, which is correct because the value of s' genuinely
  increases when a new action becomes available (monotonicity property
  of nested action spaces: V*_i(s) <= V*_j(s) for i < j).

KV-cache / state-cache interface:
  This is a DQN agent (no autoregressive generation), so no KV-cache
  is needed. The equivalent is the target network, which is synced
  periodically via hard update.
"""
import random
from collections import deque, namedtuple
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_config import ModelConfig
from networks import (
    StateEncoder,
    ActionEmbeddingTable,
    QHead,
    ValueHead,
    DynamicsHead,
    count_params,
)

# ── Replay Buffer ────────────────────────────────────────────────────────────

Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    """Fixed-capacity replay buffer for off-policy DQN.

    Stores (s, a, r, s', done) tuples. All transitions remain valid
    after action-space expansion because action indices are stable:
    old actions keep their indices, and new actions get new indices.
    """

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action: int, reward: float, next_state, done: bool) -> None:
        """Store a single transition."""
        self.buffer.append(
            Transition(
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                float(done),
            )
        )

    def sample(self, batch_size: int) -> Transition:
        """Sample a batch of transitions uniformly.

        Returns:
            Transition namedtuple with batched tensors.
            - state:      (B, obs_dim) or (B, C, H, W)
            - action:     (B,) long
            - reward:     (B,)
            - next_state: (B, obs_dim) or (B, C, H, W)
            - done:       (B,)
        """
        batch = random.sample(self.buffer, batch_size)
        return Transition(
            state=torch.tensor(
                np.stack([t.state for t in batch]), dtype=torch.float32
            ),
            action=torch.tensor([t.action for t in batch], dtype=torch.long),
            reward=torch.tensor([t.reward for t in batch], dtype=torch.float32),
            next_state=torch.tensor(
                np.stack([t.next_state for t in batch]), dtype=torch.float32
            ),
            done=torch.tensor([t.done for t in batch], dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ── Agent ────────────────────────────────────────────────────────────────────


class ActionIncrementalDQN:
    """Deep Q-Network agent that handles unexpected action-space expansion.

    The agent maintains:
    - A factorized Q-function Q(s,a) = f_Q(phi(s), e_a)
    - An expandable action embedding table E
    - Progressive freeze schedules to prevent catastrophic forgetting
    - Optimistic exploration bonuses for newly added actions

    Usage:
        agent = ActionIncrementalDQN(cfg, n_actions_init=3)
        # ... training loop ...
        if env.action_space.n > agent.action_count:
            agent.handle_action_expansion(env)
    """

    def __init__(self, cfg: ModelConfig, n_actions: int):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Online networks ─────────────────────────────────────────────
        self.encoder = StateEncoder(cfg).to(self.device)                     # phi(s)
        self.action_embeddings = ActionEmbeddingTable(n_actions, cfg).to(self.device)  # E

        if cfg.q_form == "factorized":
            self.q_head = QHead(cfg).to(self.device)                        # f_Q(phi(s), e_a)
            self.value_head = None
        elif cfg.q_form == "dueling_factorized":
            self.q_head = QHead(cfg).to(self.device)                        # A(s,a) advantage
            self.value_head = ValueHead(cfg).to(self.device)                # V(s)
        else:
            raise ValueError(f"Unknown q_form: {cfg.q_form}")

        self.aux_head = (
            DynamicsHead(cfg).to(self.device)
            if cfg.use_auxiliary_dynamics_loss
            else None
        )

        # ── Target networks (periodic hard copy) ─────────────────────────
        self.target_encoder = StateEncoder(cfg).to(self.device)
        self.target_action_embeddings = ActionEmbeddingTable(n_actions, cfg).to(self.device)
        self.target_q_head = QHead(cfg).to(self.device)

        if cfg.q_form == "dueling_factorized":
            self.target_value_head = ValueHead(cfg).to(self.device)
        else:
            self.target_value_head = None

        self._sync_target()

        # ── Optimizer (regenerated after each expansion) ─────────────────
        self.optimizer = torch.optim.Adam(
            self._trainable_params(), lr=cfg.learning_rate
        )

        # ── Replay buffer ───────────────────────────────────────────────
        self.replay_buffer = ReplayBuffer(cfg.buffer_capacity)

        # ── Expansion state ─────────────────────────────────────────────
        self.action_count = n_actions          # current |A|
        self.expansion_step = 0                # env step when last expansion occurred
        self.freeze_state: str = "none"        # "none" | "phase1" | "phase2" | "phase3"
        self.new_action_indices: set = set()   # set of indices added post-init

        # ── Exploration bonus tracking ──────────────────────────────────
        self.action_visit_counts = torch.zeros(cfg.max_n_actions, dtype=torch.long)
        self.action_bonuses: dict[int, float] = {}  # {action_idx: current_bonus_value}

        # ── Step counter ────────────────────────────────────────────────
        self.env_step = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def select_action(self, s: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy action selection with optimistic exploration bonus.

        For new actions, an additive bonus is applied during greedy
        selection to encourage directed exploration beyond uniform
        epsilon-greedy randomness.

        Args:
            s: State observation — shape (obs_dim,) for vector obs
               or (C, H, W) for image obs.
            epsilon: Probability of random action (in [0, 1]).
        Returns:
            Action index (int).
        """
        if random.random() < epsilon:
            return random.randint(0, self.action_count - 1)

        with torch.no_grad():
            s_t = torch.tensor(s, dtype=torch.float32, device=self.device)

            # Add batch dimension if needed
            if s_t.dim() == 1:
                s_t = s_t.unsqueeze(0)                                     # (1, obs_dim)
            elif s_t.dim() == 3:
                s_t = s_t.unsqueeze(0)                                     # (1, C, H, W)

            phi_s = self.encoder(s_t)                                       # (1, d_enc)

            # Batch-compute Q(s, a) for ALL known actions at once
            q_values = self._batch_q_values(phi_s)                         # (action_count,)
            q_values = q_values.cpu().numpy()                              # (action_count,)

            # Add exploration bonuses for new actions
            for a_idx, bonus in self.action_bonuses.items():
                if a_idx < self.action_count:
                    q_values[a_idx] += bonus

        return int(np.argmax(q_values))

    def update(self, batch: Transition) -> dict:
        """Single TD-learning step with optional auxiliary dynamics loss.

        Handles variable-sized action sets by computing the max over ALL
        currently known actions in the target computation. This naturally
        includes any newly added actions even for transitions recorded
        before expansion.

        Args:
            batch: Transition namedtuple from ReplayBuffer.sample().
        Returns:
            Dict of scalar metrics for logging.
        """
        states, actions, rewards, next_states, dones = batch
        states = states.to(self.device)                                     # (B, obs_dim) or (B, C, H, W)
        actions = actions.to(self.device)                                   # (B,)
        rewards = rewards.to(self.device)                                   # (B,)
        next_states = next_states.to(self.device)                           # (B, ...)
        dones = dones.to(self.device)                                       # (B,)

        # ── Online Q(s, a) ──────────────────────────────────────────────
        phi_s = self.encoder(states)                                        # (B, d_enc)
        e_actions = self.action_embeddings(actions)                         # (B, d_emb)
        q_sa = self._compute_q(phi_s, e_actions).squeeze(-1)                # (B,)

        # ── Target: y = r + gamma * max_{a'} Q'(s', a') ────────────────
        with torch.no_grad():
            phi_s_prime = self.target_encoder(next_states)                  # (B, d_enc)

            # Batch-compute Q'(s', a') for ALL known actions
            q_next_all = self._batch_target_q_values(phi_s_prime)           # (B, action_count)
            q_next = q_next_all.max(dim=1).values                           # (B,)
            y = rewards + self.cfg.gamma * q_next * (1.0 - dones)           # (B,)

        # bf16/fp16 safety: MSE loss is fine in native precision
        td_loss = F.mse_loss(q_sa, y)

        # ── Optional auxiliary dynamics loss ────────────────────────────
        aux_loss = torch.tensor(0.0, device=self.device)
        if self.aux_head is not None:
            e_actions_aux = self.action_embeddings(actions)                 # (B, d_emb)
            pred_delta = self.aux_head(phi_s, e_actions_aux)                # (B, d_enc)
            with torch.no_grad():
                target_delta = phi_s_prime - phi_s.detach()                  # (B, d_enc)
            aux_loss = F.mse_loss(pred_delta, target_delta)

        total_loss = td_loss + 0.01 * aux_loss

        # ── Optimize ────────────────────────────────────────────────────
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._trainable_params(), 10.0)
        self.optimizer.step()

        return {
            "td_loss": td_loss.item(),
            "aux_loss": aux_loss.item(),
            "q_mean": q_sa.mean().item(),
        }

    def handle_action_expansion(self, env) -> None:
        """Integrate a newly discovered action into the network.

        Called when env.action_space.n > self.action_count.

        Protocol:
        1. k-NN initialisation of the new embedding (online AND target).
        2. Register new action index and set optimistic exploration bonus.
        3. Apply Phase 1 freeze (encoder + old embeddings + Q-head frozen).
        4. Rebuild optimizer with only trainable parameters.
        """
        print(
            f"[expansion] Action space growing from {self.action_count} -> "
            f"{self.action_count + 1} at env step {self.env_step}"
        )

        # 1a. Online network: new embedding via k-NN
        self.action_embeddings.add_embedding(
            n_known=self.action_count,
            k=self.cfg.k_nn,
            similarity=self.cfg.embedding_similarity,
        )

        # 1b. Target network: copy from online + identical init
        self.target_action_embeddings.add_embedding(
            n_known=self.action_count,
            k=self.cfg.k_nn,
            similarity=self.cfg.embedding_similarity,
            source_embeddings=self.action_embeddings.weight.data,
        )

        # 2. Register new action
        new_idx = self.action_count
        self.new_action_indices.add(new_idx)
        self.action_count += 1
        self.expansion_step = self.env_step

        # 3. Set optimistic exploration bonus
        self.action_bonuses[new_idx] = self.cfg.optimistic_init_bonus
        self.action_visit_counts[new_idx] = 0

        # 4. Apply Phase 1 freeze
        self._apply_freeze_phase1()

        # 5. Rebuild optimizer with updated parameter set
        self.optimizer = torch.optim.Adam(
            self._trainable_params(), lr=self.cfg.learning_rate
        )

    def maybe_update_freeze_schedule(self) -> None:
        """Progress through freeze phases based on steps since expansion.

        Phase 1 -> Phase 2: after freeze_q_head_steps (unfreeze Q-head)
        Phase 2 -> Phase 3: after freeze_encoder_steps (unfreeze all, reduce LR)
        """
        steps_since = self.env_step - self.expansion_step

        if self.freeze_state == "phase1" and steps_since >= self.cfg.freeze_q_head_steps:
            self._apply_freeze_phase2()
            print(f"[expansion] Phase 1 -> Phase 2 at env step {self.env_step}")

        if self.freeze_state == "phase2" and steps_since >= self.cfg.freeze_encoder_steps:
            self._apply_freeze_phase3()
            print(f"[expansion] Phase 2 -> Phase 3 at env step {self.env_step}")

    def decay_exploration_bonuses(self) -> None:
        """Decay exploration bonuses for visited new actions."""
        for idx in self.new_action_indices:
            if self.action_visit_counts[idx] > 0:
                self.action_bonuses[idx] = max(
                    self.cfg.optimistic_bonus_min,
                    self.action_bonuses[idx] * self.cfg.optimistic_bonus_decay,
                )

    def get_q_for_actions(self, s: np.ndarray) -> np.ndarray:
        """Compute Q(s, a) for all currently known actions.

        Used for evaluation / logging: measure Q-value degradation
        on old actions after expansion.

        Args:
            s: State observation.
        Returns:
            (|A|,) array of Q-values.
        """
        with torch.no_grad():
            s_t = torch.tensor(s, dtype=torch.float32, device=self.device)
            if s_t.dim() == 1:
                s_t = s_t.unsqueeze(0)                                     # (1, obs_dim)
            elif s_t.dim() == 3:
                s_t = s_t.unsqueeze(0)                                     # (1, C, H, W)

            phi_s = self.encoder(s_t)                                       # (1, d_enc)
            q_vals = self._batch_q_values(phi_s)                           # (action_count,)

        return q_vals.cpu().numpy()

    # ─────────────────────────────────────────────────────────────────────────
    # Internal Q-value computation
    # ─────────────────────────────────────────────────────────────────────────

    def _batch_q_values(self, phi_s: torch.Tensor) -> torch.Tensor:
        """Compute Q(s, a) for ALL known actions in a single batched pass.

        Args:
            phi_s: (1, d_enc) or (B, d_enc) state representation.
        Returns:
            (action_count,) Q-values if phi_s is (1, d_enc),
            (B, action_count) if phi_s is (B, d_enc).
        """
        n_act = self.action_count
        all_embeds = self.action_embeddings.weight[:n_act]                  # (n_act, d_emb)

        if self.cfg.q_form == "factorized":
            # phi_s: (B, d_enc) -> (B, 1, d_enc) -> (B, n_act, d_enc)
            B = phi_s.size(0)
            phi_exp = phi_s.unsqueeze(1).expand(B, n_act, -1)              # (B, n_act, d_enc)
            e_exp = all_embeds.unsqueeze(0).expand(B, -1, -1)               # (B, n_act, d_emb)
            x = torch.cat([phi_exp, e_exp], dim=-1)                        # (B, n_act, d_enc+d_emb)
            q_all = self.q_head.net(x.view(B * n_act, -1))                  # (B*n_act, 1)
            q_all = q_all.view(B, n_act)                                    # (B, n_act)

        elif self.cfg.q_form == "dueling_factorized":
            B = phi_s.size(0)
            V = self.value_head(phi_s)                                      # (B, 1)

            # Batch-compute advantage for all actions
            phi_exp = phi_s.unsqueeze(1).expand(B, n_act, -1)              # (B, n_act, d_enc)
            e_exp = all_embeds.unsqueeze(0).expand(B, -1, -1)               # (B, n_act, d_emb)
            x = torch.cat([phi_exp, e_exp], dim=-1)                        # (B, n_act, d_enc+d_emb)
            A_all = self.q_head.net(x.view(B * n_act, -1)).view(B, n_act)   # (B, n_act)

            # Center advantage over KNOWN actions only
            n_known = n_act - len(self.new_action_indices) if self.new_action_indices else n_act
            if n_known > 0:
                A_known_mean = A_all[:, :n_known].mean(dim=1, keepdim=True)  # (B, 1)
                A_all = A_all - A_known_mean                                 # (B, n_act)

            q_all = V + A_all                                                # (B, n_act)

        else:
            raise ValueError(f"Unknown q_form: {self.cfg.q_form}")

        if B == 1:
            return q_all.squeeze(0)                                          # (n_act,)
        return q_all                                                         # (B, n_act)

    def _batch_target_q_values(self, phi_s_prime: torch.Tensor) -> torch.Tensor:
        """Compute target Q'(s', a') for ALL known actions in a single pass.

        Args:
            phi_s_prime: (B, d_enc) next state representation.
        Returns:
            (B, action_count) target Q-values.
        """
        n_act = self.action_count
        all_embeds = self.target_action_embeddings.weight[:n_act]           # (n_act, d_emb)

        B = phi_s_prime.size(0)

        if self.cfg.q_form == "factorized":
            phi_exp = phi_s_prime.unsqueeze(1).expand(B, n_act, -1)        # (B, n_act, d_enc)
            e_exp = all_embeds.unsqueeze(0).expand(B, -1, -1)               # (B, n_act, d_emb)
            x = torch.cat([phi_exp, e_exp], dim=-1)                        # (B, n_act, d_enc+d_emb)
            q_all = self.target_q_head.net(x.view(B * n_act, -1))           # (B*n_act, 1)
            q_all = q_all.view(B, n_act)                                    # (B, n_act)

        elif self.cfg.q_form == "dueling_factorized":
            V = self.target_value_head(phi_s_prime)                         # (B, 1)
            phi_exp = phi_s_prime.unsqueeze(1).expand(B, n_act, -1)        # (B, n_act, d_enc)
            e_exp = all_embeds.unsqueeze(0).expand(B, -1, -1)               # (B, n_act, d_emb)
            x = torch.cat([phi_exp, e_exp], dim=-1)                        # (B, n_act, d_enc+d_emb)
            A_all = self.target_q_head.net(x.view(B * n_act, -1)).view(B, n_act)  # (B, n_act)
            q_all = V + A_all                                                # (B, n_act)

        else:
            raise ValueError(f"Unknown q_form: {self.cfg.q_form}")

        return q_all                                                         # (B, n_act)

    def _compute_q(self, phi_s: torch.Tensor, e_a: torch.Tensor) -> torch.Tensor:
        """Compute Q(s, a) for a single action embedding.

        This is used for the TD target of SPECIFIC taken actions,
        not for the max over all actions.

        Args:
            phi_s: (B, d_enc)
            e_a:   (B, d_emb)
        Returns:
            (B, 1) Q-values.
        """
        if self.cfg.q_form == "factorized":
            return self.q_head(phi_s, e_a)                                   # (B, 1)

        elif self.cfg.q_form == "dueling_factorized":
            V = self.value_head(phi_s)                                       # (B, 1)
            A = self.q_head(phi_s, e_a)                                      # (B, 1)
            return V + A                                                     # (B, 1)

        else:
            raise ValueError(f"Unknown q_form: {self.cfg.q_form}")

    # ─────────────────────────────────────────────────────────────────────────
    # Network management
    # ─────────────────────────────────────────────────────────────────────────

    def _sync_target(self) -> None:
        """Hard copy online network parameters to target network.

        This is the 'KV-cache sync' equivalent for DQN — ensures
        target Q-values are stable during TD learning.
        """
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_action_embeddings.load_state_dict(
            self.action_embeddings.state_dict()
        )
        self.target_q_head.load_state_dict(self.q_head.state_dict())
        if self.target_value_head is not None and self.value_head is not None:
            self.target_value_head.load_state_dict(self.value_head.state_dict())

    # ─────────────────────────────────────────────────────────────────────────
    # Progressive freeze schedule
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_freeze_phase1(self) -> None:
        """Phase 1: Only the new action embedding is trainable.

        Frozen:
          - Encoder (all params)
          - Value head (dueling variant)
          - Q-head (all params)
          - Old action embeddings (all except new_action_indices)

        Trainable:
          - New action embedding(s) only
        """
        self.freeze_state = "phase1"

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Freeze value head (dueling variant)
        if self.value_head is not None:
            for p in self.value_head.parameters():
                p.requires_grad = False

        # Freeze Q-head
        for p in self.q_head.parameters():
            p.requires_grad = False

        # Freeze old action embeddings via the gradient mask
        if self.cfg.freeze_old_embeddings and self.new_action_indices:
            old_indices = torch.tensor(
                [i for i in range(self.action_count) if i not in self.new_action_indices],
                device=self.device,
            )
            if old_indices.numel() > 0:
                self.action_embeddings.set_rows_frozen(old_indices, frozen=True)

    def _apply_freeze_phase2(self) -> None:
        """Phase 2: Unfreeze Q-head and value head, keep encoder frozen.

        Trainable:
          - Q-head
          - Value head (dueling variant)
          - New action embedding(s)
        """
        self.freeze_state = "phase2"

        # Unfreeze Q-head
        for p in self.q_head.parameters():
            p.requires_grad = True

        # Unfreeze value head (dueling)
        if self.value_head is not None:
            for p in self.value_head.parameters():
                p.requires_grad = True

    def _apply_freeze_phase3(self) -> None:
        """Phase 3: Unfreeze everything with reduced learning rate.

        Trainable:
          - Encoder
          - Q-head
          - Value head
          - ALL action embeddings (including previously frozen old ones)
        """
        self.freeze_state = "phase3"

        # Unfreeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = True

        # Unfreeze all embedding rows (clear the freeze mask)
        self.action_embeddings.unfreeze_all()

        # Reduce learning rate
        for g in self.optimizer.param_groups:
            g["lr"] = self.cfg.learning_rate * 0.1

    def _trainable_params(self) -> list:
        """Return all parameters with requires_grad=True.

        This is called to rebuild the optimizer after each expansion
        and freeze-state transition.
        """
        params = []
        for module in [
            self.encoder,
            self.value_head,
            self.action_embeddings,
            self.q_head,
            self.aux_head,
        ]:
            if module is not None:
                params.extend([p for p in module.parameters() if p.requires_grad])
        return params
