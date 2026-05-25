"""Action-Incremental DQN Agent.

Handles the core RL loop with dynamic action-space expansion:
- Detects when the environment has added new actions
- Expands the action embedding table via k-NN interpolation
- Applies progressive unfreezing (Phase 1 → Phase 2 → Phase 3)
- Manages optimistic exploration bonuses for new actions
- Maintains target network synchronization
"""
import random
from collections import deque, namedtuple
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_config import ModelConfig
from components import (
    StateEncoder,
    ActionEmbeddingTable,
    QHead,
    ValueHead,
    DuelingFactorizedQ,
    DynamicsHead,
)

# ── Replay Buffer ────────────────────────────────────────────────────────────

Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    """Fixed-capacity replay buffer for off-policy DQN."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            Transition(
                np.array(state, dtype=np.float32),
                int(action),
                float(reward),
                np.array(next_state, dtype=np.float32),
                float(done),
            )
        )

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        return Transition(
            state=torch.tensor(np.stack([t.state for t in batch]), dtype=torch.float32),
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
    - A factorized Q-function Q(s,a) = f_Q(φ(s), e_a)
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
        self.encoder = StateEncoder(cfg).to(self.device)
        self.action_embeddings = ActionEmbeddingTable(n_actions, cfg).to(self.device)

        if cfg.q_form == "factorized":
            self.q_head = QHead(cfg).to(self.device)
            self.value_head = None
        elif cfg.q_form == "dueling_factorized":
            self.q_head = QHead(cfg).to(self.device)
            self.value_head = ValueHead(cfg).to(self.device)
        else:
            raise ValueError(f"Unknown q_form: {cfg.q_form}")

        self.aux_head = DynamicsHead(cfg).to(self.device) if cfg.use_auxiliary_dynamics_loss else None

        # ── Target networks (periodic hard copy) ─────────────────────────
        self.target_encoder = StateEncoder(cfg).to(self.device)
        self.target_action_embeddings = ActionEmbeddingTable(n_actions, cfg).to(self.device)
        self.target_q_head = QHead(cfg).to(self.device)

        if cfg.q_form == "dueling_factorized":
            self.target_value_head = ValueHead(cfg).to(self.device)
        else:
            self.target_value_head = None

        self._sync_target()

        # ── Optimizer (regenerated after expansion) ──────────────────────
        self.optimizer = torch.optim.Adam(
            self._trainable_params(), lr=cfg.learning_rate
        )

        # ── Replay buffer ───────────────────────────────────────────────
        self.replay_buffer = ReplayBuffer(cfg.buffer_capacity)

        # ── Expansion state ─────────────────────────────────────────────
        self.action_count = n_actions
        self.expansion_step = 0
        self.freeze_state: str = "none"   # "none" | "phase1" | "phase2" | "phase3"
        self.new_action_indices: set[int] = set()

        # ── Exploration bonus tracking ──────────────────────────────────
        self.action_visit_counts = torch.zeros(cfg.max_n_actions, dtype=torch.long)
        self.action_bonuses: dict[int, float] = {}  # {idx: current_bonus}

        # ── Step counter ────────────────────────────────────────────────
        self.env_step = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def select_action(self, s: np.ndarray, epsilon: float) -> int:
        """ε-greedy action selection with optimistic exploration bonus.

        For new actions, an additive bonus is applied during the greedy
        selection to encourage exploration.

        Args:
            s: State observation (shape depends on obs_type).
            epsilon: Probability of random action.
        Returns:
            Action index (int).
        """
        if random.random() < epsilon:
            return random.randint(0, self.action_count - 1)

        with torch.no_grad():
            s_t = torch.tensor(s, dtype=torch.float32, device=self.device)
            if s_t.dim() == 1:
                s_t = s_t.unsqueeze(0)  # (1, obs_dim)
            elif s_t.dim() == 3:
                s_t = s_t.unsqueeze(0)  # (1, C, H, W)

            phi_s = self.encoder(s_t)  # (1, d_enc)

            q_values = []
            for a in range(self.action_count):
                e_a = self.action_embeddings(
                    torch.tensor([a], device=self.device)
                ).squeeze(0)  # (d_emb,)

                q = self._compute_q(phi_s, e_a).item()
                bonus = self.action_bonuses.get(a, 0.0)
                q_values.append(q + bonus)

        return int(np.argmax(q_values))

    def update(self, batch: Transition) -> dict:
        """Single TD-learning step.

        Handles variable-sized action sets by computing max over all
        currently known actions in the target.

        Args:
            batch: Transition namedtuple from ReplayBuffer.
        Returns:
            Dict of scalar metrics.
        """
        states, actions, rewards, next_states, dones = batch
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # ── Online Q(s, a) ──────────────────────────────────────────────
        phi_s = self.encoder(states)                                   # (B, d_enc)
        e_actions = self.action_embeddings(actions)                    # (B, d_emb)
        q_sa = self._compute_q(phi_s, e_actions).squeeze(-1)           # (B,)

        # ── Target: y = r + γ * max_{a'} Q'(s', a') ────────────────────
        with torch.no_grad():
            phi_s_prime = self.target_encoder(next_states)              # (B, d_enc)

            # Compute Q'(s', a') for ALL known actions
            q_next_all = []
            for a in range(self.action_count):
                e_a_prime = self.target_action_embeddings(
                    torch.tensor([a], device=self.device)
                ).squeeze(0)                                            # (d_emb,)
                q_a = self._compute_target_q(phi_s_prime, e_a_prime)    # (B, 1)
                q_next_all.append(q_a)

            q_next = torch.stack(q_next_all, dim=1).max(dim=1).values   # (B,)
            y = rewards + self.cfg.gamma * q_next * (1.0 - dones)      # (B,)

        # ── TD loss ─────────────────────────────────────────────────────
        td_loss = F.mse_loss(q_sa, y)

        # ── Optional auxiliary dynamics loss ────────────────────────────
        aux_loss = torch.tensor(0.0, device=self.device)
        if self.aux_head is not None:
            e_actions_aux = self.action_embeddings(actions)             # (B, d_emb)
            pred_delta = self.aux_head(phi_s, e_actions_aux)            # (B, d_enc)
            with torch.no_grad():
                target_delta = phi_s_prime - phi_s.detach()             # (B, d_enc)
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
        }

    def handle_action_expansion(self, env) -> None:
        """Integrate a newly discovered action into the network.

        Called when env.action_space.n > self.action_count.

        1. k-NN initialization of the new embedding in the online AND target tables.
        2. Register new action index and set exploration bonus.
        3. Apply Phase 1 freeze (encoder + old embeddings + Q-head frozen).
        4. Rebuild optimizer with only trainable parameters.
        """
        print(
            f"[expansion] Action space growing from {self.action_count} → "
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

        # 3. Set exploration bonus
        self.action_bonuses[new_idx] = self.cfg.optimistic_init_bonus
        self.action_visit_counts[new_idx] = 0

        # 4. Phase 1 freeze
        self._apply_freeze_phase1()

        # 5. Rebuild optimizer
        self.optimizer = torch.optim.Adam(
            self._trainable_params(), lr=self.cfg.learning_rate
        )

    def maybe_update_freeze_schedule(self) -> None:
        """Progress through freeze phases based on steps since expansion."""
        steps_since = self.env_step - self.expansion_step

        if self.freeze_state == "phase1" and steps_since >= self.cfg.freeze_q_head_steps:
            self._apply_freeze_phase2()
            print(f"[expansion] Phase 1 → Phase 2 at env step {self.env_step}")

        if self.freeze_state == "phase2" and steps_since >= self.cfg.freeze_encoder_steps:
            self._apply_freeze_phase3()
            print(f"[expansion] Phase 2 → Phase 3 at env step {self.env_step}")

    def decay_exploration_bonuses(self) -> None:
        """Decay exploration bonuses for visited actions."""
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
                s_t = s_t.unsqueeze(0)
            phi_s = self.encoder(s_t)

            q_vals = []
            for a in range(self.action_count):
                e_a = self.action_embeddings(
                    torch.tensor([a], device=self.device)
                ).squeeze(0)
                q = self._compute_q(phi_s, e_a).item()
                q_vals.append(q)

        return np.array(q_vals)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_q(self, phi_s: torch.Tensor, e_a: torch.Tensor) -> torch.Tensor:
        """Compute Q(s, a) using the configured Q-function form."""
        if self.cfg.q_form == "factorized":
            return self.q_head(phi_s, e_a)                      # (B, 1)

        elif self.cfg.q_form == "dueling_factorized":
            V = self.value_head(phi_s)                          # (B, 1)
            A = self.q_head(phi_s, e_a)                         # (B, 1)

            # Center advantage over KNOWN actions only,
            # avoiding denominator shift from new actions
            with torch.no_grad():
                A_all = []
                for a in range(self.action_count):
                    e_all = self.action_embeddings(
                        torch.tensor([a], device=self.device)
                    ).squeeze(0)
                    A_all.append(self.q_head(phi_s, e_all))     # (B, 1) each
                A_all = torch.stack(A_all, dim=1)              # (B, |A|, 1)
                n_known = self.action_count - len(self.new_action_indices)
                A_known_mean = A_all[:, :n_known].mean(dim=1, keepdim=True)  # (B, 1)

            return V + A - A_known_mean                         # (B, 1)

        else:
            raise ValueError(f"Unknown q_form: {self.cfg.q_form}")

    def _compute_target_q(self, phi_s: torch.Tensor, e_a: torch.Tensor) -> torch.Tensor:
        """Compute target Q'(s, a).

        For the dueling variant, this includes V'(s) + A'(s,a).
        """
        if self.cfg.q_form == "factorized":
            return self.target_q_head(phi_s, e_a)                      # (B, 1)
        elif self.cfg.q_form == "dueling_factorized":
            V = self.target_value_head(phi_s)                          # (B, 1)
            A = self.target_q_head(phi_s, e_a)                         # (B, 1)
            return V + A                                               # (B, 1)
        else:
            raise ValueError(f"Unknown q_form: {self.cfg.q_form}")

    def _sync_target(self) -> None:
        """Hard copy online network → target network."""
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_action_embeddings.load_state_dict(
            self.action_embeddings.state_dict()
        )
        self.target_q_head.load_state_dict(self.q_head.state_dict())
        if self.target_value_head is not None and self.value_head is not None:
            self.target_value_head.load_state_dict(self.value_head.state_dict())

    def _apply_freeze_phase1(self) -> None:
        """Phase 1: Only the new action embedding is trainable.

        - Encoder frozen
        - Value head frozen (if dueling)
        - Q-head frozen
        - Old action embeddings frozen
        - New action embedding trainable
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

        # Freeze old action embeddings
        if self.cfg.freeze_old_embeddings:
            for idx in range(self.action_count):
                if idx not in self.new_action_indices:
                    self.action_embeddings.weight.data[idx].requires_grad = False

        # New embedding is already trainable by default (nn.Parameter)

    def _apply_freeze_phase2(self) -> None:
        """Phase 2: Unfreeze Q-head, keep encoder frozen."""
        self.freeze_state = "phase2"

        # Unfreeze Q-head
        for p in self.q_head.parameters():
            p.requires_grad = True

        # Unfreeze value head if dueling
        if self.value_head is not None:
            for p in self.value_head.parameters():
                p.requires_grad = True

    def _apply_freeze_phase3(self) -> None:
        """Phase 3: Unfreeze everything with reduced learning rate."""
        self.freeze_state = "phase3"

        # Unfreeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = True

        # Unfreeze all embeddings
        for idx in range(self.action_count):
            self.action_embeddings.weight.data[idx].requires_grad = True

        # Reduce learning rate
        for g in self.optimizer.param_groups:
            g["lr"] = self.cfg.learning_rate * 0.1

    def _trainable_params(self) -> list:
        """Return all parameters with requires_grad=True."""
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
