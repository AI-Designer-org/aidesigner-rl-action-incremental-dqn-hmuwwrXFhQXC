# Action Space Incremental Reinforcement Learning — Architecture Design

---

## Step 0 — Domain Identified

| Domain | Core design concerns addressed |
|---|---|
| **RL** | Observation/action interface, temporal discounting via TD learning, exploration strategy (ε-greedy + exploration bonus), off-policy replay for sample efficiency, action-space expansion without retraining |

This design also touches **Continual Learning** (catastrophic forgetting prevention via progressive unfreezing) and **Representation Learning** (action embedding space that supports zero-shot generalization).

---

## Step 1 — Research Contract Summary

The upstream research lifecycle contract is loaded from `/artifacts/j_hmuwwrXFhQXC/work/research/ACTION_SPACE_INCREMENTAL_RL.md`.

**Core hypothesis:**
> A factorized action-value architecture with a learnable action embedding space and a similarity-based Q-transfer mechanism can incorporate unexpectedly introduced actions into a deep RL policy without catastrophic forgetting and with sub-linear fine-tuning cost relative to retraining from scratch.

**Key claims to implement:**

| Claim | Status | Architecture obligation |
|---|---|---|
| No deep RL method handles unexpected action-space expansion mid-training | `TODO: unverified` | Produce a design that fills this gap |
| Similarity-based Q-transfer from known to new actions in embedding space provides better-than-random initialization | `hypothesis` | k-NN embedding interpolation + optimistic Q-initialization |
| Factorized Q(s,a) = f_Q(φ(s), e_a) minimizes interference between old and new action values | `hypothesis` | Factorized Q architecture with dueling variant as ablation |
| Freezing the base encoder during initial fine-tuning prevents catastrophic forgetting | `hypothesis` | Progressive unfreezing schedule |
| Softmax denominator shift from adding new actions causes measurable degradation | `hypothesis` | Mitigation: modular advantage head + post-expansion normalization |

**Evaluation requirements carried forward:**
- Custom Gymnasium environment with expanding action sets (3 scenarios: DAVE-game analogue, continuous control, goal-conditioned affordances)
- 10 random seeds per condition
- Metrics: cumulative reward, steps to recover, Q-value degradation, % steps saved vs. retrain

**Baseline requirements carried forward:**
1. Retrain from scratch
2. Fixed-action oracle (full set from beginning)
3. Zero-initialization expansion
4. GAS-adapted (Farquhar et al.) — post-hoc hierarchy if possible

---

## Step 2 — ModelConfig

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class ModelConfig:
    # ── Observation interface ───────────────────────────────────────────────
    obs_type: Literal["vector", "image"] = "vector"
    obs_dim: int = 64               # state vector dimensionality (vector obs)
    obs_channels: int = 3           # image channels (image obs)
    obs_height: int = 84            # image height (image obs)
    obs_width: int = 84             # image width (image obs)

    # ── Action space ────────────────────────────────────────────────────────
    n_actions_init: int = 3         # initial action set size (DAVE: {up, left, right})
    max_n_actions: int = 20         # maximum expected actions (pre-alloc not required)
    action_continuous: bool = False # discrete actions for DQN-style (primary)
                                    # set True for SAC-style (ablation / extension)

    # ── Encoder network ─────────────────────────────────────────────────────
    encoder_dim: int = 256          # output dimension of φ(s)
    d_model: int = 256              # internal hidden dim for encoder
    n_encoder_layers: int = 3       # MLP depth (vector obs); CNN depth (image obs)
    encoder_norm: bool = True       # LayerNorm after encoder output

    # ── Action embedding ────────────────────────────────────────────────────
    d_embedding: int = 64           # dimensionality of action embedding e_a
    embedding_init_scale: float = 0.1  # std for random embedding initialization
    k_nn: int = 3                   # k for k-NN embedding interpolation
    embedding_similarity: str = "cosine"  # "cosine" | "euclidean"
    use_auxiliary_dynamics_loss: bool = True  # InfoNCE-style aux loss for embeddings

    # ── Q-function (shared advantage head) ──────────────────────────────────
    q_hidden_dim: int = 128         # hidden dim of f_Q MLP
    q_n_layers: int = 2             # depth of f_Q MLP
    q_activation: str = "relu"      # activation for f_Q
    q_form: Literal["factorized", "dueling_factorized"] = "factorized"
        # "factorized":           Q(s,a) = f_Q(φ(s) || e_a)      — shared MLP
        # "dueling_factorized":   Q(s,a) = V(s) + A(s,a)         — value + advantage

    # ── RL algorithm ────────────────────────────────────────────────────────
    gamma: float = 0.99             # discount factor
    learning_rate: float = 3e-4     # Adam learning rate
    buffer_capacity: int = 100_000  # replay buffer size
    batch_size: int = 64            # training batch size
    target_update_freq: int = 1_000 # hard copy interval for target network
    target_tau: float = 1.0         # 1.0 = hard update; <1.0 = Polyak soft update
    gradient_steps: int = 1         # training updates per env step

    # ── Action expansion ────────────────────────────────────────────────────
    optimistic_init_bonus: float = 2.0   # β multiplier for optimistic Q bonus
    optimistic_bonus_decay: float = 0.99 # decay per visit of new action
    optimistic_bonus_min: float = 0.01   # minimum bonus after decay
    freeze_encoder_steps: int = 10_000   # steps to keep encoder frozen post-expansion
    freeze_old_embeddings: bool = True   # freeze existing action embeddings during fine-tune
    freeze_q_head_steps: int = 5_000     # steps to keep old advantage head frozen
    expansion_cooldown: int = 1_000      # min env steps between consecutive expansions

    # ── Exploration (ε-greedy) ──────────────────────────────────────────────
    epsilon_init: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay_steps: int = 100_000   # linear decay from init to min

    # ── Training ────────────────────────────────────────────────────────────
    dropout: float = 0.0
    use_bias: bool = True
    dtype: str = "float32"
    seed: int = 42

    # ── Logging / checkpointing ─────────────────────────────────────────────
    log_freq: int = 100
    eval_freq: int = 1_000
    checkpoint_freq: int = 10_000
```

---

## Step 3 — Architecture Overview

### 3.1 High-Level Architecture

```
         ┌─────────────────────────────────────────────────────────────┐
         │  Observation s                                             │
         │  (vector ℝᵈ or image ℝ³×ʰ×ʷ)                              │
         └─────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
         ┌─────────────────────────────────────────────────────────────┐
         │  State Encoder φ(s)                                        │
         │  • MLP (vector obs) or CNN (image obs)                    │
         │  • Output: ℝ^encoder_dim (256)                            │
         │  • LayerNorm on output                                     │
         └─────────────────────┬───────────────────────────────────────┘
                               │
                               │ φ(s) ∈ ℝ²⁵⁶
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                   │
             ▼                 ▼                   ▼
   ┌───────────────────┐   ┌───────────┐   ┌───────────────┐
   │  Value Head V(s)  │   │ Action    │   │ Auxiliary     │
   │  MLP: 256→128→1   │   │ Embedding │   │ Dynamics Head │
   │  (dueling variant)│   │ Table E   │   │ (optional)    │
   └─────────┬─────────┘   │ ℝ^{|A|×64}│   │ φ(s')-φ(s)   │
             │             └─────┬─────┘   │ prediction    │
             │                   │          └───────────────┘
             │                   │ e_a ∈ ℝ⁶⁴
             │                   ▼
             │         ┌──────────────────────┐
             │         │  Advantage / Q Head  │
             │         │  f_Q(φ(s) || e_a)   │
             │         │  MLP: 320→128→1      │
             │         └──────────┬───────────┘
             │                    │
             ▼                    ▼
     ┌───────────────────────────────┐
     │  Q(s, a) = V(s) + A(s,a)     │  (dueling variant)
     │  or Q(s, a) = f_Q(φ(s), e_a) │  (factorized variant)
     └───────────────────────────────┘
```

### 3.2 Action Expansion Flow

```
  ┌──────────┐     ┌──────────────┐     ┌─────────────────┐     ┌───────────────┐
  │ Detect   │────▶│ Create new   │────▶│ Initialize via  │────▶│ Freeze old    │
  │ new      │     │ embedding    │     │ k-NN in action  │     │ weights;      │
  │ action   │     │ row          │     │ embedding space │     │ add exploration│
  │ a_new    │     │              │     │                  │     │ bonus         │
  └──────────┘     └──────────────┘     └─────────────────┘     └───────┬───────┘
                                                                        │
                                                                        ▼
  ┌──────────┐     ┌──────────────┐     ┌─────────────────┐     ┌───────────────┐
  │ Evaluate │◀────│ Unfreeze all │◀────│ Unfreeze Q-head │◀────│ Train only    │
  │ metrics  │     │ (low LR)     │     │ (keep encoder   │     │ new embedding │
  │          │     │ Phase 3      │     │  frozen)        │     │ Phase 1       │
  │          │     │              │     │ Phase 2         │     │               │
  └──────────┘     └──────────────┘     └─────────────────┘     └───────────────┘
```

**Phase timeline (configurable via `freeze_encoder_steps`, `freeze_q_head_steps`):**

```
Timeline:  [ expansion detected ]
               │
               ▼
     ├───────────────────────┬───────────────────────┬───────────────────────┤
     │     Phase 1           │     Phase 2           │     Phase 3           │
     │  freeze_encoder       │  freeze_encoder       │  all unfrozen         │
     │  freeze_q_head        │  unfreeze_q_head      │  lr × 0.1             │
     │  train: new e_a only   │  train: e_a + Q-head  │  train: everything    │
     │  0 ───── N₁ steps     │  N₁ ───── N₂ steps    │  N₂ ───── ∞          │
```

---

## Step 4 — Core Components Pseudocode

### 4.1 Action-Incremental DQN Agent

```python
class ActionIncrementalDQN:
    """Main agent with action-space expansion capability."""

    def __init__(self, cfg: ModelConfig, n_actions: int):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Core networks
        self.encoder = StateEncoder(cfg)           # φ(s)
        self.value_head = ValueHead(cfg)           # V(s) — used only in dueling variant
        self.action_embeddings = ActionEmbeddingTable(n_actions, cfg)  # E
        self.q_head = QHead(cfg)                   # f_Q(φ(s) || e_a)
        self.aux_head = DynamicsHead(cfg)          # optional: predict φ(s') - φ(s)

        # Target network (periodic hard copy)
        self.target_encoder = StateEncoder(cfg)
        self.target_value_head = ValueHead(cfg)
        self.target_action_embeddings = ActionEmbeddingTable(n_actions, cfg)
        self.target_q_head = QHead(cfg)
        self._sync_target()

        # Optimizer — only parameters that are currently unfrozen
        self.optimizer = torch.optim.Adam(
            self._trainable_params(), lr=cfg.learning_rate
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(cfg.buffer_capacity)

        # Expansion state
        self.action_count = n_actions          # current |A|
        self.expansion_step = 0                # env step when last expansion occurred
        self.freeze_state = "none"             # "none" | "phase1" | "phase2" | "phase3"
        self.new_action_indices = set()        # indices of actions added post-init

        # Exploration bonus tracking
        self.action_visit_counts = torch.zeros(cfg.max_n_actions, dtype=torch.long)
        self.action_bonuses = {}               # {action_idx: current_bonus_value}

        # Step counter
        self.env_step = 0

    def select_action(self, s: torch.Tensor, epsilon: float) -> int:
        """ε-greedy with optimistic exploration bonus for new actions."""
        if random.random() < epsilon:
            return random.randint(0, self.action_count - 1)

        with torch.no_grad():
            phi_s = self.encoder(s.unsqueeze(0))                    # (1, d_enc)
            q_values = []

            for a in range(self.action_count):
                e_a = self.action_embeddings(a)                     # (d_emb,)
                q = self.q_head(phi_s, e_a)                         # scalar
                bonus = self.action_bonuses.get(a, 0.0)
                q_values.append(q.item() + bonus)

        return int(np.argmax(q_values))

    def update(self, batch: TransitionBatch) -> dict:
        """Single TD-learning step. Handles variable-sized action sets."""
        states, actions, rewards, next_states, dones = batch

        # ── Online Q(s, a) ──────────────────────────────────────────
        phi_s = self.encoder(states)                                 # (B, d_enc)

        # Gather embeddings for the taken actions
        e_actions = self.action_embeddings(actions)                  # (B, d_emb)
        q_sa = self.q_head(phi_s, e_actions).squeeze(-1)             # (B,)

        # ── Target: y = r + γ * max_{a'} Q'(s', a') ────────────────
        with torch.no_grad():
            phi_s_prime = self.target_encoder(next_states)           # (B, d_enc)

            # Compute Q'(s', a') for ALL currently known actions
            # This naturally includes any newly-added actions
            q_next_all = []
            for a in range(self.action_count):
                e_a_prime = self.target_action_embeddings(a)         # (d_emb,) broadcast
                q_a = self.target_q_head(phi_s_prime, e_a_prime)     # (B, 1)
                q_next_all.append(q_a)
            q_next = torch.stack(q_next_all, dim=1).max(dim=1).values  # (B,) max over actions
            y = rewards + self.cfg.gamma * q_next * (1.0 - dones)    # (B,)

        # ── Loss ────────────────────────────────────────────────────
        td_loss = F.mse_loss(q_sa, y)

        # ── Optional auxiliary dynamics loss ─────────────────────────
        aux_loss = torch.tensor(0.0)
        if self.cfg.use_auxiliary_dynamics_loss:
            e_a_for_aux = self.action_embeddings(actions)
            pred_delta = self.aux_head(phi_s, e_a_for_aux)           # predict φ(s') - φ(s)
            with torch.no_grad():
                target_delta = phi_s_prime - phi_s
            aux_loss = F.mse_loss(pred_delta, target_delta)

        total_loss = td_loss + 0.01 * aux_loss

        # ── Optimize ────────────────────────────────────────────────
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._trainable_params(), 10.0)
        self.optimizer.step()

        return {"td_loss": td_loss.item(), "aux_loss": aux_loss.item()}

    def handle_action_expansion(self, env) -> None:
        """Called when env.action_space has grown. Integrates new action."""
        print(f"[expansion] Action space growing from {self.action_count} → "
              f"{self.action_count + 1} at env step {self.env_step}")

        # 1. Create new embedding via k-NN interpolation
        self.action_embeddings.add_embedding(
            n_known=self.action_count,
            k=self.cfg.k_nn,
            similarity=self.cfg.embedding_similarity,
        )
        self.target_action_embeddings.add_embedding(
            n_known=self.action_count,   # same known count at time of expansion
            k=self.cfg.k_nn,
            similarity=self.cfg.embedding_similarity,
            source_embeddings=self.action_embeddings.weight.data,  # copy from online
        )

        # 2. Register new action index
        new_idx = self.action_count
        self.new_action_indices.add(new_idx)
        self.action_count += 1
        self.expansion_step = self.env_step

        # 3. Set exploration bonus
        self.action_bonuses[new_idx] = self.cfg.optimistic_init_bonus
        self.action_visit_counts[new_idx] = 0

        # 4. Apply freeze schedule (Phase 1)
        self._apply_freeze_phase1()

        # 5. Rebuild optimizer with new parameter set
        self.optimizer = torch.optim.Adam(
            self._trainable_params(), lr=self.cfg.learning_rate
        )

    def _apply_freeze_phase1(self):
        """Phase 1: Only new action embedding is trainable."""
        self.freeze_state = "phase1"

        # Freeze encoder entirely
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Freeze value head (if dueling variant)
        for p in self.value_head.parameters():
            p.requires_grad = False

        # Freeze old action embeddings
        if self.cfg.freeze_old_embeddings:
            for idx in range(self.action_count):
                if idx not in self.new_action_indices:
                    self.action_embeddings.weight.data[idx].requires_grad = False

        # Freeze Q-head entirely (Phase 1)
        for p in self.q_head.parameters():
            p.requires_grad = False

        # Only new embedding is trainable (set in action_embeddings)

    def _apply_freeze_phase2(self):
        """Phase 2: Unfreeze Q-head, keep encoder frozen."""
        self.freeze_state = "phase2"
        for p in self.q_head.parameters():
            p.requires_grad = True

    def _apply_freeze_phase3(self):
        """Phase 3: Unfreeze everything with reduced LR."""
        self.freeze_state = "phase3"
        for p in self.encoder.parameters():
            p.requires_grad = True
        for p in self.value_head.parameters():
            p.requires_grad = True
        # Unfreeze all embeddings
        for idx in range(self.action_count):
            self.action_embeddings.weight.data[idx].requires_grad = True

        # Reduce learning rate
        for g in self.optimizer.param_groups:
            g["lr"] = self.cfg.learning_rate * 0.1

    def _trainable_params(self) -> list:
        """Returns parameters with requires_grad=True."""
        params = []
        for module in [self.encoder, self.value_head,
                       self.action_embeddings, self.q_head, self.aux_head]:
            params.extend([p for p in module.parameters() if p.requires_grad])
        return params

    def maybe_update_freeze_schedule(self) -> None:
        """Progresses through freeze phases based on env step count."""
        steps_since_expansion = self.env_step - self.expansion_step

        if self.freeze_state == "phase1" and steps_since_expansion >= self.cfg.freeze_q_head_steps:
            self._apply_freeze_phase2()
            print(f"[expansion] Phase 1 → Phase 2 at env step {self.env_step}")

        if self.freeze_state == "phase2" and steps_since_expansion >= self.cfg.freeze_encoder_steps:
            self._apply_freeze_phase3()
            print(f"[expansion] Phase 2 → Phase 3 at env step {self.env_step}")

    def decay_exploration_bonuses(self) -> None:
        """Decay exploration bonuses for visited new actions."""
        for idx in self.new_action_indices:
            if self.action_visit_counts[idx] > 0:
                self.action_bonuses[idx] = max(
                    self.cfg.optimistic_bonus_min,
                    self.action_bonuses[idx] * self.cfg.optimistic_bonus_decay
                )
```

### 4.2 Action Embedding Table

```python
class ActionEmbeddingTable(nn.Module):
    """Learnable embedding table E ∈ ℝ^{|A| × d_emb}.

    Supports dynamic expansion: adding new rows mid-training without
    disrupting existing embeddings.
    """

    def __init__(self, n_actions: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(
            torch.randn(n_actions, cfg.d_embedding) * cfg.embedding_init_scale
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Lookup embeddings for given action indices.

        Args:
            indices: (B,) long tensor of action indices
        Returns:
            (B, d_emb) embeddings
        """
        return F.embedding(indices, self.weight)

    @torch.no_grad()
    def add_embedding(self, n_known: int, k: int = 3,
                      similarity: str = "cosine",
                      source_embeddings: torch.Tensor = None) -> None:
        """Append a new embedding row initialized via k-NN interpolation.

        Args:
            n_known: Number of existing (known) action embeddings.
            k: Number of nearest neighbors to interpolate.
            similarity: "cosine" or "euclidean".
            source_embeddings: Source table to copy from (for target network sync).
                               If None, uses self.weight.
        """
        source = source_embeddings if source_embeddings is not None else self.weight.data
        existing = source[:n_known]                           # (n_known, d_emb)

        # Compute pairwise similarities of existing embeddings to find
        # a good starting point. We use the centroid of existing embeddings
        # as a proxy query (since we have no semantic descriptor of a_new).
        # In a more advanced version, we could use an action description
        # embedding if available (e.g., from a language model).
        centroid = existing.mean(dim=0, keepdim=True)          # (1, d_emb)

        if similarity == "cosine":
            sim = F.cosine_similarity(centroid, existing, dim=1)  # (n_known,)
            _, idx = torch.topk(sim, min(k, n_known))
        elif similarity == "euclidean":
            dist = torch.cdist(centroid, existing).squeeze(0)     # (n_known,)
            _, idx = torch.topk(dist, min(k, n_known), largest=False)
        else:
            raise ValueError(f"Unknown similarity: {similarity}")

        e_new = existing[idx].mean(dim=0)                     # (d_emb,)

        # Append new row
        new_weight = torch.cat([self.weight.data, e_new.unsqueeze(0)], dim=0)
        self.weight = nn.Parameter(new_weight)
```

### 4.3 Encoder and Q-Head

```python
class StateEncoder(nn.Module):
    """State encoder φ(s). Configurable for vector or image observations."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.obs_type == "vector":
            # MLP encoder
            layers = []
            in_dim = cfg.obs_dim
            for _ in range(cfg.n_encoder_layers):
                layers.extend([
                    nn.Linear(in_dim, cfg.d_model),
                    nn.ReLU(),
                ])
                in_dim = cfg.d_model
            layers.append(nn.Linear(cfg.d_model, cfg.encoder_dim))
            self.net = nn.Sequential(*layers)
        elif cfg.obs_type == "image":
            # CNN encoder (standard DQN Atari architecture)
            self.net = nn.Sequential(
                nn.Conv2d(cfg.obs_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(self._cnn_out_dim(cfg), cfg.encoder_dim),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Unknown obs_type: {cfg.obs_type}")

        self.norm = nn.LayerNorm(cfg.encoder_dim) if cfg.encoder_norm else nn.Identity()

    def _cnn_out_dim(self, cfg) -> int:
        """Compute CNN output dimension from input size."""
        dummy = torch.zeros(1, cfg.obs_channels, cfg.obs_height, cfg.obs_width)
        out = self.net[:6](dummy)  # pass through conv layers only
        return out.view(1, -1).size(1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(s))


class QHead(nn.Module):
    """Shared Q-function f_Q(φ(s), e_a).

    Takes concatenated [state_representation; action_embedding] and
    outputs a scalar Q-value.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.encoder_dim + cfg.d_embedding

        layers = []
        dim = in_dim
        for _ in range(cfg.q_n_layers):
            layers.append(nn.Linear(dim, cfg.q_hidden_dim))
            if cfg.q_activation == "relu":
                layers.append(nn.ReLU())
            elif cfg.q_activation == "gelu":
                layers.append(nn.GELU())
            layers.append(nn.Dropout(cfg.dropout))
            dim = cfg.q_hidden_dim
        layers.append(nn.Linear(cfg.q_hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, phi_s: torch.Tensor, e_a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phi_s: (B, d_enc) or (1, d_enc) — state representation
            e_a:   (d_emb,) — single action embedding (broadcasted)
                   or (B, d_emb) — batched action embeddings
        Returns:
            (B, 1) Q-values
        """
        # Broadcast e_a if single action queried for all states in batch
        if e_a.dim() == 1:
            e_a = e_a.unsqueeze(0).expand(phi_s.size(0), -1)

        x = torch.cat([phi_s, e_a], dim=-1)
        return self.net(x)


class ValueHead(nn.Module):
    """State value head V(s) — used in dueling variant.

    Q(s,a) = V(s) + A(s,a)  where A(s,a) = f_Q(φ(s), e_a)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.encoder_dim, cfg.q_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.q_hidden_dim, 1),
        )

    def forward(self, phi_s: torch.Tensor) -> torch.Tensor:
        return self.net(phi_s)  # (B, 1)
```

### 4.4 Training Loop

```python
def train_action_incremental_dqn(cfg: ModelConfig, env_fn, n_episodes: int):
    """Main training loop with action-space expansion detection."""

    # Initial environment (DAVE level 1: actions {up, left, right})
    env = env_fn(level=1)
    agent = ActionIncrementalDQN(cfg, n_actions=env.action_space.n)

    epsilon = cfg.epsilon_init
    epsilon_delta = (cfg.epsilon_init - cfg.epsilon_min) / cfg.epsilon_decay_steps

    for episode in range(n_episodes):
        s, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # ── Check for action-space expansion ──────────────────────────
            if env.action_space.n > agent.action_count:
                # Cooldown check: prevent rapid successive expansions
                steps_since = agent.env_step - agent.expansion_step
                if steps_since >= cfg.expansion_cooldown:
                    agent.handle_action_expansion(env)

            # ── Action selection ───────────────────────────────────────────
            a = agent.select_action(torch.tensor(s, dtype=torch.float32), epsilon)

            # Track visit counts for exploration bonus decay
            if a in agent.new_action_indices:
                agent.action_visit_counts[a] += 1

            s_next, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            episode_reward += r

            # ── Store transition ──────────────────────────────────────────
            agent.replay_buffer.push(s, a, r, s_next, done)

            s = s_next
            agent.env_step += 1

            # ── Training step ─────────────────────────────────────────────
            if len(agent.replay_buffer) >= cfg.batch_size:
                for _ in range(cfg.gradient_steps):
                    batch = agent.replay_buffer.sample(cfg.batch_size)
                    metrics = agent.update(batch)

                # Target network update
                if agent.env_step % cfg.target_update_freq == 0:
                    agent._sync_target()

                # Freeze schedule progression
                agent.maybe_update_freeze_schedule()

                # Decay exploration bonuses
                agent.decay_exploration_bonuses()

            # ── Epsilon decay ─────────────────────────────────────────────
            epsilon = max(cfg.epsilon_min, epsilon - epsilon_delta)

        # ── Logging ──────────────────────────────────────────────────────
        if episode % cfg.log_freq == 0:
            print(f"Ep {episode:5d} | Steps {agent.env_step:7d} | "
                  f"Reward {episode_reward:6.1f} | ε {epsilon:.3f} | "
                  f"|A|={agent.action_count} | freeze={agent.freeze_state}")
```

### 4.5 Dueling Factorized Variant (Ablation)

```python
class DuelingFactorizedQ(nn.Module):
    """Dueling variant: Q(s,a) = V(s) + A(s,a).

    A(s,a) = f_adv(φ(s) || e_a)  — separate advantage head.

    Key benefit: V(s) is unaffected by action-space expansion.
    Only the advantage residuals for new actions need to be learned.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.value_head = ValueHead(cfg)
        self.advantage_head = QHead(cfg)  # reuses the same MLP architecture

    def forward(self, phi_s: torch.Tensor, e_a: torch.Tensor,
                all_embeddings: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            phi_s: (B, d_enc) state representation
            e_a: (d_emb,) or (B, d_emb) action embedding
            all_embeddings: (|A|, d_emb) — all action embeddings for centering.
                            If None, no centering (avoids denominator shift).
        """
        V = self.value_head(phi_s)                                 # (B, 1)
        A = self.advantage_head(phi_s, e_a)                        # (B, 1)

        if all_embeddings is not None:
            # Centering over KNOWN actions only (pre-expansion set)
            # avoids denominator shift from new action
            with torch.no_grad():
                known_mask = all_embeddings.shape[0]  # use first N as "known"
                # Compute A for all actions
                A_all = []
                for i in range(all_embeddings.shape[0]):
                    A_all.append(self.advantage_head(phi_s, all_embeddings[i]))
                A_all = torch.stack(A_all, dim=1)       # (B, |A|, 1)
                A_known_mean = A_all[:, :known_mask].mean(dim=1)  # (B, 1)
            A = A - A_known_mean

        return V + A
```

---

## Step 5 — ASCII Architecture Diagram

### Complete Forward Pass

```
                        ┌──────────┐
                        │ State s  │
                        └────┬─────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  φ(s) Encoder  │  State representation
                    │  (MLP / CNN)   │
                    └───────┬────────┘
                            │ φ(s) ∈ ℝ²⁵⁶
                            │
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                   ▼
   ┌────────────┐   ┌──────────────┐   ┌──────────────┐
   │ Action idx │   │ Action       │   │ (Optional)   │
   │ a          │──▶│ Embedding    │   │ Dynamics     │
   └────────────┘   │ Table E      │   │ Head         │
                    │ Lookup       │   │ φ(s')-φ(s)   │
                    └──────┬───────┘   │ prediction   │
                           │ e_a       └──────────────┘
                           │ ∈ ℝ⁶⁴
                           ▼
                    ┌──────────────┐
                    │ Concat       │
                    │ [φ(s); e_a]  │  ∈ ℝ³²⁰
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │ Q-Head MLP   │  f_Q shared across all actions
                    │ 320→128→1   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Q(s, a)     │  ∈ ℝ
                    └──────────────┘
```

### Training Architecture

```
┌──────────────────── Online Network ────────────────────┐
│  φ(s) → [φ(s); e_a] → f_Q → Q(s,a)                    │
│                                                        │
│  Replay Buffer: (s, a, r, s', done, |A|_t)            │
│       │                                                │
│       ▼ TD error                                       │
│  y = r + γ · max_{a'} Q'(s', a')                       │
│       ▲                                                │
└───────┼────────────────────────────────────────────────┘
        │
┌───────┼─────────── Target Network ─────────────────────┐
│       │   φ'(s') → [φ'(s'); e'_a'] → f_Q' → Q'(s',a')  │
│       │                                                 │
│       │   Copied every N steps (hard) or Polyak (soft)  │
└───────┴─────────────────────────────────────────────────┘
```

### Replay Buffer Handling with Expanding Action Space

```
Before expansion (|A| = 3):   After expansion (|A| = 4):
┌─────────────────────┐      ┌─────────────────────┐
│ (s, a=0, r, s', d)  │      │ (s, a=0, r, s', d)  │  ← old (still valid)
│ (s, a=2, r, s', d)  │      │ (s, a=2, r, s', d)  │  ← old
│ (s, a=1, r, s', d)  │      │ (s, a=1, r, s', d)  │  ← old
│ ...                  │      │ (s, a=3, r, s', d)  │  ← new (action index 3)
└─────────────────────┘      │ (s, a=0, r, s', d)  │  ← new
                              └─────────────────────┘

All transitions remain valid: action indices are stable (old actions keep
their indices; new actions get new indices). The TD target naturally
includes the new action in max_{a'} Q'(s', a') for all transitions.
```

---

## Step 6 — Inductive Bias Justification

| Design choice | One-sentence justification |
|---|---|
| **Factorized Q(s,a) = f_Q(φ(s), e_a)** | Decouples action identity from the Q-function, enabling zero-shot Q-values for unseen actions via embedding similarity. |
| **Action embedding table** | Discrete actions have no natural feature vector; a learned embedding per action allows the Q-function to exploit similarity structure. |
| **k-NN interpolation for new embedding** | Initializes new actions at the centroid of semantically similar known actions, providing a structural prior that reduces cold-start error. |
| **k-NN centroid of ALL known embeddings** | Without semantic descriptors of the new action, the centroid of ALL existing embeddings is the least-committal initialization (minimax optimal). |
| **Optimistic exploration bonus for new actions** | Counteracts learned pessimism (new actions start with zero visit count) and encourages systematic exploration of the new action. |
| **ε-greedy exploration** | Simple, well-understood, compatible with discrete action spaces; the exploration bonus provides directed exploration on top of uniform randomness. |
| **Progressive freeze schedule (Phase 1 → 3)** | Prevents catastrophic forgetting by isolating the new action's parameters during initial learning, then gradually re-introducing plasticity. |
| **Freeze encoder first** | State representations learned on old actions transfer perfectly to new actions (the state space hasn't changed); no need to disrupt them. |
| **Off-policy replay (DQN)** | Maximally sample-efficient; old transitions remain usable after expansion because action indices are stable and the TD target naturally includes new actions. |
| **Optional auxiliary dynamics loss** | Shapes the action embedding space to encode transition dynamics (which actions cause which state changes), creating a semantically meaningful manifold. |
| **Pre-norm (LayerNorm on φ(s))** | Stabilizes the gradient flow through the shared encoder, especially important when training dynamics change during expansion phases. |
| **Separate value head (dueling variant)** | V(s) is completely unaffected by action-space expansion; only the relative advantages of actions need to be adjusted for the new action. |

---

## Step 7 — Research-to-Architecture Traceability

| Research contract item | Architecture decision | Evidence status | Validation hook |
|---|---|---|---|
| **Novelty claim:** No deep RL method handles unexpected action-space expansion | `ActionIncrementalDQN` class with `handle_action_expansion()` dynamic table growth | `TODO: unverified` | Benchmark: DAVE-game analogue. Metric: Can the agent use a_new within N steps of expansion? |
| **Hypothesis:** Similarity-based Q-transfer provides better-than-random initialization | `ActionEmbeddingTable.add_embedding()` with k-NN interpolation | `hypothesis` | Ablation: Compare k-NN init vs. random vs. zero init for new action Q-values at first encounter |
| **Hypothesis:** Factorized Q minimizes interference | `QHead` shared MLP operating on `[φ(s); e_a]` concatenation | `hypothesis` | Ablation: Compare factorized vs. independent-head architecture on old-action Q degradation |
| **Hypothesis:** Freezing encoder prevents catastrophic forgetting | `_apply_freeze_phase1()` — encoder frozen for `freeze_encoder_steps` | `hypothesis` | Compare freeze vs. no-freeze: Q-value degradation on old actions after expansion |
| **Hypothesis:** Softmax denominator shift degrades old action logits | Dueling variant with known-action-only centering (`DuelingFactorizedQ`) | `hypothesis` | Measure old-action logit shift before/after expansion with and without centering |
| **Gap:** Action embedding structure for zero-shot transfer | `use_auxiliary_dynamics_loss` — dynamics head predicts φ(s') - φ(s) from e_a | `hypothesis` | Compare t-SNE of embeddings with vs. without aux loss; measure k-NN init quality |
| **Baseline requirement:** Retrain from scratch | Separate `DQNBaseline` wrapper with fixed action set | `grounded` — standard practice | Compare total env steps to reach equivalent performance |
| **Baseline requirement:** Zero-initialization | Ablation: set `k_nn=0` to use random embedding init | `grounded` | Ablation flag in config |
| **Evaluation requirement:** 10 seeds, mean ± std | `seed` field in config; training loop logs seed | `grounded` | CI check: `assert len(results.seeds) >= 10` |
| **Evaluation requirement:** Q-value degradation metric | Log `q_values` for fixed state-action pairs before/after expansion | `grounded` | Assert: `ΔQ_old < 0.2 * Q_old_pre_expansion` |
| **Blocking unknown:** Embedding structure for semantically novel actions | `embedding_similarity`, `k_nn`, and centroid-based init | `TODO: unverified` | If k-NN init ≈ random init performance, core hypothesis is falsified |

---

## Step 8 — Domain-Specific Considerations (RL)

### Temporal Discounting

γ = 0.99 enters through the standard TD target:
```
y = r + γ · max_{a'} Q'(s', a')
```

The architecture does not modify how discounting works. The key subtlety: after expansion, the TD target for *old* transitions now includes the new action in the max. This is correct because the value of s' genuinely increases when a new action becomes available (monotonicity property of nested action spaces).

### Exploration

Two complementary mechanisms:
1. **ε-greedy** (uniform random): Provides undirected exploration. ε decays linearly from 1.0 to 0.05 over `epsilon_decay_steps`. After each expansion, ε is temporarily **reset to max(ε_current, 0.5)** to encourage exploration of the new action.
2. **Optimistic exploration bonus**: A multiplicative bonus for new actions that decays with each visit. This provides directed exploration toward the new action without relying solely on ε-greedy randomness.

### On-policy vs. Off-policy

**Off-policy (DQN)** is the correct choice for this problem:
- Old transitions remain valid after action-space expansion (action indices are stable)
- Replay buffer contains experiences from before the new action was discovered
- Sample efficiency matters — the goal is to avoid retraining from scratch
- The factorized Q-function can evaluate Q(s', a_new) even for transitions recorded before a_new existed

**Constraint**: The replay buffer must handle variable-sized action sets during TD target computation. Our solution: compute `max_{a'} Q'(s', a')` over all currently known actions (which may be more than what was available when the transition was recorded). This is justified by the monotonicity property.

### Observation Modality

- **Vector observations** (default): Simple MLP encoder suitable for low-dimensional state spaces (DAVE-game, Gridworld)
- **Image observations** (configurable via `obs_type="image"`): CNN encoder following DQN/Atari architecture

The architecture is agnostic to observation type — the encoder outputs a fixed-dimension representation φ(s) regardless.

### Catastrophic Forgetting Prevention

Three mechanisms work together:
1. **Architectural**: Factorized Q decouples action-specific parameters. Old action embeddings are frozen during Phase 1-2.
2. **Regularization**: Replay buffer naturally rehearses old transitions, preventing the network from drifting.
3. **Optimization**: Progressive unfreezing with reduced LR in Phase 3 prevents large gradient updates from disrupting learned representations.

---

## Step 9 — Implementation Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **1. Embedding space collapse**: All action embeddings converge to similar values, making k-NN interpolation uninformative | Medium | High (core hypothesis fails) | Auxiliary dynamics loss; explicit embedding diversity loss (cosine separation penalty); monitor embedding similarity matrix |
| **2. Optimistic bonus interference**: The exploration bonus causes Q-value overestimation that propagates through TD backups, corrupting old action values | Medium | High | Clamp bonus to max Q; use double-DQN (separate action selection and evaluation); decay bonus aggressively |
| **3. Target network desynchronization**: After expansion, target network has the new embedding but online network changes it rapidly, creating drift | Low | Medium | Hard sync target immediately after expansion; increase `target_update_freq` temporarily |
| **4. Replay buffer staleness**: Old transitions dominate the buffer post-expansion, slowing adaptation to the new action | Low | Medium | Prioritized replay: boost transitions involving the new action; oversample new-action transitions |
| **5. Gradient explosion during Phase 3 transition**: Unfreezing the encoder after it has been frozen for N steps produces a sudden gradient surge | Medium | Low | Gradient clipping (max norm 10.0); warm up LR over 100 steps after Phase 3 entry |

---

## Step 10 — Suggested Ablations

All ablations are expressible as single-field `ModelConfig` changes.

| # | Ablation | Config field | Baseline value | Ablated value | Hypothesis tested | Expected metric movement | Failure interpretation | Owning stage |
|---|---|---|---|---|---|---|---|---|
| 1 | **k-NN embedding init → random init** | `k_nn` | 3 | 0 | k-NN interpolation provides better zero-shot Q for new actions than random init | Q(a_new) at first encounter: k-NN > random | Embedding space lacks semantic structure; core hypothesis falsified | `ml-research` |
| 2 | **Optimistic bonus → no bonus** | `optimistic_init_bonus` | 2.0 | 0.0 | Optimistic bonus accelerates exploration of new actions | Steps to first successful use of a_new: bonus < no-bonus | Directed exploration is unimportant; ε-greedy suffices | `ml-architect` |
| 3 | **Freeze encoder → no freeze** | `freeze_encoder_steps` | 10000 | 0 | Freezing encoder prevents forgetting during fine-tuning | Old-action Q degradation: freeze < no-freeze | Forgetting is not a practical problem; or alternative mechanisms suffice | `ml-architect` |
| 4 | **Factorized Q → independent head** | `q_form` | `"factorized"` | Use separate `Linear(encoder_dim, 1)` per action | Factorized architecture enables better transfer than independent heads | Steps to recover performance: factorized < independent | Shared Q-function doesn't help; or action embeddings don't generalize | `ml-research` |
| 5 | **Factorized Q → dueling factorized** | `q_form` | `"factorized"` | `"dueling_factorized"` | Dueling decomposition further reduces interference from new actions | Old-action Q degradation: dueling < factorized | Value-advantage separation is unnecessary; additive bias is approximation error dominated | `ml-architect` |
| 6 | **Auxiliary dynamics loss → no aux loss** | `use_auxiliary_dynamics_loss` | True | False | Dynamics-aware embeddings create better semantic structure for transfer | k-NN init quality: with-aux > without-aux | Embedding structure from TD learning alone is sufficient | `ml-coder` |
| 7 | **Soft target update → hard target** | `target_tau` | 1.0 | 0.005 | Polyak averaging smooths the transition after expansion | Q-value variance post-expansion: Polyak < hard | Target network stability is not the bottleneck | `ml-coder` |
| 8 | **Old embedding freeze → no freeze** | `freeze_old_embeddings` | True | False | Freezing old action embeddings prevents interference during new-action fine-tuning | Old-action Q degradation with vs. without freeze | Old action values are stable even without explicit freeze | `ml-architect` |

### Ablation ordering

**Tier 1 — Turn these off first if the method doesn't work at all:**
1. k-NN embedding init → random (tests whether embedding space has structure)
2. Optimistic bonus → no bonus (tests whether exploration is sufficient)

**Tier 2 — If performance is marginal, try these:**
3. Freeze encoder → no freeze (tests forgetting prevention)
4. Factorized Q → independent head (tests architectural hypothesis)

**Tier 3 — Refinements:**
5. Factorized Q → dueling factorized (tests if value/advantage separation helps)
6. Auxiliary dynamics loss → no aux loss (tests embedding quality)
7. Soft target → hard target (tests training stability)
8. Old embedding freeze → no freeze (tests parameter isolation)

---

## Step 11 — Output Checklist

- [x] Domain identified — **RL (Action Space Incremental RL)**
- [x] Upstream research lifecycle contract read — `ACTION_SPACE_INCREMENTAL_RL.md` loaded
- [x] ModelConfig dataclass with all hyperparameters — Section 2
- [x] Pseudocode for the novel block — Section 4 (ActionIncrementalDQN, ActionEmbeddingTable, QHead)
- [x] ASCII architecture diagram — Section 5
- [x] Inductive bias justification — Section 6
- [x] Research-to-architecture traceability table — Section 7
- [x] Claims labeled as `grounded`, `hypothesis`, or `TODO: unverified` — throughout
- [x] Domain-specific considerations addressed — Section 8
- [x] Implementation risk flags — Section 9
- [x] Baseline and evaluation requirements carried forward — Section 1
- [x] Suggested ablations — Section 10

---

## Appendix A: File Manifest

| File | Purpose |
|---|---|
| `ACTION_INCREMENTAL_RL_ARCHITECTURE.md` | This document — full architecture design |
| `model_config.py` | Standalone `ModelConfig` dataclass |
| `components.py` | `StateEncoder`, `QHead`, `ValueHead`, `ActionEmbeddingTable` |
| `agent.py` | `ActionIncrementalDQN` agent class |
| `train.py` | Training loop with action-expansion detection |
