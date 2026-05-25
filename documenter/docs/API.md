# API Reference

## `model_config.py`

### `class ModelConfig`
Dataclass containing all hyperparameters for the Action-Incremental DQN agent. Every architectural choice, RL hyperparameter, and expansion protocol setting is controlled through this single config object.

**Fields:**

| Field | Type | Default | Rationale |
|---|---|---|---|
| `obs_type` | `Literal["vector", "image"]` | `"vector"` | Observation modality |
| `obs_dim` | `int` | `64` | State vector dimensionality |
| `obs_channels` | `int` | `3` | Image channels (image obs) |
| `obs_height` | `int` | `84` | Image height |
| `obs_width` | `int` | `84` | Image width |
| `n_actions_init` | `int` | `3` | Initial action set size (DAVE: {up, left, right}) |
| `max_n_actions` | `int` | `20` | Maximum actions (pre-allocation cap for tracking tensors) |
| `action_continuous` | `bool` | `False` | Discrete actions (DQN-style); set True for SAC extensions |
| `encoder_dim` | `int` | `256` | Output dimension of φ(s) |
| `d_model` | `int` | `256` | Internal hidden dimension for encoder MLP |
| `n_encoder_layers` | `int` | `3` | Encoder depth (MLP layers for vector; CNN layers for image) |
| `encoder_norm` | `bool` | `True` | LayerNorm after encoder output |
| `d_embedding` | `int` | `64` | Action embedding dimensionality |
| `embedding_init_scale` | `float` | `0.1` | Std for random embedding initialisation |
| `k_nn` | `int` | `3` | k for k-NN embedding interpolation |
| `embedding_similarity` | `Literal["cosine", "euclidean"]` | `"cosine"` | Similarity metric for k-NN |
| `use_auxiliary_dynamics_loss` | `bool` | `True` | Enable dynamics prediction auxiliary loss |
| `q_hidden_dim` | `int` | `128` | Hidden dim of Q-head MLP |
| `q_n_layers` | `int` | `2` | Depth of Q-head MLP |
| `q_activation` | `Literal["relu", "gelu"]` | `"relu"` | Activation for Q-head |
| `q_form` | `Literal["factorized", "dueling_factorized"]` | `"factorized"` | Q-function architecture variant |
| `gamma` | `float` | `0.99` | Discount factor |
| `learning_rate` | `float` | `3e-4` | Adam learning rate |
| `buffer_capacity` | `int` | `100_000` | Replay buffer size |
| `batch_size` | `int` | `64` | Training batch size |
| `target_update_freq` | `int` | `1_000` | Hard copy interval for target network |
| `target_tau` | `float` | `1.0` | 1.0 = hard update; <1.0 = Polyak soft update |
| `gradient_steps` | `int` | `1` | Training updates per env step |
| `optimistic_init_bonus` | `float` | `2.0` | Additive Q-bonus for new actions |
| `optimistic_bonus_decay` | `float` | `0.99` | Decay per visit of new action |
| `optimistic_bonus_min` | `float` | `0.01` | Minimum bonus after decay |
| `freeze_encoder_steps` | `int` | `10_000` | Steps to keep encoder frozen post-expansion |
| `freeze_old_embeddings` | `bool` | `True` | Freeze existing action embeddings during fine-tune |
| `freeze_q_head_steps` | `int` | `5_000` | Steps to keep old advantage head frozen |
| `expansion_cooldown` | `int` | `1_000` | Min env steps between expansions |
| `epsilon_init` | `float` | `1.0` | Initial ε for ε-greedy |
| `epsilon_min` | `float` | `0.05` | Minimum ε after decay |
| `epsilon_decay_steps` | `int` | `100_000` | Linear decay steps from init to min |
| `dropout` | `float` | `0.0` | Dropout probability |
| `use_bias` | `bool` | `True` | Whether linear layers use bias |
| `dtype` | `str` | `"float32"` | Tensor dtype |
| `seed` | `int` | `42` | Random seed |
| `log_freq` | `int` | `100` | Logging interval (episodes) |
| `eval_freq` | `int` | `1_000` | Evaluation interval (episodes) |
| `checkpoint_freq` | `int` | `10_000` | Checkpoint interval (episodes) |

---

## `networks.py`

### `class BaseOperator(nn.Module)`
Abstract base class for all network components. Provides a uniform interface for forward passes, parameter counting, and checkpointing.

**Methods:**
- `forward(*args, **kwargs)` — abstract; subclasses define specific shapes
- `count_parameters() -> int` — returns total trainable parameters

---

### `def count_params(model: nn.Module) -> None`
Print total and trainable parameter counts for any model.

---

### `class StateEncoder(BaseOperator)`
State encoder φ(s). Configurable for vector or image observations.

**Constructor:** `StateEncoder(cfg: ModelConfig)`

**Methods:**
- `forward(s: torch.Tensor) -> torch.Tensor`
  - `s`: (B, obs_dim) for vector obs, or (B, C, H, W) for image obs
  - Returns: (B, encoder_dim) — state representation, LayerNorm applied
  - dtype: float32 or bfloat16; float16 not tested
- `forward_checkpoint(s: torch.Tensor) -> torch.Tensor`
  - Same as forward but with gradient checkpointing (memory-efficient)

---

### `class ActionEmbeddingTable(BaseOperator)`
Learnable embedding table E ∈ R^{|A| × d_emb}. Supports dynamic row addition mid-training via k-NN interpolation, and per-row gradient freezing via a gradient hook (compile-compatible).

**Constructor:** `ActionEmbeddingTable(n_actions: int, cfg: ModelConfig)`

**Properties:**
- `weight` — `nn.Parameter` of shape (n_actions, d_emb)
- `n_actions` — current number of action embeddings
- `_freeze_mask` — buffer of shape (n_actions, 1); 1.0 = trainable, 0.0 = frozen

**Methods:**
- `forward(indices: torch.Tensor) -> torch.Tensor`
  - `indices`: (B,) long tensor of action indices
  - Returns: (B, d_emb) action embeddings
- `add_embedding(n_known: int, k: int = 3, similarity: str = "cosine", source_embeddings: Optional[torch.Tensor] = None) -> None`
  - Appends a new embedding row initialised via k-NN centroid interpolation
  - `n_known`: number of existing embeddings
  - `k`: nearest neighbours to average
  - `similarity`: "cosine" or "euclidean"
  - `source_embeddings`: source table for target-network sync (optional)
- `set_rows_frozen(indices: torch.Tensor, frozen: bool = True) -> None`
  - Set freeze status for specific rows via the gradient mask
- `freeze_all() -> None` — freeze all rows
- `unfreeze_all() -> None` — unfreeze all rows

---

### `class QHead(BaseOperator)`
Shared Q-function f_Q(φ(s), e_a). Takes concatenated [state_representation; action_embedding] and outputs a scalar Q-value.

Architecture: (encoder_dim + d_embedding) → q_hidden_dim (× q_n_layers) → 1

**Constructor:** `QHead(cfg: ModelConfig)`

**Methods:**
- `forward(phi_s: torch.Tensor, e_a: torch.Tensor) -> torch.Tensor`
  - `phi_s`: (B, d_enc) — state representation
  - `e_a`: (d_emb,) single embedding (broadcast to batch) or (B, d_emb) batched
  - Returns: (B, 1) Q-values
  - Residual NOT added; this is the raw Q output
- `forward_checkpoint(phi_s, e_a) -> torch.Tensor` — gradient checkpointing variant

---

### `class ValueHead(BaseOperator)`
State-value head V(s) — used only in the dueling variant. Architecture: encoder_dim → q_hidden_dim → 1.

**Constructor:** `ValueHead(cfg: ModelConfig)`

**Methods:**
- `forward(phi_s: torch.Tensor) -> torch.Tensor`
  - `phi_s`: (B, d_enc)
  - Returns: (B, 1) scalar state value

---

### `class DuelingFactorizedQ(BaseOperator)`
Dueling decomposition: Q(s,a) = V(s) + A(s,a). The centering over known actions only (not new actions) avoids the softmax denominator shift problem.

**Constructor:** `DuelingFactorizedQ(cfg: ModelConfig)`

**Methods:**
- `forward(phi_s: torch.Tensor, e_a: torch.Tensor, all_embeddings: Optional[torch.Tensor] = None, n_known: Optional[int] = None) -> torch.Tensor`
  - `phi_s`: (B, d_enc)
  - `e_a`: (d_emb,) or (B, d_emb)
  - `all_embeddings`: (|A_total|, d_emb) — for centering
  - `n_known`: number of pre-expansion actions (defaults to all_embeddings.shape[0])
  - Returns: (B, 1) Q-values

---

### `class DynamicsHead(BaseOperator)`
Auxiliary dynamics prediction head. Predicts φ(s') − φ(s) from (φ(s), e_a). Loss: MSE(Δ_pred, φ(s') − φ(s)). Weighted by 0.01 in the total loss.

**Constructor:** `DynamicsHead(cfg: ModelConfig)`

**Methods:**
- `forward(phi_s: torch.Tensor, e_a: torch.Tensor) -> torch.Tensor`
  - `phi_s`: (B, d_enc)
  - `e_a`: (d_emb,) or (B, d_emb)
  - Returns: (B, d_enc) predicted state delta

---

## `agent.py`

### `Transition`
Namedtuple with fields: `state`, `action`, `reward`, `next_state`, `done`.

---

### `class ReplayBuffer`
Fixed-capacity replay buffer for off-policy DQN. Stores (s, a, r, s', done) tuples.

**Constructor:** `ReplayBuffer(capacity: int)`

**Methods:**
- `push(state, action: int, reward: float, next_state, done: bool) -> None`
  - Stores a single transition. Internally converts to float32 numpy arrays.
- `sample(batch_size: int) -> Transition`
  - Uniform random batch. Returns batched torch tensors.
  - `state`: (B, obs_dim) or (B, C, H, W)
  - `action`: (B,) long
  - `reward`: (B,)
  - `next_state`: (B, obs_dim) or (B, C, H, W)
  - `done`: (B,)
- `__len__() -> int` — current buffer fill level

---

### `class ActionIncrementalDQN`
Deep Q-Network agent that handles unexpected action-space expansion. Maintains a factorized Q-function, expandable action embedding table, progressive freeze schedules, and optimistic exploration bonuses.

**Constructor:** `ActionIncrementalDQN(cfg: ModelConfig, n_actions: int)`

**Public methods:**

- `select_action(s: np.ndarray, epsilon: float) -> int`
  - ε-greedy action selection with optimistic exploration bonus for new actions
  - `s`: (obs_dim,) for vector or (C, H, W) for image
  - `epsilon`: probability of random action
  - Returns: action index in [0, action_count)

- `update(batch: Transition) -> dict`
  - Single TD-learning step. Includes optional auxiliary dynamics loss.
  - `batch`: Transition from ReplayBuffer.sample()
  - Returns: dict with keys `td_loss`, `aux_loss`, `q_mean`
  - Shape invariants:
    - All tensors moved to `self.device`
    - grad_norm clipped to 10.0 globally

- `handle_action_expansion(env) -> None`
  - Integrate a newly discovered action into the network.
  - Called when `env.action_space.n > self.action_count`.
  - Protocol: (1) k-NN embedding init, (2) register new index + exploration bonus, (3) Phase 1 freeze, (4) rebuild optimizer.

- `maybe_update_freeze_schedule() -> None`
  - Progress through freeze phases based on steps since expansion.
  - Phase 1→2: after `freeze_q_head_steps` (unfreeze Q-head)
  - Phase 2→3: after `freeze_encoder_steps` (unfreeze all, 0.1× LR)

- `decay_exploration_bonuses() -> None`
  - Decay exploration bonuses for visited new actions (bonus *= decay).

- `get_q_for_actions(s: np.ndarray) -> np.ndarray`
  - Compute Q(s, a) for all currently known actions.
  - `s`: state observation
  - Returns: (action_count,) array of Q-values

**Internal methods (used by the public API):**

- `_batch_q_values(phi_s: torch.Tensor) -> torch.Tensor`
  - Batched Q(s,a) for ALL known actions. Returns (action_count,) for single state or (B, action_count) for batch.
- `_batch_target_q_values(phi_s_prime: torch.Tensor) -> torch.Tensor`
  - Target network batched Q. Returns (B, action_count).
- `_compute_q(phi_s: torch.Tensor, e_a: torch.Tensor) -> torch.Tensor`
  - Single-action Q(s,a). Returns (B, 1).
- `_sync_target() -> None` — hard copy online → target
- `_apply_freeze_phase1() -> None` — freeze all except new embedding
- `_apply_freeze_phase2() -> None` — unfreeze Q-head, keep encoder frozen
- `_apply_freeze_phase3() -> None` — unfreeze all, reduce LR to 0.1×
- `_trainable_params() -> list` — params with requires_grad=True

---

## `env_wrapper.py`

### `class ExpandingActionWrapper(gym.Wrapper)`
Minimal wrapper that simulates action-space expansion at a fixed step threshold. Maps out-of-range actions to 0 (no-op in many environments).

**Constructor:** `ExpandingActionWrapper(env, expand_at_step=1000, new_action_name="new_action", expand_at_episode=None, max_expansions=1)`

**Properties:**
- `has_expanded` — bool flag
- `action_space` — returns the (potentially expanded) action space

**Methods:**
- `reset(**kwargs)` — reset environment (step counter persists across episodes)
- `step(action) -> (obs, reward, terminated, truncated, info)` — standard Gymnasium step; checks for expansion at each call

---

## `train.py`

### `def build_env(env_name: str, seed: int = 42) -> gym.Env`
Create an environment with optional expansion wrapper.

Supported environments:
- `"cartpole"` — CartPole-v1 with single expansion at step 1000
- `"cartpole_multi"` — CartPole-v1 with 2 expansions at step 800
- `"gridworld"` — CartPole-v1 placeholder with expansion at step 2000
- `"dave"` / `"custom"` — raises `NotImplementedError` (TODO)

### `def compute_epsilon(cfg, env_step, last_expansion_step) -> float`
Annealed ε for ε-greedy with post-expansion boost (floor at 0.5 during cooldown).

### `def train_episode(agent, env, cfg, epsilon) -> tuple`
Run a single training episode with expansion detection, Q-value logging, and replay buffer updates.

### `def evaluate(agent, env, n_episodes=5) -> float`
Evaluate agent with greedy (ε=0) policy. Returns average reward.

### `def save_checkpoint(agent, cfg, episode, path=None) -> None`
Save agent checkpoint to disk.

### `def train(cfg, env_name, n_episodes) -> dict`
Main training loop with expansion detection, freeze-schedule management, and logging. Returns dict with `episode_rewards`, `expansion_events`, `q_degradation_log`, and `agent`.

### `def main()`
CLI entry point. Arguments:
- `--env` (default: `cartpole`)
- `--n_episodes` (default: 2000)
- `--seed` (default: 42)
- `--q_form` (default: `factorized`)
- `--d_embedding` (default: 64)
- `--k_nn` (default: 3)
- `--optimistic_bonus` (default: 2.0)
- `--freeze_steps` (default: 10000)
- `--no_aux_loss` (flag)
