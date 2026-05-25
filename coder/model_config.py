"""Model configuration for Action-Space Incremental Reinforcement Learning.

This config controls every aspect of the architecture: observation modality,
action embedding dimensions, Q-function form, RL hyperparameters, action
expansion protocol, and exploration schedule.
"""
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

    # ── Encoder network ─────────────────────────────────────────────────────
    encoder_dim: int = 256          # output dimension of phi(s)
    d_model: int = 256              # internal hidden dim for encoder
    n_encoder_layers: int = 3       # MLP depth (vector obs); CNN depth (image obs)
    encoder_norm: bool = True       # LayerNorm after encoder output

    # ── Action embedding ────────────────────────────────────────────────────
    d_embedding: int = 64           # dimensionality of action embedding e_a
    embedding_init_scale: float = 0.1  # std for random embedding initialization
    k_nn: int = 3                   # k for k-NN embedding interpolation
    embedding_similarity: Literal["cosine", "euclidean"] = "cosine"
    use_auxiliary_dynamics_loss: bool = True  # aux loss for embedding quality

    # ── Q-function (shared advantage head) ──────────────────────────────────
    q_hidden_dim: int = 128         # hidden dim of f_Q MLP
    q_n_layers: int = 2             # depth of f_Q MLP
    q_activation: Literal["relu", "gelu"] = "relu"
    q_form: Literal["factorized", "dueling_factorized"] = "factorized"
        # "factorized":           Q(s,a) = f_Q(phi(s) || e_a)      — shared MLP
        # "dueling_factorized":   Q(s,a) = V(s) + A(s,a)           — value + advantage

    # ── RL algorithm ────────────────────────────────────────────────────────
    gamma: float = 0.99             # discount factor
    learning_rate: float = 3e-4     # Adam learning rate
    buffer_capacity: int = 100_000  # replay buffer size
    batch_size: int = 64            # training batch size
    target_update_freq: int = 1_000 # hard copy interval for target network
    target_tau: float = 1.0         # 1.0 = hard update; <1.0 = Polyak soft update
    gradient_steps: int = 1         # training updates per env step

    # ── Action expansion ────────────────────────────────────────────────────
    optimistic_init_bonus: float = 2.0   # beta multiplier for optimistic Q bonus
    optimistic_bonus_decay: float = 0.99 # decay per visit of new action
    optimistic_bonus_min: float = 0.01   # minimum bonus after decay
    freeze_encoder_steps: int = 10_000   # steps to keep encoder frozen post-expansion
    freeze_old_embeddings: bool = True   # freeze existing action embeddings during fine-tune
    freeze_q_head_steps: int = 5_000     # steps to keep old advantage head frozen
    expansion_cooldown: int = 1_000      # min env steps between consecutive expansions

    # ── Exploration (epsilon-greedy) ────────────────────────────────────────
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
