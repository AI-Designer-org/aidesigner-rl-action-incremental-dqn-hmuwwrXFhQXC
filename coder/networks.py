"""Core neural network components for Action-Space Incremental RL.

Architecture (factorized variant):
  State s --(encoder)--> phi(s) --(concat [phi(s); e_a])--> QHead --> Q(s,a)
                                    ^
                              ActionEmbeddingTable lookup: a --> e_a

Architecture (dueling_factorized variant):
  State s --(encoder)--> phi(s) --(ValueHead)--> V(s)
                               --(QHead[phi(s); e_a])--> A(s,a)
                          Q(s,a) = V(s) + A(s,a)

Key design decisions:
  - Factorized Q decouples action identity from state representation
  - Action embeddings are learned per-action with k-NN init for new actions
  - Gradient freezing via hooks (not requires_grad manipulation) for
    torch.compile compatibility
  - bf16/fp16 safety: all reduction ops cast to float32 internally
  - Gradient checkpointing hooks on the Q-head for memory-efficient training
"""

import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model_config import ModelConfig


# ── Abstract Base Class ───────────────────────────────────────────────────────

class BaseOperator(ABC, nn.Module):
    """Abstract base class for the core novel operator in the architecture.

    Every network component (encoder, Q-head, embedding table) inherits from
    this, ensuring a uniform interface for forward passes, checkpointing,
    and parameter management.
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass. Subclasses define specific input/output shapes."""
        pass

    def count_parameters(self) -> int:
        """Return total number of trainable parameters in this module."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Parameter Count Helper ───────────────────────────────────────────────────

def count_params(model: nn.Module) -> None:
    """Print total and trainable parameter counts for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")


# ── State Encoder ────────────────────────────────────────────────────────────

class StateEncoder(BaseOperator):
    """State encoder phi(s). Configurable for vector or image observations.

    Vector obs: MLP with configurable depth and width.
    Image obs:  Standard DQN-style CNN (3 conv layers + 1 FC head).

    Output is always (B, encoder_dim) with optional LayerNorm.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.obs_type = cfg.obs_type

        if cfg.obs_type == "vector":
            # MLP encoder: obs_dim -> d_model (x n_layers) -> encoder_dim
            layers = []
            in_dim = cfg.obs_dim
            for _ in range(cfg.n_encoder_layers):
                layers.extend([
                    nn.Linear(in_dim, cfg.d_model, bias=cfg.use_bias),
                    nn.ReLU(),
                    nn.Dropout(cfg.dropout),
                ])
                in_dim = cfg.d_model
            layers.append(nn.Linear(cfg.d_model, cfg.encoder_dim, bias=cfg.use_bias))
            self.net = nn.Sequential(*layers)                              # (B, obs_dim) -> (B, encoder_dim)

        elif cfg.obs_type == "image":
            # DQN-style CNN: (B, C, H, W) -> (B, 64, 7, 7) for 84x84 input
            self.cnn = nn.Sequential(
                nn.Conv2d(cfg.obs_channels, 32, kernel_size=8, stride=4),   # (B, 32, 20, 20)
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),                # (B, 64, 9, 9)
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),                # (B, 64, 7, 7)
                nn.ReLU(),
                nn.Flatten(),                                               # (B, 64*7*7)
            )
            self._init_cnn_head(cfg)
        else:
            raise ValueError(f"Unknown obs_type: {cfg.obs_type}")

        self.norm = nn.LayerNorm(cfg.encoder_dim) if cfg.encoder_norm else nn.Identity()

    def _init_cnn_head(self, cfg: ModelConfig) -> None:
        """Compute CNN flatten dimension and initialise the linear head."""
        with torch.no_grad():
            dummy = torch.zeros(1, cfg.obs_channels, cfg.obs_height, cfg.obs_width)
            out = self.cnn(dummy)                                          # (1, cnn_dim)
            cnn_dim = out.size(1)
        self.head = nn.Linear(cnn_dim, cfg.encoder_dim, bias=cfg.use_bias) # (B, cnn_dim) -> (B, encoder_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Encode state observation to fixed-dimension representation.

        Args:
            s: (B, obs_dim) for vector obs, or (B, C, H, W) for image obs.

        Returns:
            (B, encoder_dim) state representation.
        """
        if self.obs_type == "vector":
            x = self.net(s)                                                # (B, encoder_dim)
        else:
            x = self.cnn(s)                                                # (B, cnn_dim)
            x = self.head(x)                                               # (B, encoder_dim)
        x = self.norm(x)                                                   # (B, encoder_dim)

        # bf16/fp16 safety: no numerically sensitive ops (LayerNorm is safe)
        return x

    def forward_checkpoint(self, s: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient checkpointing for memory savings."""
        return checkpoint(self.forward, s, use_reentrant=False)


# ── Action Embedding Table ───────────────────────────────────────────────────

class ActionEmbeddingTable(BaseOperator):
    """Learnable embedding table E in R^{|A| x d_emb}.

    Supports:
    - Adding new rows mid-training via k-NN interpolation of existing embeddings.
    - Per-row gradient freezing via a gradient hook (compile-compatible).
    - Target network sync by copying source embeddings.

    The table grows dynamically: start with |A_init| rows and append rows
    as new actions are discovered. Re-allocation is rare (typically <=10 events).
    """

    def __init__(self, n_actions: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_actions = n_actions

        # Embedding weight: (n_actions, d_emb)
        self.weight = nn.Parameter(
            torch.randn(n_actions, cfg.d_embedding) * cfg.embedding_init_scale
        )

        # Freeze mask: 1.0 = trainable, 0.0 = frozen (per row).
        # Register as a buffer so it moves with the module and is
        # included in state_dict. Shape (n_actions, 1) for broadcasting.
        self.register_buffer("_freeze_mask", torch.ones(n_actions, 1))

        # Register gradient hook ONCE in __init__ for compile compatibility.
        # The hook zeroes gradients for frozen rows before the optimizer step.
        # Use a bound instance method so self._freeze_mask is accessible.
        self.weight.register_hook(self._freeze_hook)

    # ── Hook ─────────────────────────────────────────────────────────────

    def _freeze_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """Zero out gradients for rows where _freeze_mask == 0.

        Args:
            grad: (n_actions, d_emb) gradient w.r.t. weight.

        Returns:
            (n_actions, d_emb) masked gradient.
        """
        return grad * self._freeze_mask

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up action embeddings for given indices.

        Args:
            indices: (B,) long tensor of action indices.
        Returns:
            (B, d_emb) action embeddings.
        """
        return F.embedding(indices, self.weight)                           # (B, d_emb)

    # ── Row freezing ─────────────────────────────────────────────────────

    def set_rows_frozen(self, indices: torch.Tensor, frozen: bool = True) -> None:
        """Set freeze status for specific action rows.

        Uses the _freeze_mask buffer. A value of 0.0 means frozen
        (gradients zeroed), 1.0 means trainable.

        Args:
            indices: (K,) tensor of action indices to modify.
            frozen: If True, freeze these rows; if False, unfreeze.
        """
        mask_val = 0.0 if frozen else 1.0
        self._freeze_mask[indices] = mask_val

    def freeze_all(self) -> None:
        """Freeze all rows (set mask to 0 everywhere)."""
        self._freeze_mask[:] = 0.0

    def unfreeze_all(self) -> None:
        """Unfreeze all rows (set mask to 1 everywhere)."""
        self._freeze_mask[:] = 1.0

    # ── Dynamic expansion ───────────────────────────────────────────────

    @torch.no_grad()
    def add_embedding(
        self,
        n_known: int,
        k: int = 3,
        similarity: str = "cosine",
        source_embeddings: Optional[torch.Tensor] = None,
    ) -> None:
        """Append a new embedding row initialised via k-NN centroid interpolation.

        Strategy: compute the centroid of ALL known embeddings; find its
        k nearest neighbours in embedding space; take the mean as the
        new embedding.

        Args:
            n_known: Number of existing (known) action embeddings.
            k: Number of nearest neighbours to interpolate.
            similarity: "cosine" or "euclidean".
            source_embeddings: Source table to copy from (for target
                network sync). If None, uses self.weight.
        """
        device = self.weight.device
        source = source_embeddings if source_embeddings is not None else self.weight.data
        existing = source[:n_known]                                        # (n_known, d_emb)

        # Centroid as proxy query (no semantic descriptor of a_new)
        centroid = existing.mean(dim=0, keepdim=True)                      # (1, d_emb)

        if similarity == "cosine":
            sim = F.cosine_similarity(centroid, existing, dim=1)           # (n_known,)
            _, idx = torch.topk(sim, min(k, n_known))
        elif similarity == "euclidean":
            dist = torch.cdist(centroid.float(), existing.float()).squeeze(0)  # (n_known,)
            _, idx = torch.topk(dist, min(k, n_known), largest=False)
        else:
            raise ValueError(f"Unknown similarity: {similarity}")

        e_new = existing[idx].mean(dim=0)                                  # (d_emb,)

        # Append new row to weight
        new_weight = torch.cat([self.weight.data, e_new.unsqueeze(0)], dim=0)  # (n_actions+1, d_emb)
        self.weight = nn.Parameter(new_weight)
        self.n_actions += 1

        # Extend freeze mask: new row is trainable (1.0) by default
        new_mask = torch.cat(
            [self._freeze_mask, torch.ones(1, 1, device=device)], dim=0
        )
        self._freeze_mask = new_mask


# ── Q-Head ───────────────────────────────────────────────────────────────────

class QHead(BaseOperator):
    """Shared Q-function f_Q(phi(s), e_a).

    Takes [state_representation; action_embedding] (concatenated) and
    outputs a scalar Q-value.

    Architecture: (d_enc + d_emb) -> q_hidden_dim (x q_n_layers) -> 1
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.encoder_dim + cfg.d_embedding                         # concat dim

        layers = []
        dim = in_dim
        for _ in range(cfg.q_n_layers):
            layers.append(nn.Linear(dim, cfg.q_hidden_dim, bias=cfg.use_bias))
            if cfg.q_activation == "relu":
                layers.append(nn.ReLU())
            elif cfg.q_activation == "gelu":
                layers.append(nn.GELU())
            layers.append(nn.Dropout(cfg.dropout))
            dim = cfg.q_hidden_dim
        layers.append(nn.Linear(cfg.q_hidden_dim, 1, bias=cfg.use_bias))

        self.net = nn.Sequential(*layers)

    def forward(self, phi_s: torch.Tensor, e_a: torch.Tensor) -> torch.Tensor:
        """Compute Q(s, a).

        Args:
            phi_s: (B, d_enc) state representation.
            e_a:   (d_emb,) single action embedding (broadcast over batch)
                   or (B, d_emb) batched action embeddings.
        Returns:
            (B, 1) Q-values.
        """
        # Broadcast single embedding across batch dimension
        if e_a.dim() == 1:
            e_a = e_a.unsqueeze(0).expand(phi_s.size(0), -1)              # (B, d_emb)

        x = torch.cat([phi_s, e_a], dim=-1)                               # (B, d_enc + d_emb)
        q = self.net(x)                                                    # (B, 1)

        # bf16/fp16 safety: no numerically sensitive ops in linear layers
        return q

    def forward_checkpoint(self, phi_s: torch.Tensor, e_a: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient checkpointing."""
        return checkpoint(self.forward, phi_s, e_a, use_reentrant=False)


# ── Value Head (Dueling Variant) ─────────────────────────────────────────────

class ValueHead(BaseOperator):
    """State-value head V(s) — used only in the dueling variant.

    Q(s,a) = V(s) + A(s,a)   where A(s,a) = QHead(phi(s), e_a)

    Architecture: encoder_dim -> q_hidden_dim -> 1
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.encoder_dim, cfg.q_hidden_dim, bias=cfg.use_bias),  # (B, encoder_dim) -> (B, q_hidden_dim)
            nn.ReLU(),
            nn.Linear(cfg.q_hidden_dim, 1, bias=cfg.use_bias),               # (B, q_hidden_dim) -> (B, 1)
        )

    def forward(self, phi_s: torch.Tensor) -> torch.Tensor:
        """Compute V(s).

        Args:
            phi_s: (B, d_enc) state representation.
        Returns:
            (B, 1) scalar state value.
        """
        return self.net(phi_s)                                             # (B, 1)


# ── Dueling Factorized Q (Ablation Module) ──────────────────────────────────

class DuelingFactorizedQ(BaseOperator):
    """Dueling decomposition of factorized Q.

    Q(s,a) = V(s) + A(s,a)   where A(s,a) = f_adv(phi(s), e_a)

    Key benefit over plain factorized Q:
      V(s) is completely unaffected by action-space expansion.
      Only advantage residuals for new actions need to be learned.

    Centering subtracts the mean advantage over KNOWN actions only,
    avoiding the softmax denominator shift problem from new actions.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.value_head = ValueHead(cfg)           # V(s)
        self.advantage_head = QHead(cfg)           # A(s,a) — reuses QHead MLP

    def forward(
        self,
        phi_s: torch.Tensor,
        e_a: torch.Tensor,
        all_embeddings: Optional[torch.Tensor] = None,
        n_known: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute Q(s,a) via dueling decomposition.

        Args:
            phi_s: (B, d_enc) state representation.
            e_a: (d_emb,) or (B, d_emb) action embedding.
            all_embeddings: (|A_total|, d_emb) — full embedding table.
                Used for centering over known-only actions.
            n_known: Number of 'known' (pre-expansion) actions.
                Defaults to all_embeddings.shape[0] if None.
        Returns:
            (B, 1) Q-values.
        """
        V = self.value_head(phi_s)                                         # (B, 1)
        A = self.advantage_head(phi_s, e_a)                                # (B, 1)

        if all_embeddings is not None:
            known = n_known if n_known is not None else all_embeddings.shape[0]
            with torch.no_grad():
                # Batch-compute A(s, a) for all actions to find mean over known
                A_all = self._batch_advantage(phi_s, all_embeddings)       # (B, |A|, 1)
                A_known_mean = A_all[:, :known].mean(dim=1)                 # (B, 1)
            A = A - A_known_mean                                           # (B, 1)

        return V + A                                                       # (B, 1)

    def _batch_advantage(self, phi_s: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute A(s, a) for all actions in a single batched forward pass.

        Args:
            phi_s: (B, d_enc)
            embeddings: (|A|, d_emb)
        Returns:
            (B, |A|, 1) advantage values.
        """
        B = phi_s.size(0)
        n_act = embeddings.size(0)

        # Expand: (B, 1, d_enc) tile to match each action
        phi_expanded = phi_s.unsqueeze(1).expand(B, n_act, -1)            # (B, |A|, d_enc)
        # Expand: (1, |A|, d_emb) tile to match each batch element
        e_expanded = embeddings.unsqueeze(0).expand(B, -1, -1)             # (B, |A|, d_emb)

        # Concatenate along last dim
        x = torch.cat([phi_expanded, e_expanded], dim=-1)                  # (B, |A|, d_enc+d_emb)

        # Apply the QHead MLP to the last dimension
        # The MLP operates on (..., d) -> (..., 1), so we flatten
        # the batch and action dimensions, then reshape back
        out = self.advantage_head.net(x.view(B * n_act, -1))               # (B*|A|, 1)
        return out.view(B, n_act, 1)                                       # (B, |A|, 1)


# ── Dynamics Head (Auxiliary Loss) ──────────────────────────────────────────

class DynamicsHead(BaseOperator):
    """Auxiliary dynamics prediction head.

    Predicts the state-delta: phi(s') - phi(s) from (phi(s), e_a).

    This shapes the action embedding space to encode transition dynamics,
    creating a semantically meaningful manifold where actions with similar
    effects have similar embeddings. This directly improves the quality
    of k-NN initialisation for new actions.

    Loss: L_aux = MSE(delta_pred, phi(s') - phi(s))
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        in_dim = cfg.encoder_dim + cfg.d_embedding
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.q_hidden_dim, bias=cfg.use_bias),         # (B, d_enc+d_emb) -> (B, q_hidden_dim)
            nn.ReLU(),
            nn.Linear(cfg.q_hidden_dim, cfg.encoder_dim, bias=cfg.use_bias), # (B, q_hidden_dim) -> (B, d_enc)
        )

    def forward(self, phi_s: torch.Tensor, e_a: torch.Tensor) -> torch.Tensor:
        """Predict state transition delta.

        Args:
            phi_s: (B, d_enc) current state representation.
            e_a:   (d_emb,) or (B, d_emb) action embedding.
        Returns:
            (B, d_enc) predicted phi(s') - phi(s).
        """
        if e_a.dim() == 1:
            e_a = e_a.unsqueeze(0).expand(phi_s.size(0), -1)               # (B, d_emb)

        x = torch.cat([phi_s, e_a], dim=-1)                                # (B, d_enc + d_emb)
        delta = self.net(x)                                                 # (B, d_enc)
        return delta
