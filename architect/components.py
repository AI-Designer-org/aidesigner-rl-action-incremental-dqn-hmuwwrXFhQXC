"""Core neural network components for Action-Space Incremental RL.

Architecture:
  State s ──→ φ(s) ──→ Q(s,a) = f_Q(φ(s), e_a)
                    ↕
               Action Embedding Table E: lookup e_a for action a

When a new action is added mid-training:
  1. Append a row to E, initialized via k-NN centroid of known embeddings
  2. Set optimistic exploration bonus for the new action
  3. Progressively unfreeze: new embedding → Q-head → encoder
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_config import ModelConfig


class StateEncoder(nn.Module):
    """State encoder φ(s). Configurable for vector or image observations.

    Vector obs: MLP with configurable depth.
    Image obs:  Standard DQN-style CNN (3 conv + 1 FC).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.obs_type == "vector":
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
            self.net = nn.Sequential(*layers)

        elif cfg.obs_type == "image":
            self.cnn = nn.Sequential(
                nn.Conv2d(cfg.obs_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            self._init_cnn_head(cfg)
        else:
            raise ValueError(f"Unknown obs_type: {cfg.obs_type}")

        self.norm = nn.LayerNorm(cfg.encoder_dim) if cfg.encoder_norm else nn.Identity()

    def _init_cnn_head(self, cfg: ModelConfig):
        """Compute CNN flatten dim via forward pass on dummy input."""
        with torch.no_grad():
            dummy = torch.zeros(1, cfg.obs_channels, cfg.obs_height, cfg.obs_width)
            out = self.cnn(dummy)
            cnn_dim = out.size(1)
        self.head = nn.Linear(cnn_dim, cfg.encoder_dim, bias=cfg.use_bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        if self.cfg.obs_type == "vector":
            x = self.net(s)
        else:
            x = self.cnn(s)
            x = self.head(x)
        return self.norm(x)


class ActionEmbeddingTable(nn.Module):
    """Learnable embedding table E ∈ ℝ^{|A| × d_emb}.

    Supports:
    - adding new rows mid-training via k-NN interpolation of existing embeddings
    - freezing old rows while allowing new-row training (by requires_grad control)
    - target network sync by copying source embeddings

    The table grows dynamically: we start with |A_init| rows and append rows
    as new actions are discovered. There is no pre-allocation cap; the tensor
    is re-allocated on each expansion (which is rare, typically ≤10 events).
    """

    def __init__(self, n_actions: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(
            torch.randn(n_actions, cfg.d_embedding) * cfg.embedding_init_scale
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up action embeddings for given indices.

        Args:
            indices: (B,) long tensor of action indices.
        Returns:
            (B, d_emb) embeddings.
        """
        return F.embedding(indices, self.weight)

    @torch.no_grad()
    def add_embedding(
        self,
        n_known: int,
        k: int = 3,
        similarity: str = "cosine",
        source_embeddings: torch.Tensor | None = None,
    ) -> None:
        """Append a new embedding row initialized via k-NN centroid interpolation.

        Strategy: Compute the centroid of ALL known embeddings; find its
        k nearest neighbors in embedding space; take the mean.

        Args:
            n_known: Number of existing (known) action embeddings.
            k: Number of nearest neighbors to interpolate.
            similarity: "cosine" or "euclidean".
            source_embeddings: Source table to copy from (for target network sync).
                               If None, uses self.weight.
        """
        source = source_embeddings if source_embeddings is not None else self.weight.data
        existing = source[:n_known]  # (n_known, d_emb)

        # Use centroid as proxy query (no semantic descriptor of a_new)
        centroid = existing.mean(dim=0, keepdim=True)  # (1, d_emb)

        if similarity == "cosine":
            sim = F.cosine_similarity(centroid, existing, dim=1)  # (n_known,)
            _, idx = torch.topk(sim, min(k, n_known))
        elif similarity == "euclidean":
            dist = torch.cdist(centroid, existing).squeeze(0)  # (n_known,)
            _, idx = torch.topk(dist, min(k, n_known), largest=False)
        else:
            raise ValueError(f"Unknown similarity: {similarity}")

        e_new = existing[idx].mean(dim=0)  # (d_emb,)

        # Append new row
        new_weight = torch.cat([self.weight.data, e_new.unsqueeze(0)], dim=0)
        self.weight = nn.Parameter(new_weight)


class QHead(nn.Module):
    """Shared Q-function f_Q(φ(s), e_a).

    Takes [state_representation; action_embedding] (concatenated) and
    outputs a scalar Q-value.

    This is a 2-layer MLP: (d_enc + d_emb) → hidden → 1
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.encoder_dim + cfg.d_embedding

        layers = []
        dim = in_dim
        for i in range(cfg.q_n_layers):
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
            e_a: (d_emb,) single action embedding (broadcast over batch)
                 or (B, d_emb) batched action embeddings.
        Returns:
            (B, 1) Q-values.
        """
        if e_a.dim() == 1:
            e_a = e_a.unsqueeze(0).expand(phi_s.size(0), -1)

        x = torch.cat([phi_s, e_a], dim=-1)  # (B, d_enc + d_emb)
        return self.net(x)


class ValueHead(nn.Module):
    """State-value head V(s) — used only in the dueling variant.

    Q(s,a) = V(s) + A(s,a)  where A(s,a) = QHead(φ(s), e_a)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.encoder_dim, cfg.q_hidden_dim, bias=cfg.use_bias),
            nn.ReLU(),
            nn.Linear(cfg.q_hidden_dim, 1, bias=cfg.use_bias),
        )

    def forward(self, phi_s: torch.Tensor) -> torch.Tensor:
        """Compute V(s).

        Args:
            phi_s: (B, d_enc)
        Returns:
            (B, 1) scalar value.
        """
        return self.net(phi_s)


class DuelingFactorizedQ(nn.Module):
    """Dueling decomposition of factorized Q.

    Q(s,a) = V(s) + A(s,a)   where A(s,a) = f_adv(φ(s), e_a)

    Key benefit over plain factorized Q:
    V(s) is completely unaffected by action-space expansion.
    Only advantage residuals for new actions need to be learned.

    The centering term subtracts mean advantage over KNOWN actions only,
    avoiding the denominator shift problem.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.value_head = ValueHead(cfg)
        self.advantage_head = QHead(cfg)  # reuses same MLP architecture

    def forward(
        self,
        phi_s: torch.Tensor,
        e_a: torch.Tensor,
        all_embeddings: torch.Tensor | None = None,
        n_known: int | None = None,
    ) -> torch.Tensor:
        """Compute Q(s,a) via dueling decomposition.

        Args:
            phi_s: (B, d_enc) state representation.
            e_a: (d_emb,) or (B, d_emb) action embedding.
            all_embeddings: (|A_total|, d_emb) — full embedding table.
                             Used for centering. If None, no centering.
            n_known: Number of 'known' (pre-expansion) actions.
                     If None and all_embeddings provided, uses all_embeddings.shape[0].
        Returns:
            (B, 1) Q-values.
        """
        V = self.value_head(phi_s)   # (B, 1)
        A = self.advantage_head(phi_s, e_a)  # (B, 1)

        if all_embeddings is not None:
            # Center over KNOWN actions only — avoids denominator shift
            # from newly added actions
            known = n_known if n_known is not None else all_embeddings.shape[0]
            with torch.no_grad():
                A_all = []
                for i in range(all_embeddings.shape[0]):
                    A_all.append(self.advantage_head(phi_s, all_embeddings[i]))
                A_all = torch.stack(A_all, dim=1)       # (B, |A|, 1)
                A_known_mean = A_all[:, :known].mean(dim=1, keepdim=True)  # (B, 1)
            A = A - A_known_mean

        return V + A


class DynamicsHead(nn.Module):
    """Auxiliary dynamics prediction head.

    Predicts the state-delta: φ(s') - φ(s) from (φ(s), e_a).

    This shapes the action embedding space to encode transition dynamics,
    creating a semantically meaningful manifold where actions with similar
    effects have similar embeddings. This directly improves the quality
    of k-NN initialization for new actions.

    Loss: L_aux = MSE(Δ_pred, φ(s') - φ(s))
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        in_dim = cfg.encoder_dim + cfg.d_embedding
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.q_hidden_dim, bias=cfg.use_bias),
            nn.ReLU(),
            nn.Linear(cfg.q_hidden_dim, cfg.encoder_dim, bias=cfg.use_bias),
        )

    def forward(self, phi_s: torch.Tensor, e_a: torch.Tensor) -> torch.Tensor:
        """Predict state transition delta.

        Args:
            phi_s: (B, d_enc)
            e_a: (d_emb,) or (B, d_emb)
        Returns:
            (B, d_enc) predicted φ(s') - φ(s).
        """
        if e_a.dim() == 1:
            e_a = e_a.unsqueeze(0).expand(phi_s.size(0), -1)

        x = torch.cat([phi_s, e_a], dim=-1)
        return self.net(x)
