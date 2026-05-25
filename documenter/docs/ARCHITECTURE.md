# Architecture

## 1. Motivation

Standard deep reinforcement learning assumes a fixed action space A of size |A| that is known at the start of training and never changes. In many real-world settings this assumption fails: video games unlock new abilities at higher levels, robots acquire new end-effectors, and skill-learning agents discover new affordances. Retraining from scratch is sample-inefficient and discards the knowledge encoded in the existing policy and value function.

Three existing lines of work address related but distinct settings:

- **Growing Action Spaces (GAS)** — Farquhar et al., ICML 2020. Uses a predefined nested hierarchy A₀ ⊂ A₁ ⊂ ... ⊂ A_{N-1} with parent-child Q-decomposition. The hierarchy must be known before training, so it cannot handle *unexpected* action additions (e.g., a "shoot" action appearing at level 3 when only movement actions existed at level 1).
- **morphin** — de la Rosa et al., arXiv 2026. Detects action-space changes via Page-Hinkley drift and expands Q-tables on-the-fly, but is tabular only — it does not scale to deep RL with function approximation.
- **Headless-AD** — Sinii et al., NeurIPS 2023 workshop. Uses action embeddings and InfoNCE loss to generalise to unseen action set sizes at test time, but does not address incremental introduction *during training*.

No existing deep RL method handles the scenario where a truly unexpected action appears mid-training, without a predefined hierarchy, a tabular state space, or the requirement to retrain from scratch. The core architectural question is therefore:

> Can a factorized action-value architecture with a learnable action-embedding space and a similarity-based Q-transfer mechanism incorporate unexpectedly introduced actions into a deep RL policy without catastrophic forgetting and with sub-linear fine-tuning cost relative to retraining from scratch?

This architecture tests the hypothesis that the answer is yes.

## 2. At a glance

```
         ┌─────────────────────────────────────────────────────────────┐
         │  Observation s                                             │
         │  (vector R^d or image R^{3 x H x W})                       │
         └─────────────────────┬───────────────────────────────────────┘
                               │
                               v
         ┌─────────────────────────────────────────────────────────────┐
         │  State Encoder phi(s)                                      │
         │  * MLP (vector obs) or CNN (image obs)                     │
         │  * Output: R^{encoder_dim} (256)                           │
         │  * LayerNorm on output                                     │
         └─────────────────────┬───────────────────────────────────────┘
                               │
                               │ phi(s) in R^{256}
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                   │
             v                 v                   v
   ┌───────────────────┐   ┌───────────┐   ┌───────────────┐
   │  Value Head V(s)  │   │ Action    │   │ Auxiliary     │
   │  MLP: 256->128->1 │   │ Embedding │   │ Dynamics Head │
   │  (dueling variant)│   │ Table E   │   │ (optional)    │
   └─────────┬─────────┘   │ R^{|A|x64}│   │ phi(s')-phi(s)│
             │             └─────┬─────┘   │ prediction    │
             │                   │          └───────────────┘
             │                   │ e_a in R^{64}
             │                   v
             │         ┌──────────────────────┐
             │         │  Advantage / Q Head  │
             │         │  f_Q(phi(s) || e_a)  │
             │         │  MLP: 320->128->1    │
             │         └──────────┬───────────┘
             │                    │
             v                    v
     ┌───────────────────────────────┐
     │  Q(s, a) = V(s) + A(s,a)     │  (dueling variant)
     │  or Q(s, a) = f_Q(phi(s),e_a)│  (factorized variant)
     └───────────────────────────────┘
```

| Property | Value |
|---|---|
| Parameter count (default config) | 272,513 |
| Time complexity (forward, single action) | O(d_enc * d_model * n_enc + (d_enc + d_emb) * q_hidden * q_n_layers) |
| Time complexity (forward, all actions) | O(|A| * d_enc * d_model * n_enc + |A| * (d_enc + d_emb) * q_hidden * q_n_layers) |
| Time complexity (expansion) | O(|A_known| * d_emb) for k-NN + O(1) for table append |
| Space complexity | O(|A_max| * d_emb + d_enc * (d_model + 1) + q_hidden * (d_enc + d_emb + 1)) |
| Hardware requirements | CPU (inference); any GPU with >= 1 GB VRAM (training) |
| Observation types | Vector (MLP encoder) or Image (CNN encoder) |

## 3. The core component

### 3.1 Intuition

**Why factorized Q?** In a standard DQN, the Q-function is a single network that takes a state and outputs one scalar per action: Q(s) ∈ R^{|A|}. This ties the number of output units to the action-set size — adding an action requires adding an output neuron, which shifts the softmax denominator and changes all existing Q-values. In a factorized architecture, Q(s, a) = f_Q(φ(s), e_a), the Q-function is a function of *both* the state and a per-action embedding. New actions can be added by inserting a new embedding into the table E; the Q-function itself does not change shape.

**Why k-NN embedding initialisation?** When a new action arrives (say "shoot" in a game where only "left", "right", "up" exist), we have no training data to estimate its value. But the embedding table already encodes the relationships among existing actions. The centroid of the existing embeddings is the least-committal point in embedding space (minimax optimal under no semantic descriptor). Finding its k nearest neighbours and averaging their embeddings gives the new action an initial position that is plausible given the existing action manifold.

**Why progressive unfreezing?** The state encoder φ(s) has learned useful features from thousands of steps of interaction with the old action set — there is no reason to disrupt it. Phase 1 freezes everything except the new action's embedding, so only that embedding is tuned. Phase 2 unfreezes the Q-head (which maps [φ(s); e_a] → Q), allowing the utility computation to adjust. Phase 3 unfreezes the encoder at a reduced learning rate, permitting fine-grained adaptation without catastrophic forgetting.

### 3.2 Equations

Let:
- s ∈ R^{obs_dim} be a vector observation (or s ∈ R^{C × H × W} for images)
- φ: R^{obs_dim} → R^{d_enc} be the state encoder (MLP or CNN)
- E ∈ R^{|A| × d_emb} be the action embedding table
- e_a = E[a] ∈ R^{d_emb} be the embedding for action a
- f_Q: R^{d_enc + d_emb} → R be the Q-head MLP

**Factorized variant (default):**
```
Q(s, a) = f_Q([φ(s); e_a])
```
where [·; ·] denotes concatenation along the feature dimension.

**Dueling factorized variant (ablation):**
```
Q(s, a) = V(φ(s)) + f_adv([φ(s); e_a])
```
where V: R^{d_enc} → R is the state-value head.

**k-NN embedding initialisation for new action a_new:**
```
centroid = (1 / |A_known|) * sum_{a in A_known} e_a
neighbours = top_k_{e in E_known} sim(centroid, e)
e_new = (1 / k) * sum_{e in neighbours} e
```
where sim is cosine similarity or negative Euclidean distance.

**TD target (after expansion):**
```
y = r + gamma * max_{a' in A_current} Q'(s', a')
```
Note that max is taken over the *current* action set (which may be larger than the set that existed when the transition was recorded). This is justified by the monotonicity property of nested action spaces: V*_i(s) ≤ V*_j(s) for i < j (Farquhar et al., 2020).

**Auxiliary dynamics loss (optional):**
```
L_aux = MSE( Δ_pred, phi(s') - phi(s) )
where Δ_pred = DynamicsHead([φ(s); e_a])
```

### 3.3 Reference implementation walk-through

The following annotated excerpt from `coder/networks.py` shows the `ActionEmbeddingTable.add_embedding` method, which is the core novel operation:

```python
@torch.no_grad()
def add_embedding(self, n_known, k=3, similarity="cosine",
                  source_embeddings=None):
    """Append a new embedding row via k-NN centroid interpolation."""

    # Grab either the online or source (target-network) embeddings
    source = source_embeddings if source_embeddings is not None else self.weight.data
    existing = source[:n_known]                     # (n_known, d_emb)

    # Centroid as proxy query (no semantic descriptor of a_new)
    centroid = existing.mean(dim=0, keepdim=True)   # (1, d_emb)

    # Find k nearest neighbours by cosine similarity
    sim = F.cosine_similarity(centroid, existing, dim=1)  # (n_known,)
    _, idx = torch.topk(sim, min(k, n_known))

    # Mean of neighbours = new embedding
    e_new = existing[idx].mean(dim=0)                # (d_emb,)

    # Append to weight tensor (grow dynamically)
    new_weight = torch.cat([self.weight.data, e_new.unsqueeze(0)], dim=0)
    self.weight = nn.Parameter(new_weight)
    self.n_actions += 1
```

**Shapes at each step (d_emb=64, n_known=3, k=2):**
1. `source[:n_known]`: (3, 64) — three existing action embeddings
2. `centroid`: (1, 64) — mean along dim 0
3. `sim`: (3,) — cosine similarity of each existing embedding to centroid
4. `topk(sim, 2)` → `idx`: (2,) — indices of the two nearest neighbours
5. `existing[idx]`: (2, 64) — the two nearest embeddings
6. `existing[idx].mean(dim=0)`: (64,) — the new embedding
7. `new_weight`: (4, 64) — table grown by one row

## 4. Tensor shape evolution

Default config: vector obs (`obs_dim=64`), `encoder_dim=256`, `d_embedding=64`, `n_actions_init=3`.

| Stage | Shape | Notes |
|---|---|---|
| Input s | (B, 64) | dtype: float32 |
| After StateEncoder | (B, 256) | φ(s); LayerNorm applied |
| ActionEmbeddingTable lookup [a] | (d_emb,) = (64,) | or (B, 64) for batched |
| Concat [φ(s); e_a] | (B, 320) | 256 + 64 |
| After QHead first hidden | (B, 128) | ReLU activation + Dropout |
| After QHead output | (B, 1) | scalar Q(s,a) |
| After ValueHead (dueling) | (B, 1) | V(s), added to A(s,a) |
| After DynamicsHead | (B, 256) | predicted φ(s') − φ(s) |

## 5. Design decisions

| Decision | Alternative considered | Why we chose this | Trade-off accepted |
|---|---|---|---|
| **Factorized Q(s,a) = f_Q([φ(s); e_a])** | Independent head per action (Linear(d_enc, 1)) | Decouples action identity from Q-function; enables zero-shot Q for new actions | Shared MLP may have limited capacity for action-specific interactions |
| **Action embedding table (learned, per-action)** | Action features from domain knowledge | No natural feature vector for discrete actions; learned embeddings adapt to the task | Requires training to learn embedding structure; cold-start for new actions |
| **k-NN centroid interpolation for new embeddings** | Random init, zero init | Centroid is minimax-optimal with no semantic descriptor; provides structural prior | May collapse all new actions to the same point if embedding space is degenerate |
| **Optimistic exploration bonus (additive, decaying)** | Intrinsic motivation (ICM), count-based bonus | Simple, deterministic, interpretable; decays automatically with visits | May cause Q-overestimation if not clamped; adds a hyperparameter |
| **Progressive freeze (Phase 1 → 2 → 3)** | Full fine-tuning, EWC regularization | Prevents catastrophic forgetting by construction; no auxiliary loss required | Takes longer to converge than full fine-tuning; the optimal freeze duration is unknown |
| **Off-policy (DQN) with replay buffer** | On-policy (PPO, A2C) | Old transitions remain valid after expansion; sample efficiency matters | Off-policy learning is less stable than on-policy; requires target network |
| **Dueling variant centering over known actions only** | Centering over all actions (standard dueling) | Avoids denominator shift from new actions' advantage values | Requires tracking which actions are "known" vs. "new" |
| **Optional auxiliary dynamics loss (L_aux)** | No auxiliary loss, InfoNCE loss | Shapes embedding space to encode transition dynamics; computationally cheap (0.01 weight) | Adds a hyperparameter; may not help if TD loss dominates |

## 6. Domain-specific considerations (RL)

### Temporal discounting

γ = 0.99 enters through the standard TD target:

```
y = r + γ · max_{a'} Q'(s', a')
```

The architecture does not modify how discounting works. After expansion, the TD target for *old* transitions now includes the new action in the max. This is correct because the value of s' genuinely increases when a new action becomes available (monotonicity property of nested action spaces: A₀ ⊂ A₁ ⇒ V*₀(s) ≤ V*₁(s)).

### Exploration

Two complementary mechanisms:

1. **ε-greedy** — provides undirected exploration. ε decays linearly from 1.0 to 0.05 over `epsilon_decay_steps` (default 100,000). After each expansion, ε is temporarily floored at 0.5 during the `expansion_cooldown` window to encourage exploration of the new action.
2. **Optimistic exploration bonus** — an additive bonus for new actions that decays geometrically with each visit (`bonus *= optimistic_bonus_decay`, default 0.99). This provides directed exploration toward the new action without relying solely on ε-greedy randomness.

### On-policy vs. off-policy

**Off-policy (DQN)** is the correct choice for this problem:
- Old transitions remain valid after action-space expansion (action indices are stable)
- The factorized Q-function can evaluate Q(s', a_new) even for transitions recorded before a_new existed
- Sample efficiency matters — the goal is to avoid retraining from scratch

### Observation modality

- **Vector observations** (default): MLP encoder (3 hidden layers, 256 units each) suitable for low-dimensional state spaces
- **Image observations** (configurable): CNN encoder (3 conv layers + 1 FC head) following standard DQN Atari architecture

### Catastrophic forgetting prevention

Three mechanisms work together:

1. **Architectural**: Factorized Q decouples action-specific parameters. Old action embeddings are frozen during Phase 1-2 via a gradient hook that zeroes gradients for frozen rows.
2. **Regularization**: The replay buffer naturally rehearses old transitions, preventing the network from drifting during fine-tuning on the new action.
3. **Optimization**: Progressive unfreezing (Phase 1 → 2 → 3) with reduced LR (0.1×) in Phase 3 prevents large gradient updates from disrupting learned representations.

## 7. Action expansion flow

```
  ┌──────────┐     ┌──────────────┐     ┌─────────────────┐     ┌───────────────┐
  │ Detect   │────│ Create new   │────│ Initialize via  │────│ Freeze old    │
  │ new      │     │ embedding    │     │ k-NN in action  │     │ weights;      │
  │ action   │     │ row          │     │ embedding space │     │ add exploration│
  │ a_new    │     │              │     │                  │     │ bonus         │
  └──────────┘     └──────────────┘     └─────────────────┘     └───────┬───────┘
                                                                        │
                                                                        v
  ┌──────────┐     ┌──────────────┐     ┌─────────────────┐     ┌───────────────┐
  │ Evaluate │────│ Unfreeze all │────│ Unfreeze Q-head │────│ Train only    │
  │ metrics  │     │ (low LR)     │     │ (keep encoder   │     │ new embedding │
  │          │     │ Phase 3      │     │  frozen)        │     │ Phase 1       │
  │          │     │              │     │ Phase 2         │     │               │
  └──────────┘     └──────────────┘     └─────────────────┘     └───────────────┘
```

**Phase timeline (configurable via `freeze_encoder_steps`, `freeze_q_head_steps`):**

```
     [expansion detected]
               |
               v
     +----------------------+----------------------+----------------------+
     |     Phase 1          |     Phase 2          |     Phase 3          |
     |  freeze_encoder      |  freeze_encoder      |  all unfrozen        |
     |  freeze_q_head       |  unfreeze_q_head     |  lr x 0.1            |
     |  train: new e_a only |  train: e_a + Q-head |  train: everything   |
     |  0 -- N1 steps       |  N1 -- N2 steps      |  N2 -- infty        |
```

**Replay buffer handling with expanding action space:**

Before expansion (|A| = 3):          After expansion (|A| = 4):
```
+---------------------+             +---------------------+
| (s, a=0, r, s', d)  |             | (s, a=0, r, s', d)  |  ← old (still valid)
| (s, a=2, r, s', d)  |             | (s, a=2, r, s', d)  |  ← old
| (s, a=1, r, s', d)  |             | (s, a=1, r, s', d)  |  ← old
| ...                  |             | (s, a=3, r, s', d)  |  ← new (action index 3)
+---------------------+             | (s, a=0, r, s', d)  |  ← new
                                      +---------------------+
```
All transitions remain valid: action indices are stable (old actions keep their indices; new actions get new indices). The TD target naturally includes the new action in max_{a'} Q'(s', a') for all transitions.

## 8. Inductive bias justification

| Design choice | One-sentence justification |
|---|---|
| **Factorized Q(s,a) = f_Q(φ(s), e_a)** | Decouples action identity from the Q-function, enabling zero-shot Q-values for unseen actions via embedding similarity. |
| **Action embedding table** | Discrete actions have no natural feature vector; a learned embedding per action allows the Q-function to exploit similarity structure. |
| **k-NN interpolation for new embedding** | Initialises new actions at the centroid of semantically similar known actions, providing a structural prior that reduces cold-start error. |
| **k-NN centroid of ALL known embeddings** | Without semantic descriptors of the new action, the centroid of ALL existing embeddings is the least-committal initialisation (minimax optimal). |
| **Optimistic exploration bonus for new actions** | Counteracts learned pessimism (new actions start with zero visit count) and encourages systematic exploration of the new action. |
| **ε-greedy exploration** | Simple, well-understood, compatible with discrete action spaces; the exploration bonus provides directed exploration on top of uniform randomness. |
| **Progressive freeze schedule (Phase 1→3)** | Prevents catastrophic forgetting by isolating the new action's parameters during initial learning, then gradually re-introducing plasticity. |
| **Freeze encoder first** | State representations learned on old actions transfer perfectly to new actions (the state space has not changed); no need to disrupt them. |
| **Off-policy replay (DQN)** | Maximally sample-efficient; old transitions remain usable after expansion because action indices are stable and the TD target naturally includes new actions. |
| **Optional auxiliary dynamics loss** | Shapes the action embedding space to encode transition dynamics (which actions cause which state changes), creating a semantically meaningful manifold. |
| **Pre-norm (LayerNorm on φ(s))** | Stabilises the gradient flow through the shared encoder, especially important when training dynamics change during expansion phases. |
| **Separate value head (dueling variant)** | V(s) is completely unaffected by action-space expansion; only the relative advantages of actions need to be adjusted for the new action. |

## 9. Research-to-architecture traceability

| Research contract item | Architecture decision | Status | Validation hook |
|---|---|---|---|
| **Novelty claim:** No deep RL method handles unexpected action-space expansion | `ActionIncrementalDQN` with `handle_action_expansion()` dynamic table growth | `TODO: unverified` | Benchmark: DAVE-game analogue. Metric: Can the agent use a_new within N steps of expansion? |
| **Hypothesis:** Similarity-based Q-transfer provides better-than-random initialisation | `ActionEmbeddingTable.add_embedding()` with k-NN interpolation | `hypothesis` | Ablation: Compare k-NN init vs. random init for new action Q-values at first encounter |
| **Hypothesis:** Factorized Q minimises interference | `QHead` shared MLP operating on [φ(s); e_a] concatenation | `hypothesis` | Ablation: Compare factorized vs. independent-head architecture on old-action Q degradation |
| **Hypothesis:** Freezing encoder prevents catastrophic forgetting | `_apply_freeze_phase1()` — encoder frozen for `freeze_encoder_steps` | `hypothesis` | Compare freeze vs. no-freeze: Q-value degradation on old actions after expansion |
| **Hypothesis:** Softmax denominator shift degrades old action logits | Dueling variant with known-action-only centering (`DuelingFactorizedQ`) | `hypothesis` | Measure old-action logit shift before/after expansion with and without centering |
| **Gap:** Action embedding structure for zero-shot transfer | `use_auxiliary_dynamics_loss` — dynamics head predicts φ(s') − φ(s) from e_a | `hypothesis` | Compare t-SNE of embeddings with vs. without aux loss; measure k-NN init quality |
| **Baseline:** Retrain from scratch | Separate `train.py` supports --env flag; retrain-from-scratch wrapper not yet implemented | `TODO: unverified` | Compare total env steps to reach equivalent performance |
| **Baseline:** Zero-initialisation | Ablation: set `k_nn=0` to use random embedding init | `grounded` | Ablation flag in config |
| **Evaluation:** 10 seeds, mean ± std | `seed` field in config; training loop logs seed | `TODO: unverified` | CI check: `assert len(results.seeds) >= 10` |
| **Evaluation:** Q-value degradation metric | `get_q_for_actions()` logs Q-values for fixed state-action pairs before/after expansion | `grounded` | Assert: ΔQ_old < 0.2 * Q_old_pre_expansion |
| **Blocking unknown:** Embedding structure for semantically novel actions | `embedding_similarity`, `k_nn`, and centroid-based init | `TODO: unverified` | If k-NN init ≈ random init performance, core hypothesis is falsified |

## 10. Known limitations

- **No custom benchmark environment yet** — the DAVE-game analogue, continuous control, and goal-conditioned scenarios all raise `NotImplementedError`. All evaluations use synthetic proxy metrics (Q-stats on random states) rather than actual environment interaction rewards.
- **Gradient hook freeze mask is not parameter-norm-preserving** — when `freeze_old_embeddings=True`, the gradient for old rows is zeroed via a hook, but the forward pass still computes updates through the new embedding's interactions. This is correct for Phase 1 but means old embeddings are not truly "frozen" in the sense of zero gradient flow through the computation graph.
- **k-NN init collapses if embedding space lacks structure** — if the auxiliary dynamics loss does not create semantically meaningful embeddings, the k-NN centroid initialisation degrades to a random or near-zero embedding, falsifying the core hypothesis. No embedding-space quality analysis (t-SNE, similarity matrix) has been performed yet.
- **Single seed only** — all tests and ablations use a single seed. No statistical rigour (10 seeds, mean ± std) has been applied.
- **No retrain-from-scratch baseline** — the sub-linear fine-tuning claim cannot be validated without a comparison script that trains the same architecture from scratch with the full action set from episode 0.
- **No fixed-action-oracle baseline** — the upper-bound achievable performance is unknown.
- **CartPole demoware only** — the `ExpandingActionWrapper` maps new actions to a safe default (action 0), which means the agent only experiences the actual effect of a new action if the environment natively supports it. In CartPole, the new action is a no-op, which is degenerate.
- **Optimistic bonus may cause Q-overestimation** — the additive bonus shifts Q-values and may propagate through TD backups. Double-DQN and bonus clamping are noted as mitigations but not implemented.
- **Phase 3 LR reset may be too aggressive** — the learning rate is reduced by 0.1× globally at Phase 3 entry. This may be too low for the encoder to adapt meaningfully, or it may cause the encoder to converge to a poor local optimum before adaptation completes.

> TODO: unverified — all of the above limitations are known and documented upstream. The `research_eval/scorecard.json` identifies these as blocking gaps. Full empirical validation requires the custom benchmark environment as a prerequisite.
