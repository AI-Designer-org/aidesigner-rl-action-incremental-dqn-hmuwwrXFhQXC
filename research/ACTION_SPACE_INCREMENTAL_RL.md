# Action Space Incremental Reinforcement Learning

## Research Synthesis & Lifecycle Contract

---

## 1. Landscape Summary

### Problem Definition
Standard RL assumes a fixed action space throughout training and deployment. In real-world settings (video games, robotics, skill acquisition), new actions become available at runtime. Retraining from scratch wastes prior experience. The core challenge: **learn to incorporate new actions without catastrophic forgetting of existing knowledge.**

### Key Architecture Families

| Family | Representative Works | Core Mechanism | Handles Unexpected Addition? | Scales to Deep RL? |
|---|---|---|---|---|
| **Nested Curriculum** | Farquhar et al., "Growing Action Spaces" (ICML 2020) | Predefined action hierarchy; off-policy transfer + parent-child Q-decomposition | No — hierarchy must be known in advance | Yes — DQN on StarCraft |
| **Drift-Triggered Expansion** | de la Rosa et al., "morphin" (arXiv 2026) | Page-Hinkley drift detection; Q-table row insertion; adaptive ε/α | Yes — on-the-fly detection | No — tabular only |
| **In-Context Generalization** | Sinii et al., "Headless-AD" (NeurIPS 2023 workshop) | Action embeddings + InfoNCE loss; random action features at train time | Yes — handles unseen action counts at test time | Yes — transformer |
| **Skill Incremental** | iManip (2025), TOPIC (2025) | Temporal replay; extendable action prompts; task relation graphs | Partial — designed for skill sequences, not single-policy expansion | Yes — PerceiverIO / Transformer |
| **Progressive Architecture** | Rusu et al., "Progressive Neural Networks" (DeepMind 2016) | Frozen prior columns + lateral connections for new tasks | Yes — new column per task | Yes — but linear parameter growth |
| **Feature-Based Q Generalization** | Community proposals (StackExchange, 2018) | Q(s, a) = f(φ(s), ψ(a)); generalize via action features | Partial — requires action feature engineering | Yes — but ad-hoc |

### Core Insight: The Monotonicity Property

For nested action spaces A₀ ⊂ A₁ ⊂ ... ⊂ A_{N-1}, the optimal value function satisfies:

    V*_i(s) ≤ V*_j(s)   for all s if i < j

This ensures larger action spaces never degrade optimal value — foundational for value transfer (Farquhar et al., 2020). However, this property *only holds for predefined nested supersets*, not for arbitrary unexpected action additions. **This is the critical gap for the DAVE-game scenario.**

---

## 2. Complexity / Properties Table

| Method | Time Complexity | Space Complexity | Expressiveness | Parallelism | Hardware Fit |
|---|---|---|---|---|---|
| **GAS (Farquhar)** | O(N·|A_max|·d) forward | O(N·|A_max|·d) params | Hierarchical action structure; fails if hierarchy misspecified | Episode-level parallel; sequential hierarchy scan | GPU-friendly (shared torso) |
| **morphin (de la Rosa)** | O(|S|·|A|) per update | O(|S|·|A|) table | Tabular only; no generalization across states | Trivially parallel envs | CPU only |
| **Headless-AD (Sinii)** | O(T²) transformer | O(L²) transformer + O(|A|·d_emb) | Handles variable action sets at test time; limited to bandit/MDP | Full sequence parallelism | GPU-friendly |
| **Progressive Net (Rusu)** | O(K·d) forward | O(K·d) params (linear in tasks) | Strong; separate capacity per task | Independent columns parallel | GPU-friendly |
| **iManip (2025)** | O(T·d) + replay overhead | O(d) + O(|skills|·d_prompt) | Skill-sequence specific; prompt-based | Task-parallel prompts | GPU-friendly |

**Legend:** N = hierarchy levels, |A_max| = max action count, d = hidden dimension, L = sequence length, K = number of tasks.

---

## 3. Novelty Gaps

### Gap 1: Unexpected Action-Space Expansion in Deep RL

**What exists:** Farquhar et al. (2020) handles *predefined* nested action-space growth. The hierarchy A₀ ⊂ A₁ ⊂ ... must be specified before training, and actions grow according to a known curriculum.

**What remains missing:** No deep RL method handles the scenario where a *truly unexpected* action appears mid-training (e.g., "shoot" unlocked upon reaching game level 3). The hierarchy cannot be known in advance because the action's semantics are unavailable.

**Status:** `TODO: unverified` — no named paper addresses this exact setting in deep RL.

---

### Gap 2: Q-Value Initialization for Novel Actions Without Retraining

**What exists:** Morphin (de la Rosa et al., 2026) initializes new Q-table rows to zero — acceptable in tabular settings but poor in deep RL where the network must be structurally modified. Progressive Networks (Rusu et al., 2016) add entire new columns, which is expensive and doesn't integrate new actions into the *same* policy.

**What remains missing:** A principled method to initialize Q-values (or policy logits) for a newly added action using the structure of known actions — e.g., via action embeddings, learned similarity, or parent-action decomposition from Farquhar et al.

**Status:** `grounded hypothesis` — Farquhar et al. shows parent-child transfer works for predefined hierarchies; the open question is whether this can be adapted for *unexpected* additions where the hierarchy must be inferred.

---

### Gap 3: Stable Fine-Tuning After Network Expansion

**What exists:** Catastrophic forgetting in continual learning is well-studied (EWC, progressive nets, replay). But these focus on *task* boundaries, not *action-space* boundaries within a single task.

**What remains missing:** When a new action is added to a policy head, the existing action logits shift because the softmax denominator changes. The Q-values of old actions may degrade during fine-tuning on the new action. No existing work quantifies this effect or proposes mitigations targeted at action-space expansion.

**Status:** `hypothesis` — plausible from first principles; no empirical literature validates this claim specifically.

---

### Gap 4: Standardized Benchmark for Action-Space Incremental RL

**What exists:** Atari provides no mechanism for expanding action spaces mid-episode. DM Control doesn't either. StarCraft has naturally varying unit counts but not action-set changes.

**What remains missing:** A standardized benchmark (gymnasium-compatible) where the action space grows at known milestones, with clear evaluation protocols. The DAVE-game scenario is a concrete exemplar but no gym environment exists.

**Status:** `TODO: unverified`

---

### Gap 5: Detection of Action-Space Drift Without Prior Knowledge

**What exists:** Morphin uses Page-Hinkley for drift detection but fails when the new reward distribution is a subset of previously seen values. No other method treats action-space expansion as a *detection* problem.

**What remains missing:** Reliable detection that a new action has become available (vs. natural variance in returns) would allow autonomous network expansion without oracle knowledge.

**Status:** `grounded` — de la Rosa et al. explicitly identify this limitation; their PH-test failed on a second drift in traffic control.

---

### Gap 6: Action Embedding Spaces That Generalize Across Introduced Actions

**What exists:** Headless-AD (Sinii et al.) uses random action embeddings during training and InfoNCE loss to align action semantics. It generalizes to unseen action counts at test time but doesn't handle *incremental introduction* during training.

**What remains missing:** An action encoder that: (a) maps each action to a semantically meaningful embedding, (b) can accommodate new action embeddings mid-training, and (c) facilitates zero-shot or few-shot Q-value estimates for new actions by interpolating in embedding space.

**Status:** `hypothesis` — plausible from Headless-AD + GAS synthesis; no single work combines both.

---

## 4. Recommended Research Direction

### Hypothesis

> **A factorized action-value architecture with a learnable action embedding space and a similarity-based Q-transfer mechanism can incorporate unexpectedly introduced actions into a deep RL policy without catastrophic forgetting and with sub-linear fine-tuning cost relative to retraining from scratch.**

### Architecture Sketch

1. **Base encoder** φ: S → ℝ^d (shared, any architecture — CNN, ViT, etc.)
2. **Action embedding table** E ∈ ℝ^{|A| × d_emb}: each action a maps to embedding e_a
3. **Action-conditional Q-function**:
   Q(s, a) = f_Q(φ(s), e_a)  where f_Q is a small MLP
4. **Similarity-based initialization for new action a_new**:
   e_{a_new} ← mean pooling over embeddings of the k-nearest known actions (distance in embedding space)
   Q(s, a_new) ← max_{a_known} Q(s, a_known) — optimistic initialization
5. **Fine-tuning protocol**: freeze base encoder φ, finetune only the new embedding e_{a_new} and f_Q for N steps, then unfreeze all.

### Key Design Choices to Validate

| Choice | Options | Trade-off |
|---|---|---|
| Q-function form | Additive: Q(s,a) = V(s) + A(φ(s), e_a) | Dueling-style; reduces interference |
| Embedding init | Uniform random vs. k-NN interpolation | Random provides less bias but slower |
| Optimistic bias | max vs mean Q of known actions | max encourages exploration but may overestimate |
| Freeze schedule | Full freeze → partial → all | Prevents forgetting vs. allows adaptation |

### Expected Observable Behavior

1. **Zero-shot transfer**: On first encounter of new action a_new, the agent's Q-value estimates for a_new are above random, and the agent occasionally selects a_new in states where similar known actions are valuable.
2. **Sub-linear fine-tuning**: After N steps of fine-tuning the expanded network, performance recovers to the level of an oracle that was trained from scratch with a_new from the beginning — requiring fewer total environment steps than retraining.
3. **No forgetting**: Q-values for old actions (A_old) do not degrade by more than 5% during fine-tuning.

### Falsification Conditions

The hypothesis is falsified if any of the following hold:
1. **Zero-shot Q(a_new) is indistinguishable from random initialization** (no structure in the embedding space transfers).
2. **Fine-tuning requires ≥ 80% of the steps needed to retrain from scratch** (sub-linear claim fails).
3. **Q-values of old actions degrade by > 20%** (catastrophic forgetting during action integration).
4. **Performance on the original action set degrades below 80% of the pre-expansion level** (overall capability loss).

---

## 5. Research Lifecycle Contract

```yaml
task_level: level_1
domain: RL
research_question: >
  Can a factorized action-value architecture with learnable action embeddings
  incorporate unexpectedly introduced actions into a deep RL policy without
  catastrophic forgetting and with sub-linear fine-tuning cost?

novelty_claims:
  - claim: >
      No existing deep RL method handles truly unexpected action-space expansion
      mid-training without a predefined hierarchy.
    status: "TODO: unverified"
    evidence: >
      Farquhar et al. (ICML 2020) requires a known nested hierarchy A_0 ⊂ A_1 ⊂ ...
      Morphin (de la Rosa et al., 2026) is tabular only. Headless-AD (Sinii et al.,
      2023) handles variable action sets at test time but not incremental addition
      during training.

  - claim: >
      A similarity-based Q-value initialization for newly added actions can provide
      meaningful zero-shot estimates by interpolating in a learned action embedding
      space.
    status: hypothesis
    evidence: >
      Parent-child Q-decomposition in GAS (Farquhar et al., 2020) shows that value
      transfer across action resolutions is effective. Headless-AD shows action
      embeddings can generalize across action set sizes. No work combines both for
      unexpected additions.

  - claim: >
      Softmax denominator shift from adding new actions causes measurable
      degradation in existing action logits during fine-tuning.
    status: hypothesis
    evidence: >
      Plausible from first principles (softmax normalization depends on all logits).
      No empirical literature quantifies this for action-space expansion specifically.

known_related_work:
  - work: "Growing Action Spaces — Farquhar et al., ICML 2020"
    covers: >
      Off-policy RL with nested predefined action hierarchies. Parent-child Q-value
      transfer. Demonstrated on discretized continuous control and StarCraft
      micromanagement with 50-100 agents.
    leaves_open: >
      Requires known action hierarchy before training. Cannot handle truly unexpected
      action additions. Requires hand-tuned curriculum schedule. No mechanism for
      detecting when new actions become available.

  - work: "morphin — de la Rosa et al., arXiv 2026"
    covers: >
      On-the-fly Q-table expansion with Page-Hinkley drift detection. Adaptive
      exploration (ε) and learning rate (α). Demonstrated on Gridworld and traffic
      signal control.
    leaves_open: >
      Tabular only — does not scale to deep RL with function approximation. Drift
      detection fails when new reward distribution is subset of previously seen
      values. Hyperparameters require empirical tuning.

  - work: "Headless-AD — Sinii et al., NeurIPS 2023 workshop"
    covers: >
      In-context RL for variable action spaces. Random action embeddings at training
      time; InfoNCE loss for action semantics. Handles more actions at test time
      than seen during training.
    leaves_open: >
      Bandit/MDP settings only, not deep RL with complex state spaces. Does not
      address incremental introduction during training — only generalization to
      larger fixed sets at test time.

  - work: "Progressive Neural Networks — Rusu et al., DeepMind 2016"
    covers: >
      Task-incremental learning with frozen prior columns and lateral connections.
      Prevents catastrophic forgetting by construction. Tested on Atari and 3D
      mazes.
    leaves_open: >
      Linear parameter growth in number of tasks. Requires task identity at test
      time. Adds entire network columns per task, not lightweight action-head
      expansion. No backward transfer.

  - work: "iManip — arXiv 2025; TOPIC — arXiv 2025"
    covers: >
      Skill-incremental learning with temporal replay and extendable action prompts
      for robotic manipulation. TOPIC adds task-specific prompts with task relation
      graphs.
    leaves_open: >
      Designed for robotic manipulation skill sequences, not general deep RL.
      Depends on prompt engineering and pre-defined task boundaries.

baseline_requirements:
  - "Retrain from scratch: train the same DQN/SAC architecture with the expanded
     action space from episode 0, compare total environment steps to reach
     equivalent performance."
  - "Fixed-action-space upper bound: train with the full action set from the
     beginning (assumes oracle knowledge of all future actions). This upper-bounds
     achievable performance."
  - "Zero-initialization baseline: naively expand network head with zero-initialized
     weights for new actions, no embedding transfer or optimistic initialization."
  - "GAS adapted baseline: if actions can be partially ordered (even post-hoc),
     apply Farquhar et al.'s parent-child Q-decomposition as a strong baseline."

evaluation_requirements:
  - "Benchmark: custom Gymnasium environment with expanding action sets.
     Minimum 3 scenarios: (a) DAVE-game analogue (actions added at level
     transitions), (b) continuous control with new effectors (e.g., robot with new
     gripper added), (c) goal-conditioned with new affordances unlocked."
  - "Metrics: (1) cumulative reward, (2) steps to recover pre-expansion performance,
     (3) Q-value degradation on old actions post-expansion, (4) % of total training
     steps saved vs. retrain-from-scratch."
  - "Ablations: (1) embedding initialization strategy (zero, random, k-NN mean,
     optimistic max), (2) freeze schedule (full freeze, head-only, all unfrozen),
     (3) Q-function form (additive dueling vs. multiplicative)."
  - "Statistical rigor: 10 random seeds per condition; report mean ± std.
     Effect size (Cohen's d) vs. retrain-from-scratch baseline."

blocking_unknowns:
  - "Does the action embedding space learned on the initial action set capture
     enough structure to provide useful Q-initialization for semantically novel
     actions (e.g., 'shoot' when only movement actions exist)? If not, the
     k-NN interpolation collapses to random initialization and the core
     hypothesis fails."
  - "Are there domains where the optimal policy after action addition is so
     different from before that any parameter freeze strategy prevents reaching
     the new optimum? This would constrain the hypothesis to domains with
     compositional action semantics."
  - "How catastrophic is the softmax denominator shift when |A| grows by 1 vs.
     doubling? At what scale does the shift become negligible, and is that scale
     reached in practice?"
  - "Can the drift-detection problem (Gap 5) be solved reliably in deep RL, or
     does the method require oracle knowledge of when new actions become
     available?"

claim_status:
  grounded:
    - "Nested action hierarchy enables provable monotonic value transfer
      (Farquhar et al., ICML 2020)."
    - "Page-Hinkley drift detection enables on-the-fly detection of action-set
      changes in tabular settings (de la Rosa et al., 2026)."
    - "Action embeddings via InfoNCE can generalize across action-set sizes
      (Sinii et al., NeurIPS 2023 workshop)."
    - "Progressive networks prevent catastrophic forgetting at the cost of
      linear parameter growth (Rusu et al., 2016)."
  hypotheses:
    - "Similarity-based Q-transfer from known to new actions in embedding space
      provides better-than-random initialization."
    - "A factorized Q(s,a) = f_Q(φ(s), e_a) architecture minimizes interference
      between old and new action values during fine-tuning."
    - "Freezing the base encoder during initial fine-tuning prevents catastrophic
      forgetting of old actions."
  TODO_unverified:
    - "No deep RL method handles truly unexpected action-space expansion without
      a predefined hierarchy or tabular state space."
    - "No standardized Gymnasium benchmark exists for action-space incremental RL."
    - "The quantitative impact of softmax denominator shift on old-action logits
      during action expansion is unmeasured."
```

---

## 6. Recommended Next Steps

1. **Construct the benchmark**: Build a Gymnasium environment with expanding action sets (DAVE-game analogue, continuous control, goal-conditioned affordances). Define evaluation protocol with 10 seeds.
2. **Implement the baseline suite**: DQN with retrain-from-scratch, fixed-action-oracle, zero-init expansion, and GAS-adapted (if action hierarchy can be post-hoc imposed).
3. **Implement the proposed architecture**: Factorized Q(s,a) with action embedding table. Start with k-NN interpolation + optimistic max for new-action Q-initialization.
4. **Run ablation sweep**: Embedding init strategies × freeze schedules × Q-function forms. This is the minimum experiment to validate or falsify the core hypothesis.
5. **If hypothesis holds**: Scale to more complex domains (Atari with action unlocks, procedural game levels). If falsified, investigate which component failed (embedding structure? interference during fine-tuning?).

---

## Sources

- Farquhar et al., "Growing Action Spaces" — ICML 2020. [ar5iv](https://ar5iv.labs.arxiv.org/html/1906.12266)
- de la Rosa et al., "morphin: Self-Adaptive Q-Learning for Changing Action Spaces" — arXiv 2026. [arxiv](https://arxiv.org/html/2601.20714v1)
- Sinii et al., "In-Context RL with Variable Action Spaces" — NeurIPS 2023 workshop. [arXiv](http://arxiv.org/pdf/2312.13327v1)
- Rusu et al., "Progressive Neural Networks" — DeepMind 2016. [ar5iv](https://ar5iv.labs.arxiv.org/html/1606.04671)
- Venkatesan & Er, "A novel progressive learning technique for multi-class classification" — Neurocomputing 2016. [arXiv](https://arxiv.org/abs/1609.00085)
- Mnih et al., "Human-level control through deep reinforcement learning" — Nature 2015. (DQN, standard RL reference)
- Seyde et al., "Growing Q-Networks (GQN)" — PMLR 2024. [PMLR](https://proceedings.mlr.press/v242/seyde24a/seyde24a.pdf)
- iManip: "Skill-Incremental Learning for Robotic Manipulation" — arXiv 2025.
- TOPIC: "Few-Shot Vision-Language Action-Incremental Policy Learning" — arXiv 2025.
