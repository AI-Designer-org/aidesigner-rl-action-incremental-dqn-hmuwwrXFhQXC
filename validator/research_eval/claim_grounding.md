# Claim Grounding — Action Space Incremental RL

Every research claim in the lifecycle contract must point to evidence (file,
function, test, command, or `TODO: unverified`). This document enforces that.

---

## Novelty Claims

### Claim 1: No deep RL method handles truly unexpected action-space expansion mid-training without a predefined hierarchy.

**Status:** `TODO: unverified`

**Evidence:**
- Research contract (ACTION_SPACE_INCREMENTAL_RL.md, lines 49-55) identifies Gap 1
  and cites Farquhar et al. (requires hierarchy), de la Rosa et al. (tabular), and
  Sinii et al. (bandit/MDP only).
- No test or benchmark compares the proposed method against these baselines on
  the unexpected-expansion setting.
- The claim is definitional (literature analysis) and cannot be verified by the
  current codebase alone — it requires a literature review + empirical comparison.

**Grounding:** `research/ACTION_SPACE_INCREMENTAL_RL.md` lines 49-55 (gap analysis) + literature survey.

---

### Claim 2: Similarity-based Q-value initialization for newly added actions provides better-than-random zero-shot estimates.

**Status:** `hypothesis`

**Evidence:**
- `networks.py::ActionEmbeddingTable.add_embedding` (lines 108-146): implements
  k-NN centroid interpolation for new embeddings.
- `agent.py::ActionIncrementalDQN.handle_action_expansion` (lines 277-324): calls
  `add_embedding` during action expansion.
- `validator/run_ablations.py`: `no_knn_init` ablation (k_nn=3 → k_nn=0) compares
  k-NN init against random init.
- `validator/test_model.py::TestRLBenchmarks::test_monotonicity_property`: checks
  that max Q over the expanded set is ≥ max Q over the original set.
- `validator/test_model.py::TestRLBenchmarks::test_q_degradation_metric`: measures
  old-action Q-values before/after expansion (no training).

**Missing:** No training-based comparison of zero-shot Q(a_new) with k-NN init
vs. random init on an actual RL task. Proxy metrics only.

**Grounding:** `validator/run_ablations.py` (no_knn_init condition),
`networks.py::ActionEmbeddingTable.add_embedding`,
`agent.py::ActionIncrementalDQN.handle_action_expansion`.

---

### Claim 3: Factorized Q(s,a) = f_Q(φ(s), e_a) minimizes interference between old and new action values during fine-tuning.

**Status:** `hypothesis`

**Evidence:**
- `networks.py::QHead` implements the shared MLP operating on `[φ(s); e_a]`.
- `networks.py::DuelingFactorizedQ` implements the separation of V(s) and A(s,a).
- `validator/run_ablations.py`: `dueling_q` ablation compares factorized vs.
  dueling_factorized variants.
- `validator/test_model.py::TestGradients::test_gradient_flow_after_expansion`:
  verifies that in Phase 1, only the new embedding receives gradients.
- `validator/test_model.py::TestRLBenchmarks::test_freezing_prevents_encoder_drift`:
  verifies encoder weights are unchanged in Phase 1.

**Missing:** No direct comparison of old-action Q-degradation between factorized
and independent-head architectures (where each action has its own `Linear(d_enc, 1)`).

**Grounding:** `networks.py::QHead`, `networks.py::DuelingFactorizedQ`,
`validator/run_ablations.py` (dueling_q condition),
`validator/test_model.py::TestGradients::test_gradient_flow_after_expansion`.

---

## Architectural Claims

### Claim 4: Freezing the base encoder during initial fine-tuning prevents catastrophic forgetting of old actions.

**Status:** `hypothesis`

**Evidence:**
- `agent.py::ActionIncrementalDQN._apply_freeze_phase1` (lines 502-536): freezes
  encoder, Q-head, and old embeddings.
- `agent.py::ActionIncrementalDQN._apply_freeze_phase2` (lines 538-555): unfreezes
  Q-head, keeps encoder frozen.
- `agent.py::ActionIncrementalDQN._apply_freeze_phase3` (lines 557-577): unfreezes
  all, reduces LR.
- `validator/run_ablations.py`: `no_freeze_encoder`, `short_freeze`, `long_freeze`
  ablations.
- `validator/test_model.py::TestRLBenchmarks::test_freezing_prevents_encoder_drift`:
  verifies encoder weights are invariant in Phase 1.
- `validator/test_model.py::TestGradients::test_gradient_flow_after_expansion`:
  verifies encoder does not receive gradients in Phase 1.
- `validator/test_model.py::TestRLProperties::test_freeze_schedule_progression`:
  verifies Phase 1 → 2 → 3 transitions.

**Missing:** No direct measurement of old-action Q-degradation with freeze vs.
without freeze during actual training.

**Grounding:** `agent.py::ActionIncrementalDQN._apply_freeze_phase1`,
`validator/test_model.py::TestRLBenchmarks::test_freezing_prevents_encoder_drift`,
`validator/test_model.py::TestGradients::test_gradient_flow_after_expansion`,
`validator/run_ablations.py` (no_freeze_encoder condition).

---

### Claim 5: Softmax denominator shift from adding new actions causes measurable degradation in existing action logits.

**Status:** `hypothesis`

**Evidence:**
- `networks.py::DuelingFactorizedQ.forward` (lines 381-411): implements
  known-action-only centering to mitigate denominator shift.
- `agent.py::ActionIncrementalDQN._batch_q_values` (lines 378-422): in dueling
  variant, centers advantages over known actions only.
- `validator/test_model.py::TestRLBenchmarks::test_dueling_centering_correctness`:
  verifies that known-action advantages sum to ~zero.

**Missing:** No direct measurement of old-action logit shift before/after
expansion. The centering mitigation is implemented but the degradation
magnitude is not measured.

**Grounding:** `networks.py::DuelingFactorizedQ.forward`,
`agent.py::ActionIncrementalDQN._batch_q_values` (dueling branch),
`validator/test_model.py::TestRLBenchmarks::test_dueling_centering_correctness`.

---

### Claim 6: Action embedding space learned with auxiliary dynamics loss creates semantically meaningful structure.

**Status:** `hypothesis`

**Evidence:**
- `networks.py::DynamicsHead` implements φ(s') - φ(s) prediction.
- `agent.py::ActionIncrementalDQN.update` (lines 217-275): includes auxiliary
  dynamics loss with coefficient 0.01.
- `validator/run_ablations.py`: `no_aux_loss` ablation.
- `validator/test_model.py::TestRLBenchmarks::test_aux_loss_reduces_mse`: verifies
  aux loss is non-negative MSE-like quantity.

**Missing:** No embedding visualization (t-SNE), no cosine similarity matrix
analysis, no comparison of k-NN init quality with vs. without aux loss.
The claim that aux loss produces "semantically meaningful" embeddings is
an assertion without evidence.

**Grounding:** `networks.py::DynamicsHead`, `agent.py::ActionIncrementalDQN.update`,
`validator/run_ablations.py` (no_aux_loss condition).

---

## Grounded Claims (from literature)

These claims are imported from the research contract and are not verified by
this codebase. They are based on published literature.

| Claim | Source | Code evidence |
|---|---|---|
| Nested action hierarchy enables monotonic value transfer | Farquhar et al., ICML 2020 | `agent.py::update` uses max over all actions in TD target (correctly implements monotonicity assumption) |
| Action embeddings can generalize across action-set sizes | Sinii et al., 2023 | `networks.py::ActionEmbeddingTable` + `add_embedding` |
| Progressive networks prevent catastrophic forgetting | Rusu et al., 2016 | `agent.py::_apply_freeze_phase1/2/3` progressive unfreezing (inspired by PNN) |

---

## Falsification Conditions

The core hypothesis is falsified if any of the following conditions are met.
Each condition's verification status is documented.

### Condition 1: Zero-shot Q(a_new) is indistinguishable from random initialization.

**Verification:** `validator/run_ablations.py::no_knn_init` compares k-NN init
(k_nn=3) against random init (k_nn=0). The proxy metric measures Q-stats on
random states, NOT actual zero-shot performance on an RL task.

**TODO:** Run on actual environment: compare Q(a_new) at first encounter after
expansion for k-NN init vs. random init. Requires custom environment.

---

### Condition 2: Fine-tuning requires ≥ 80% of steps needed to retrain from scratch.

**Verification:** No comparison script exists. Requires custom environment +
retrain-from-scratch baseline.

**TODO:** Implement retrain-from-scratch script and compare total environment
steps to reach equivalent performance.

---

### Condition 3: Q-values of old actions degrade by > 20%.

**Verification:** `validator/test_model.py::TestRLBenchmarks::test_q_degradation_metric`
measures old-action Q-values before/after expansion. With frozen params (Phase 1),
Q-values are identical (passes with atol=1e-6).

**TODO:** Measure Q-degradation through full training (Phase 1 → 2 → 3) with
training steps. Current test only checks immediately after expansion with
no training.

---

### Condition 4: Performance on the original action set degrades below 80% of pre-expansion level.

**Verification:** No training-based performance metric exists.

**TODO:** Requires custom environment where per-action-set performance can
be measured separately.

---

## Summary of TODO Items

| # | Item | Priority | Owner |
|---|---|---|---|
| 1 | Custom Gymnasium environment (DAVE-game analogue) | P0 | validator (next stage) |
| 2 | Retrain-from-scratch baseline script | P0 | validator (next stage) |
| 3 | Zero-shot Q(a_new) comparison (k-NN vs. random) | P1 | validator (next stage) |
| 4 | 10-seed evaluation harness | P1 | validator (next stage) |
| 5 | Embedding space visualization (t-SNE, similarity matrix) | P1 | validator (next stage) |
| 6 | Q-degradation measurement through full training | P2 | validator (next stage) |
| 7 | Hyperparameter sensitivity sweep | P2 | validator (next stage) |
| 8 | Fixed-action-oracle baseline | P2 | validator (next stage) |
| 9 | Continuous control scenario | P3 | validator (next stage) |
| 10 | Paper / results section | P3 | validator (next stage) |
