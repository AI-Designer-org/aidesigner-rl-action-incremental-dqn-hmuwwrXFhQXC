# Experiment Coverage — Action Space Incremental RL

This document maps required experiments from the research lifecycle contract
against what is implemented in the validator, coder, and architect stages.

---

## Legend

| Symbol | Meaning |
|---|---|
| ✓ | Implemented and passing |
| ~ | Partially implemented (proxy / stub) |
| ✗ | Not implemented |
| ? | Not applicable / unclear |

---

## Baselines Required (from Research Contract)

| Baseline | Status | Location | Notes |
|---|---|---|---|
| Retrain from scratch | ✗ | N/A | Needs custom environment + full training run. `train.py` can support but no comparison script exists. |
| Fixed-action oracle | ✗ | N/A | Needs custom environment with full action set from episode 0. |
| Zero-initialization expansion | ~ | `run_ablations.py::no_knn_init` | Uses `k_nn=0` which gives random init (proxy for zero-init). Architecturally identical. |
| GAS-adapted baseline | ✗ | N/A | Requires known action hierarchy which is unavailable for unexpected expansion setting. |

---

## Evaluation Requirements (from Research Contract)

| Requirement | Status | Location | Notes |
|---|---|---|---|
| Custom Gymnasium environment (DAVE-game) | ✗ | `train.py::build_env` | Raises `NotImplementedError` |
| Custom environment (continuous control) | ✗ | `train.py::build_env` | Raises `NotImplementedError` |
| Custom environment (goal-conditioned) | ✗ | `train.py::build_env` | Raises `NotImplementedError` |
| CartPole demoware | ✓ | `env_wrapper.py::ExpandingActionWrapper` | 1-2 expansions with no-op mapping |
| Metric: cumulative reward | ~ | `train.py::train` | Logged but no comparative analysis script |
| Metric: steps to recover | ✗ | N/A | Requires full training with expansion |
| Metric: Q-value degradation | ~ | `agent.py::get_q_for_actions` | Computed but no threshold assertion in production |
| Metric: % steps saved vs retrain | ✗ | N/A | Requires retrain-from-scratch baseline |
| Ablation: embedding init strategy | ✓ | `run_ablations.py` | k-NN → random init (k_nn: 3 → 0) |
| Ablation: freeze schedule | ✓ | `run_ablations.py` | freeze_encoder_steps: 10000 → 0; also 100, 100000 |
| Ablation: Q-function form | ✓ | `run_ablations.py` | factorized → dueling_factorized |
| Statistical rigor (10 seeds) | ✗ | N/A | All tests use single seed. No seed-sweep harness. |

---

## Synthetic Benchmarks (Layer 2)

| Benchmark | Status | Location | Line |
|---|---|---|---|
| Discouraged return sanity | ✓ | `test_model.py::TestSyntheticBenchmarks::test_discounted_return_finite` | Proxy metric |
| TD target with new action | ✓ | `test_model.py::TestSyntheticBenchmarks::test_td_target_with_new_action` | Validates TD target computation |
| k-NN init produces valid embeddings | ✓ | `test_model.py::TestSyntheticBenchmarks::test_embedding_knn_init_produces_valid_embeddings` | Cosine + Euclidean |
| Embedding similarity metrics | ✓ | `test_model.py::TestSyntheticBenchmarks::test_embedding_similarity_metrics` | Both metrics tested |
| Parameter counts positive | ✓ | `test_model.py::TestSyntheticBenchmarks::test_parameter_counts_positive` | All components |
| CNN encoder forward | ✓ | `test_model.py::TestSyntheticBenchmarks::test_cnn_forward_with_encoder` | 84×84 image input |
| Random rollout no crash | ✓ | `test_model.py::TestSyntheticBenchmarks::test_random_rollout_no_crash_with_wrapper` | 50 steps |
| Expansion at different steps | ✓ | `test_model.py::TestSyntheticBenchmarks::test_expansion_at_different_steps` | 3 thresholds |
| Q-degradation metric | ✓ | `test_model.py::TestRLBenchmarks::test_q_degradation_metric` | Before/after comparison |
| Monotonicity property | ✓ | `test_model.py::TestRLBenchmarks::test_monotonicity_property` | Max Q over old vs. full set |
| Rollout sanity with wrapper | ✓ | `test_model.py::TestRLBenchmarks::test_rollout_sanity_with_env_wrapper` | 200-step CartPole |
| Multiple expansions | ✓ | `test_model.py::TestRLBenchmarks::test_multiple_expansions_consistency` | 2→3→4→5 |
| Dueling centering correctness | ✓ | `test_model.py::TestRLBenchmarks::test_dueling_centering_correctness` | Advantage mean |
| Freezing prevents encoder drift | ✓ | `test_model.py::TestRLBenchmarks::test_freezing_prevents_encoder_drift` | Phase 1 invariance |
| Aux loss MSE non-negative | ✓ | `test_model.py::TestRLBenchmarks::test_aux_loss_reduces_mse` | aux_loss ≥ 0 |

---

## Ablations (Layer 3)

| Ablation | Baseline | Ablated | Status | Proxy metric | Reference |
|---|---|---|---|---|---|
| k-NN embedding init → random init | k_nn=3 | k_nn=0 | ✓ | Q-stats, TD loss | `run_ablations.py` |
| Optimistic bonus → no bonus | bonus=2.0 | bonus=0.0 | ✓ | Q-stats, expansion success | `run_ablations.py` |
| Freeze encoder → no freeze | steps=10000 | steps=0 | ✓ | Q-stats, TD loss | `run_ablations.py` |
| Factorized Q → dueling | factorized | dueling_factorized | ✓ | Q-stats, expansion success | `run_ablations.py` |
| Aux loss → no aux loss | True | False | ✓ | Q-stats, aux loss value | `run_ablations.py` |
| Old embedding freeze → no freeze | True | False | ✓ | Q-stats | `run_ablations.py` |
| Small embedding | d_emb=64 | d_emb=8 | ✓ | Q-stats | `run_ablations.py` |
| Large embedding | d_emb=64 | d_emb=256 | ✓ | Q-stats | `run_ablations.py` |
| Cosine → Euclidean | cosine | euclidean | ✓ | Q-stats | `run_ablations.py` |
| High bonus | bonus=2.0 | bonus=10.0 | ✓ | Q-stats | `run_ablations.py` |
| Short freeze | steps=10000 | steps=100 | ✓ | Q-stats | `run_ablations.py` |
| Long freeze | steps=10000 | steps=100000 | ✓ | Q-stats | `run_ablations.py` |
| Small Q-head | hidden=128 | hidden=16 | ✓ | Q-stats | `run_ablations.py` |
| Deep Q-head | n_layers=2 | n_layers=4 | ✓ | Q-stats | `run_ablations.py` |

**NOTE:** All ablation evaluations use a proxy metric (Q-value statistics on
random states, not actual environment interaction rewards). Full training-based
ablations require the custom benchmark environment.

---

## Profiling (Layer 4)

| Profile mode | Status | Location |
|---|---|---|
| Forward pass (inference) | ✓ | `profile_model.py::profile_forward` |
| Train step (forward + backward) | ✓ | `profile_model.py::profile_train` |
| Action expansion cost | ✓ | `profile_model.py::profile_expansion` |
| Component parameter counts | ✓ | `profile_model.py::display_component_params` |
| Gradient checkpointing comparison | ✓ | `profile_model.py::profile_gradient_checkpointing` |

---

## Unit Tests (Layer 1)

| Class | Test count | Coverage |
|---|---|---|
| `TestShapes` | 12 | All component shapes, batch consistency, expansion shapes |
| `TestGradients` | 5 | Gradient flow, NaN detection, freeze mask, Phase 1 isolation |
| `TestRLProperties` | 10 | Action range, exploration, target sync, bonus init, cooldown |
| `TestNumerics` | 9 | NaN/Inf detection, bf16, extreme values, TD loss finiteness |
| `TestRLBenchmarks` | 8 | Q-degradation, monotonicity, rollout sanity, multiple expansion |
| `TestSyntheticBenchmarks` | 9 | Discounted return, TD target, k-NN init, parameters, CNN, rollout |
| **Total** | **53** | |

---

## Gaps Summary

| Gap | Severity | What's needed |
|---|---|---|
| Custom Gymnasium environment | BLOCKING | DAVE-game analogue, continuous control, goal-conditioned scenarios |
| Retrain-from-scratch baseline | BLOCKING | Full training comparison on actual environment |
| Fixed-action-oracle baseline | BLOCKING | Full action set from episode 0 |
| Training-based evaluation | HIGH | Replace proxy metrics with actual reward curves |
| Multiple seeds | HIGH | 10-seed harness with mean ± std reporting |
| Embedding quality analysis | MEDIUM | t-SNE, cosine similarity matrix, k-NN vs random init |
| Hyperparameter sensitivity | MEDIUM | Systematic sweep of k_nn, d_embedding, freeze_steps |
| Q-degradation through training | HIGH | Log per-state-action Q values across full training run |
| GAS-adapted baseline | MEDIUM | Post-hoc hierarchy construction if possible |
| Paper draft / results section | LOW | Conference-paper format results |
