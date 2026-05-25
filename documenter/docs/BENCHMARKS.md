# Benchmarks

All numbers are reproducible with the commands shown. Numbers marked `TODO` have not been measured — do not cite them.

> **Important**: All evaluations in this document use **synthetic proxy metrics** (Q-value statistics on random states, not actual environment interaction rewards). Full training-based evaluation on the custom benchmark environment is a prerequisite for publication. See [Research-quality evaluation](#research-quality-evaluation) for the gap summary.

## RL synthetic benchmarks

These benchmarks verify the correctness and numerical stability of the action-expansion protocol under controlled synthetic conditions. They are **not** task-performance benchmarks.

| Test | Result | Command | Notes |
|---|---|---|---|
| Q-degradation metric | old-action Q-values preserved (atol=1e-6) with frozen params | `pytest test_model.py::TestRLBenchmarks::test_q_degradation_metric` | Phase 1 freezing preserves old-action Q exactly |
| Q-degradation after training | old-action Q norm change < 5× | `pytest test_model.py::TestRLBenchmarks::test_q_degradation_after_training` | Proxy measurement; generous margin |
| Monotonicity property | max Q(expanded set) >= max Q(original set) | `pytest test_model.py::TestRLBenchmarks::test_monotonicity_property` | Verified with random init (no training) |
| Rollout sanity (CartPole) | finite TD loss, no crashes | `pytest test_model.py::TestRLBenchmarks::test_rollout_sanity_with_env_wrapper` | 200-step CartPole with simulated expansion |
| Multiple expansions (2→5) | forward pass, Q-values, TD update all valid | `pytest test_model.py::TestRLBenchmarks::test_multiple_expansions_consistency` | 3 expansions in sequence |
| Dueling centering | mean known-action advantage ~0 | `pytest test_model.py::TestRLBenchmarks::test_dueling_centering_correctness` | Centering over known actions works |
| Freezing prevents encoder drift | encoder weights invariant in Phase 1 | `pytest test_model.py::TestRLBenchmarks::test_freezing_prevents_encoder_drift` | Verified after 10 training steps |
| Batch Q consistency | batch vs. per-action Q match at 1e-5 | `pytest test_model.py::TestShapes::test_batch_q_consistency` | Batched computation validated |
| TD loss finite (pre-expansion) | always finite | `pytest test_model.py::TestNumerics::test_td_loss_finite` | 53/53 tests pass |
| TD loss finite (post-expansion) | always finite | `pytest test_model.py::TestNumerics::test_td_loss_after_expansion_finite` | After Phase 1 freeze |
| bf16 encoder forward | no NaN/Inf | `pytest test_model.py::TestNumerics::test_bf16_encoder_forward` | bf16 forward supports mixed precision |
| Freeze schedule progression | Phase 1→2→3 correct | `pytest test_model.py::TestRLBenchmarks::test_freeze_schedule_progression` | Phase transitions verified |

Reproduce all synthetic benchmarks: `PYTHONPATH=$PWD/coder:$PYTHONPATH pytest validator/test_model.py -v --tb=short`

## Test suite summary

| Class | Tests | Coverage area |
|---|---|---|
| `TestShapes` | 12 | Component shapes, batch consistency, expansion shapes |
| `TestGradients` | 5 | Gradient flow, NaN detection, freeze mask, Phase 1 isolation |
| `TestRLProperties` | 10 | Action range, exploration, target sync, bonus init, cooldown |
| `TestNumerics` | 9 | NaN/Inf detection, bf16, extreme values, TD loss finiteness |
| `TestRLBenchmarks` | 8 | Q-degradation, monotonicity, rollout sanity, multiple expansions |
| `TestSyntheticBenchmarks` | 9 | Discounted return, TD target, k-NN init, parameters, CNN, rollout |
| **Total** | **53** | All passing |

## Ablation study

All ablations use a single-field change to `ModelConfig`, evaluated on Q-value statistics from random states (proxy metric — not actual environment rewards). The baseline configuration is:

- `q_form= factorized, d_embedding=64, k_nn=3, q_hidden_dim=128, q_n_layers=2`
- `optimistic_init_bonus=2.0, freeze_encoder_steps=10,000`
- `use_auxiliary_dynamics_loss=True, freeze_old_embeddings=True`
- Total parameters: 272,513

Reproduce: `PYTHONPATH=$PWD/coder:$PYTHONPATH python validator/run_ablations.py`

| Ablation | Config delta | q_mean | q_std | td_loss_mean | q_new_action | params | Δ vs. baseline |
|---|---|---|---|---|---|---|---|
| **baseline** | — | 0.1162 | 0.0022 | 0.8455 | 0.0318 | 272,513 | — |
| no_knn_init | k_nn=3→0 | 0.1162 | 0.0022 | 0.8247 | NaN | 272,513 | td_loss -2.5% |
| no_optimistic_bonus | bonus=2.0→0.0 | 0.1162 | 0.0022 | 0.8629 | 0.1224 | 272,513 | td_loss +2.1% |
| no_freeze_encoder | steps=10K→0 | 0.1162 | 0.0022 | 0.8242 | 0.0863 | 272,513 | td_loss -2.5% |
| dueling_q | q_form=dueling | -0.1781 | 0.0022 | 0.6941 | -0.7109 | 272,513 | td_loss -17.9% |
| no_aux_loss | aux=False | 0.1162 | 0.0022 | 0.7977 | -0.1085 | 272,513 | td_loss -5.7% |
| no_old_embedding_freeze | freeze=False | 0.1162 | 0.0022 | 0.8072 | 0.0371 | 272,513 | td_loss -4.5% |
| small_embedding | d_emb=64→8 | 0.0458 | 0.0005 | 0.7602 | -0.2706 | 265,121 | td_loss -10.1% |
| large_embedding | d_emb=64→256 | -0.1600 | 0.0027 | 0.9024 | -0.2012 | 297,857 | td_loss +6.7% |
| euclidean_sim | cosine→euclidean | 0.1162 | 0.0022 | 0.8379 | 0.0463 | 272,513 | td_loss -0.9% |
| high_bonus | bonus=2.0→10.0 | 0.1162 | 0.0022 | 0.7797 | 0.0716 | 272,513 | td_loss -7.8% |
| short_freeze | steps=10K→100 | 0.1162 | 0.0022 | 0.7938 | -0.0525 | 272,513 | td_loss -6.1% |
| long_freeze | steps=10K→100K | 0.1162 | 0.0022 | 0.7993 | -0.0456 | 272,513 | td_loss -5.5% |
| small_q | hidden=128→16 | 0.1446 | 0.0014 | 0.8312 | 0.0020 | 220,209 | td_loss -1.7% |
| deep_q | n_layers=2→4 | -0.0854 | 0.0004 | 0.8943 | -0.0675 | 305,537 | td_loss +5.8% |

> **Interpretation caveat**: These are proxy metrics on random states with no environment interaction. A "lower td_loss" in an ablation does not mean better performance — it may mean a degenerate Q-function. The `q_new_action` column shows the Q-value for the newly added action immediately after expansion; values near zero may indicate poor initialisation, but values far from zero with random initialisation may be meaningless. **Full training-based evaluation is required before drawing conclusions.**

### Observed patterns (provisional)

1. **Dueling variant shows lower mean Q and lower TD loss** — the centering mechanism shifts Q-values, which may change the loss landscape. Whether this translates to better task performance is unknown without training.
2. **Small embedding (d_emb=8) collapses Q-range** — q_std drops to 0.0005 from 0.0022, suggesting the embedding space lacks capacity to differentiate actions. This is a warning sign for the k-NN init hypothesis.
3. **No auxiliary loss flips q_new_action negative** — without the dynamics head, the new action's Q-value is negative (vs. positive for baseline). This tentatively suggests the aux loss helps produce better initialised embeddings.
4. **k-NN vs. random init shows identical Q-stats** — the proxy metric cannot distinguish between these conditions, which is expected because Q-value statistics on random states do not measure initialisation quality.

## Profiling

All profiling was run on CPU (no GPU available in CI). FLOPs are estimated as 2× parameter count for forward and 6× for forward+backward.

| Phase | Estimated time (CPU, ms/step) | Peak mem (MB) | Estimated FLOPs |
|---|---|---|---|
| Forward pass (inference, 1 state, 3 actions) | < 1 ms | < 50 MB | ~545M (2× params) |
| Train step (forward + backward, batch=64) | < 5 ms | < 200 MB | ~1.64G (6× params) |
| Action expansion (k-NN init + table append) | < 0.1 ms | negligible | O(|A|·d_emb) |

Reproduce:
```bash
# Forward profile
PYTHONPATH=$PWD/coder:$PYTHONPATH python validator/profile_model.py --mode forward --steps 20

# Train profile
PYTHONPATH=$PWD/coder:$PYTHONPATH python validator/profile_model.py --mode train --steps 10

# Expansion profile
PYTHONPATH=$PWD/coder:$PYTHONPATH python validator/profile_model.py --mode expansion

# Component parameter counts
PYTHONPATH=$PWD/coder:$PYTHONPATH python validator/profile_model.py --mode component_params
```

### Component parameter counts

| Component | Params |
|---|---|
| StateEncoder | 199,168 |
| ActionEmbeddingTable (3 actions) | 192 |
| QHead | 73,089 |
| ValueHead (dueling) | 32,897 |
| DynamicsHead | 49,408 |
| **Total (all components)** | **354,754** |
| **Trainable (baseline config)** | **272,513** |

The difference between total and trainable arises because some components are optional (DuelingFactorizedQ includes both QHead and ValueHead; the baseline factorized variant uses QHead only but ValueHead is counted in the total).

## Research-quality evaluation

| Dimension | Score (1-5) | Evidence | Gaps |
|---|---|---|---|
| **Novelty** | 3 / 5 | Factorized Q + k-NN init + progressive unfreeze differs from GAS (requires hierarchy), morphin (tabular), Headless-AD (bandit/MDP only) | No empirical validation on actual RL task; novelty may be an engineering combination of existing techniques |
| **Experimental comprehensiveness** | 3 / 5 | 53 tests, 14 ablations, profiling script, both q_form variants | **BLOCKING**: No custom environment; no retrain-from-scratch baseline; no fixed-action-oracle; single seed only; proxy metrics only |
| **Theoretical foundation** | 4 / 5 | Monotonicity property correctly cited; factorized Q justified; k-NN centroid as minimax optimal; off-policy choice justified | Auxiliary dynamics loss justification (semantic embeddings) is untested; no formal proof of monotonicity preservation under function approximation |
| **Result analysis** | 2 / 5 | Ablation runner produces comparative metrics; Q-degradation computed before/after expansion | No training-based results; no confidence intervals; no sensitivity analysis; no training curves |
| **Implementation reproducibility** | 4 / 5 | All tests pass; ablation runner produces JSON; profiling runnable; ModelConfig documents all hyperparameters | No pinned dependencies; no Dockerfile; custom environment raises NotImplementedError |
| **Writing readiness** | 3 / 5 | Architecture well-documented with diagrams; inductive bias table; traceability table; risk flags | No paper draft; no results section; no limitations section beyond implementation risks |

### Blocking gaps (from `research_eval/scorecard.json`)

1. **`benchmark_not_beaten`**: No custom Gymnasium environment (DAVE-game analogue, continuous control, goal-conditioned). All three raise `NotImplementedError`.
2. **`benchmark_not_beaten`**: Retrain-from-scratch baseline not implemented. Cannot claim sub-linear fine-tuning.
3. **`benchmark_not_beaten`**: Fixed-action-oracle baseline not implemented. Cannot compute % total training steps saved.
4. **`coverage_gap`**: Only proxy evaluation metrics (Q-stats on random states), not actual environment rewards.
5. **`coverage_gap`**: No multiple-seed (>=10) experiments. All tests use single seed.
6. **`coverage_gap`**: No embedding quality evaluation (t-SNE, cosine similarity matrix, k-NN init quality vs. random).
7. **`novelty_unverified`**: Core hypothesis (k-NN init provides better-than-random Q-values for new actions) untested on actual RL task.
8. **`claim_not_grounded`**: "Sub-linear fine-tuning cost" and "no catastrophic forgetting" claims cannot be validated without full training comparison.

### Required next experiments

| Priority | Experiment | Addresses |
|---|---|---|
| **P0** | Implement custom Gymnasium environment (DAVE-game analogue) | benchmark_not_beaten, coverage_gap |
| **P0** | Full training with retrain-from-scratch baseline | benchmark_not_beaten, claim_not_grounded |
| **P1** | k-NN init vs. random init on actual RL task (zero-shot Q(a_new)) | novelty_unverified |
| **P1** | 10-seed sweep per condition, report mean ± std | coverage_gap |
| **P1** | Embedding space analysis (t-SNE, cosine similarity matrix) | coverage_gap |
| **P2** | Hyperparameter sensitivity sweep (k_nn, d_embedding, freeze_steps) | coverage_gap |
| **P2** | Q-degradation through full training (Phase 1→2→3) | claim_not_grounded |
| **P3** | Continuous control scenario (MuJoCo) | coverage_gap |
| **P3** | Auxiliary dynamics loss ablation with embedding visualisation | coverage_gap |

> TODO: unverified — the five `TODO: unverified` claims from the research contract remain ungrounded. See `validator/research_eval/claim_grounding.md` for the detailed status of each claim and its falsification conditions.
