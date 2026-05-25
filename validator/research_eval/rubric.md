# Research Quality Rubric — Action Space Incremental Reinforcement Learning

## Domain: RL (Action Space Incremental Reinforcement Learning)

This rubric evaluates the quality and credibility of the generated architecture,
implementation, and validation infrastructure as a research artifact. It
complements the automated test suite and ablation runner — it does not replace them.

---

## Scoring Scale

| Score | Meaning |
|---|---|
| 0 | Not addressed or no artifact exists |
| 1 | Mentioned but unsupported |
| 2 | Partially supported with major gaps |
| 3 | Plausible and minimally supported |
| 4 | Strong, with clear evidence and reproducible checks |
| 5 | Publication-ready for this scaffold's scope |

---

## Dimension 1: Novelty

**Domain-specific questions for RL:**

- Does the approach demonstrably fill the identified gap (unexpected action-space
  expansion in deep RL, no predefined hierarchy)?
- Is there evidence that the method differs meaningfully from Growing Action Spaces
  (Farquhar et al.), morphin (de la Rosa et al.), Headless-AD (Sinii et al.), and
  Progressive Networks (Rusu et al.)?
- Does the k-NN embedding initialization + optimistic bonus + progressive unfreeze
  protocol constitute a non-obvious combination of existing ideas?
- Is the novelty claim falsifiable? (i.e., can it be shown to fail if the core
  hypothesis is wrong?)

**Scoring guidance:**
- 0-1: No novelty claim or the method is trivially different from baselines
- 2-3: Reasonable claim of novelty but no comparative evaluation against baselines
- 4-5: Clear novelty with baseline comparisons and ablation evidence

---

## Dimension 2: Experimental Comprehensiveness

**Domain-specific questions for RL:**

| Question | Weight |
|---|---|
| Does the benchmark test action-space expansion with multiple seeds (≥10)? | Critical |
| Does it compare against retrain-from-scratch baseline? | Critical |
| Does it compare against fixed-action-space oracle? | Critical |
| Does it compare against zero-initialization expansion? | Critical |
| Does it measure Q-value degradation on old actions after expansion? | High |
| Does it measure steps to recover pre-expansion performance? | High |
| Does it measure % total training steps saved vs. retrain? | High |
| Does it test the monotonicity property V*_i(s) ≤ V*_j(s)? | Medium |
| Does it include ablations for: embedding init, freeze schedule, Q-form? | Critical |
| Does it include ablations for: optimistic bonus, aux loss, target update? | Medium |
| Does it test multiple environments (DAVE-analogue, continuous control)? | Medium |
| Does it explicitly test for catastrophic forgetting? | High |
| Does the expansion protocol handle multiple sequential expansions? | Medium |

---

## Dimension 3: Theoretical Foundation

**Domain-specific questions for RL:**

- Is the factorized Q(s,a) = f_Q(φ(s), e_a) architecture justified beyond
  "it seems like it should work"?
- Is the k-NN centroid initialization justified (minimax optimal under no
  semantic descriptor)?
- Is the monotonicity property correctly cited and applied?
- Is the choice of off-policy (DQN) vs. on-policy correctly justified for
  the expanding-action-space setting?
- Is the progressive unfreezing schedule grounded in continual learning
  theory (EWC, elastic weight consolidation analog)?
- Are the softmax denominator shift concerns addressed (dueling variant with
  known-action-only centering)?

---

## Dimension 4: Result Analysis

**Domain-specific questions for RL:**

- Are the Q-value degradation metrics reported before/after expansion?
- Is there a clear interpretation of what constitutes acceptable degradation?
- Are the ablation results interpreted in terms of the core hypothesis?
- Are results reported with error bars / multiple seeds?
- Is there a sensitivity analysis for key hyperparameters (k_nn, embedding dim,
  freeze_steps, bonus magnitude)?

---

## Dimension 5: Implementation Reproducibility

- Can all tests be run with a single command (`pytest test_model.py -v`)?
- Can ablations be reproduced with a single command?
- Is the random seed propagation correct and complete?
- Are all hyperparameters documented in ModelConfig?
- Does the smoke test cover both q_form variants?
- Is the profiling script runnable and documented?
- Are all environment dependencies specified?

---

## Dimension 6: Writing Readiness

- Is the architecture clearly documented (README or equivalent)?
- Are the inductive biases of each design choice justified?
- Are the novelty claims precisely stated and bounded?
- Are the limitations and failure modes documented?
- Are the results presented in a format suitable for a conference paper
  (tables, ablations, comparisons)?

---

## Domain-Specific Research Questions (RL)

1. **Causal masking**: For RL with Q-learning, does the target computation
   correctly use the max over the current action set (which may differ from
   the action set at data-collection time)? ✓ — the batch Q-value computation
   always uses `self.action_count`.

2. **Policy validity**: Do actions always fall within the current valid range?
   ✓ — `select_action` samples from `[0, action_count)` and greedy action
   selection also respects this bound.

3. **Critic sanity**: Is the Q-function well-behaved (finite, no NaN, no
   saturation to constant values)? ✓ — tested via `test_td_loss_finite`
   and `test_no_nan_gradients`.

4. **Rollout stability**: Does the agent + environment interaction produce
   finite rewards without crashing through multiple expansions?
   ✓ — tested via `test_rollout_sanity_with_env_wrapper`.

5. **Exploration**: Does the agent explore new actions? ✓ — tested via
   `test_exploration_entropy` (high epsilon gives variety) and
   `test_optimistic_bonus` (bonus initialized for new actions).

6. **Catastrophic forgetting**: Do old-action Q-values degrade? ✓ — tested via
   `test_q_degradation_metric` and `test_freezing_prevents_encoder_drift`.

7. **Baseline comparison readiness**: Are there scripts/kits to compare against
   retrain-from-scratch, fixed-action-oracle, zero-init expansion? Only proxy
   evaluations exist (Tier 1-2 ablations). Full custom benchmark is still TODO.

8. **Multiple expansions**: Does the system handle >1 expansions robustly?
   ✓ — tested via `test_multiple_expansions_consistency`.

9. **Embedding structure**: Does k-NN init produce meaningful embeddings?
   Proxy test exists (`test_embedding_knn_init_produces_valid_embeddings`)
   but no semantic quality verification.

10. **Hyperparameter sensitivity**: Are key hyperparameters ablated?
    Partially — ablation runner covers 14 conditions but the proxy metric
    is not a substitute for training-based evaluation.
