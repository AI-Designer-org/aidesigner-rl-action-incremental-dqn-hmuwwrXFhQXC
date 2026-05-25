# Changelog

## [0.1.0] — 2026-05-25

### Added
- Initial implementation of Action-Incremental DQN.
  - Factorized Q(s,a) = f_Q(φ(s), e_a) architecture with expandable action embedding table.
  - Two Q-function variants: `factorized` and `dueling_factorized`.
  - State encoder supporting both vector (MLP) and image (CNN) observations.
  - k-NN centroid interpolation for new action embedding initialisation (cosine and Euclidean similarity).

- Action expansion protocol.
  - `handle_action_expansion()` dynamically grows the embedding table mid-training.
  - `expansion_cooldown` guard prevents rapid successive expansions.
  - Optimistic exploration bonus (additive, geometrically decaying) for new actions.
  - ε-greedy epsilon schedule with post-expansion boost (floor at 0.5 during cooldown).

- Progressive unfreezing (Phase 1 → 2 → 3).
  - Phase 1: only new action embedding trainable (encoder, Q-head, old embeddings frozen).
  - Phase 2: Q-head and value head unfrozen (encoder remains frozen).
  - Phase 3: all unfrozen with 0.1× learning rate reduction.
  - Gradient hook freeze mask on embedding table for torch.compile compatibility.

- Auxiliary dynamics prediction head (optional, 0.01× loss weight).
  - Predicts φ(s') − φ(s) from (φ(s), e_a) to shape the embedding space.

- Off-policy DQN training with target network (hard copy at fixed intervals).
  - Replay buffer with uniform sampling.
  - Gradient clipping (global norm 10.0).
  - Batch Q-value computation for all actions via batched embedding expansion.

### Test suite (22 smoke tests + 53 pytest tests)
- Unit test suite covering shape correctness, gradient flow, RL-specific properties, and numerical stability.
- Domain-specific RL benchmarks: Q-degradation metric, monotonicity property, expansion robustness, freeze schedule progression, dueling centering correctness.
- CartPole demoware with `ExpandingActionWrapper` for end-to-end rollouts.
- bf16 forward pass stability verified for encoder and Q-head.

### Ablation infrastructure
- 14 ablation conditions in `run_ablations.py`: k-NN init, optimistic bonus, freeze schedule, Q-form, auxiliary loss, old embedding freeze, embedding dimension, similarity metric, bonus magnitude, freeze duration, Q-head capacity.
- JSON output with comparative metrics.

### Profiling
- `profile_model.py` with four modes: forward pass, train step, action expansion cost, component parameter counts.
- Gradient checkpointing comparison script.

### Documentation
- README with quick-start and repository layout.
- ARCHITECTURE.md with ASCII diagrams, equations, tensor shape evolution, design-decision table, inductive bias justification, and research-to-architecture traceability.
- TRAINING.md with hyperparameter table, training recipe, and troubleshooting guide.
- BENCHMARKS.md with synthetic benchmark results, ablation analysis, profiling data, and research-quality evaluation with blocking gaps.
- API.md documenting every public class, method, and function.
- Research evaluation artifacts: `scorecard.json`, `claim_grounding.md`, `experiment_coverage.md`, `rubric.md`.

### Research contributions (hypotheses, not yet validated)
- Novel combination of factorized Q + k-NN embedding initialisation for unexpected action-space expansion in deep RL.
- Progressive unfreezing protocol tailored to action-space boundaries (not task boundaries).
- Dueling factorized variant with known-action-only centering to mitigate softmax denominator shift.
- Optimistic exploration bonus combined with embedding interpolation for directed new-action exploration.
