> **Project layout** — this bundle contains five stage directories from the
> AI-Designer pipeline:
> `research/` (literature survey), `architect/` (blueprint + `ModelConfig`),
> `coder/` (PyTorch implementation), `validator/` (tests + benchmarks), and
> `documenter/` (this README plus `docs/` and `CHANGELOG.md`).
> An optional `paper/` directory holds the NeurIPS-format writeup when the
> paper-generation step was triggered.
>
> The original research request that produced this bundle is preserved
> verbatim in [`prompt.md`](prompt.md) — if any URLs in the prompt were
> fetched server-side for additional context, their cleaned contents are
> appended there too.

---

# Action-Incremental DQN

A deep Q-network agent that handles unexpectedly introduced actions without retraining from scratch, using a factorized Q-architecture with learnable action embeddings and progressive unfreezing.

Standard deep RL assumes a fixed action space throughout training. In video games, robotics, and skill acquisition, new actions can become available mid-training (e.g., a "shoot" action unlocked upon reaching game level 3). Retraining from scratch wastes prior experience. This project implements a DQN agent whose action-value function is factorised as Q(s,a) = f_Q(φ(s), e_a), where action embeddings are stored in a dynamically expandable table. When a new action appears, its embedding is initialised via k-NN interpolation over existing embeddings, an optimistic exploration bonus encourages its use, and a three-phase progressive unfreeze schedule prevents catastrophic forgetting of old actions.

> TODO: unverified — the core hypothesis (k-NN initialisation provides better-than-random zero-shot Q-values and sub-linear fine-tuning) has not yet been validated on a full RL task. All evaluations to date use synthetic proxy metrics. See [BENCHMARKS.md](documenter/docs/BENCHMARKS.md#research-quality-evaluation).

## Highlights

- **Factorised Q(s,a) = f_Q(φ(s), e_a)** — decouples action identity from the Q-function, enabling zero-shot Q-value estimates for unseen actions via embedding similarity. See [ARCHITECTURE.md](documenter/docs/ARCHITECTURE.md#3-the-core-component).
- **k-NN embedding initialisation for new actions** — a new action's embedding is initialised as the mean of its k nearest neighbours in the existing embedding space, providing a structural prior that reduces cold-start error. See [ARCHITECTURE.md](documenter/docs/ARCHITECTURE.md#action-embedding-table).
- **Progressive three-phase unfreezing** — encoder → Q-head → all, with reduced learning rate in Phase 3, prevents catastrophic forgetting during action integration. See [ARCHITECTURE.md](documenter/docs/ARCHITECTURE.md#action-expansion-flow).
- **Optimistic exploration bonus** — newly added actions receive a decaying additive bonus to encourage systematic exploration beyond ε-greedy randomness. See [ARCHITECTURE.md](documenter/docs/ARCHITECTURE.md#exploration).

## Quick start

```bash
# Run the full smoke test suite (22 tests, no GPU required)
PYTHONPATH=$PWD/coder:$PYTHONPATH python coder/smoke_test.py

# Run the comprehensive pytest suite (53 tests)
PYTHONPATH=$PWD/coder:$PYTHONPATH pytest validator/test_model.py -v --tb=short

# Run ablations (14 conditions, proxy evaluation)
PYTHONPATH=$PWD/coder:$PYTHONPATH python validator/run_ablations.py

# Train on CartPole demoware with simulated expansion
PYTHONPATH=$PWD/coder:$PYTHONPATH python coder/train.py --env cartpole --n_episodes 2000
```

## Repository layout

```
coder/
  model_config.py        ModelConfig dataclass — all hyperparameters
  networks.py            Network components: StateEncoder, ActionEmbeddingTable,
                          QHead, ValueHead, DynamicsHead, DuelingFactorizedQ
  agent.py               ActionIncrementalDQN agent + ReplayBuffer
  env_wrapper.py         ExpandingActionWrapper for simulated expansion
  train.py               Training loop with expansion detection + CLI
  smoke_test.py          22-test smoke suite
validator/
  test_model.py          53-test pytest suite (Layer 1-2)
  run_ablations.py       14-condition ablation runner
  profile_model.py       torch.profiler-based profiling script
  ablation_results.json  Recorded ablation metrics
  research_eval/         Research quality evaluation (scorecard, coverage,
                          claim grounding, rubric)
architect/
  model_config.py, components.py, agent.py, train.py   Reference design
  ACTION_INCREMENTAL_RL_ARCHITECTURE.md                 Full architecture design
research/
  ACTION_SPACE_INCREMENTAL_RL.md    Research lifecycle contract
```

## Documentation

- [docs/ARCHITECTURE.md](documenter/docs/ARCHITECTURE.md) — design rationale, inductive biases, equations, tensor shapes
- [docs/TRAINING.md](documenter/docs/TRAINING.md) — environment setup, hyperparameters, recipe, troubleshooting
- [docs/BENCHMARKS.md](documenter/docs/BENCHMARKS.md) — synthetic benchmarks, ablation results, profiling, research-evaluation gaps
- [docs/API.md](documenter/docs/API.md) — module-level API reference

## Citation

```bibtex
@misc{action-incremental-dqn,
  title  = {Action-Incremental DQN: Deep RL with Unexpected Action-Space Expansion},
  author = {<TODO>},
  year   = {2026},
  note   = {Generated via ml-designer pipeline}
}
```
