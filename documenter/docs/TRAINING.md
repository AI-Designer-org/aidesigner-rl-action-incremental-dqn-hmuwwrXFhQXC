# Training & Reproduction

## Environment

- Python: 3.10+
- PyTorch: >= 2.0
- Gymnasium: >= 0.28
- NumPy: >= 1.24
- CUDA: optional (CPU-only development is supported)
- GPU: any NVIDIA GPU with >= 1 GB VRAM (for training at scale)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch numpy gymnasium
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

No `requirements.txt` is provided yet — the dependencies are minimal (torch, numpy, gymnasium). Full reproducibility will require a pinned environment specification.

## Default hyperparameters

All hyperparameters are defined in the `ModelConfig` dataclass (`coder/model_config.py`).

| Field | Default | Rationale |
|---|---|---|
| `obs_type` | `"vector"` | Default to vector observations for faster iteration |
| `obs_dim` | 64 | Placeholder state vector dimension |
| `n_actions_init` | 3 | DAVE-game: {up, left, right} in level 1 |
| `max_n_actions` | 20 | Safety cap for pre-allocated tracking tensors |
| `encoder_dim` | 256 | Output dimension of state encoder φ(s) |
| `d_model` | 256 | Internal hidden dimension for encoder MLP |
| `d_embedding` | 64 | Dimensionality of action embedding e_a |
| `k_nn` | 3 | Nearest neighbours for k-NN embedding initialisation |
| `embedding_similarity` | `"cosine"` | Cosine similarity for k-NN neighbour selection |
| `q_hidden_dim` | 128 | Hidden dimension of Q-head MLP |
| `q_n_layers` | 2 | Depth of Q-head MLP |
| `q_form` | `"factorized"` | Factorized Q(s,a) = f_Q([φ(s); e_a]) |
| `gamma` | 0.99 | Standard TD discount factor |
| `learning_rate` | 3e-4 | Adam default for DQN |
| `buffer_capacity` | 100,000 | Replay buffer size |
| `batch_size` | 64 | Training batch size |
| `target_update_freq` | 1,000 | Hard-copy interval for target network |
| `optimistic_init_bonus` | 2.0 | Additive bonus for new actions |
| `optimistic_bonus_decay` | 0.99 | Per-visit decay multiplier |
| `optimistic_bonus_min` | 0.01 | Floor after decay |
| `freeze_encoder_steps` | 10,000 | Phase 1 duration |
| `freeze_old_embeddings` | True | Freeze old action embeddings during fine-tuning |
| `freeze_q_head_steps` | 5,000 | Phase 1→2 transition (subset of encoder freeze) |
| `expansion_cooldown` | 1,000 | Min env steps between expansions |
| `use_auxiliary_dynamics_loss` | True | InfoNCE-style dynamics auxiliary loss |
| `epsilon_init` | 1.0 | Initial ε-greedy exploration |
| `epsilon_min` | 0.05 | Minimum ε after decay |
| `epsilon_decay_steps` | 100,000 | Linear decay duration |

## Recommended training recipe

| Setting | Value | Notes |
|---|---|---|
| Optimizer | Adam | β1=0.9, β2=0.999 |
| Peak LR | 3e-4 | Fixed (no scheduler beyond Phase 3 reduction) |
| Batch size | 64 | Gradient accumulation not needed at this scale |
| Epsilon decay | 100,000 steps | Linear from 1.0 to 0.05 |
| Target update | Hard copy every 1,000 steps | Full state_dict copy |
| Grad clip | 10.0 | Global norm |
| Precision | float32 | bf16 forward pass tested numerically stable but training uses float32 |

### Training command

```bash
# CartPole demoware — single expansion at step 1000
PYTHONPATH=$PWD/coder:$PYTHONPATH python coder/train.py \
    --env cartpole \
    --n_episodes 2000 \
    --seed 42 \
    --q_form factorized \
    --d_embedding 64 \
    --k_nn 3 \
    --optimistic_bonus 2.0 \
    --freeze_steps 10000
```

### Dueling variant

```bash
PYTHONPATH=$PWD/coder:$PYTHONPATH python coder/train.py \
    --env cartpole \
    --q_form dueling_factorized \
    --n_episodes 2000
```

### Disable auxiliary dynamics loss

```bash
PYTHONPATH=$PWD/coder:$PYTHONPATH python coder/train.py \
    --env cartpole \
    --no_aux_loss
```

## Expected behavior

> TODO: unverified — no reference training run has been completed on the custom benchmark environment. The CartPole demoware provides a functional test of the training loop but is not a meaningful evaluation of the action-expansion protocol because the new action maps to a no-op. See [BENCHMARKS.md](BENCHMARKS.md#research-quality-evaluation) for the current evaluation gaps.

### What to observe during training (once an appropriate environment is available)

1. **Before expansion (|A| = initial set):** Standard DQN learning curve. Episode rewards should increase over time as ε decays.
2. **At expansion (env detects new action):** `[expansion] Action space growing from N to N+1 at env step X` is printed. ε is temporarily boosted to 0.5. The freeze state transitions from `none` to `phase1`.
3. **Phase 1 (new embedding only):** Only the new action embedding receives gradients. Old-action Q-values should remain stable.
4. **Phase 2 (unfreeze Q-head):** `[expansion] Phase 1 → Phase 2 at env step X` is printed after `freeze_q_head_steps` (default 5,000). Reward may dip temporarily as the Q-head adjusts.
5. **Phase 3 (unfreeze all):** `[expansion] Phase 2 → Phase 3 at env step X` printed after `freeze_encoder_steps` (default 10,000). Learning rate drops to 0.1×. Gradual recovery.

### Log output format

```
Ep     1 | Step       5 | Reward   42.0 | eps 1.000 | |A|=3 | freeze=none  | buf=    0 | time=  1s
Ep   100 | Step    1024 | Reward  150.2 | eps 0.490 | |A|=3 | freeze=none  | buf= 1024 | time=  8s
[expansion] Action space growing from 3 to 4 at env step 1000
Ep   200 | Step    2025 | Reward  180.5 | eps 0.460 | |A|=4 | freeze=phase1| buf= 2025 | time= 15s
[expansion] Phase 1 -> Phase 2 at env step 6000
...
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `NotImplementedError` for "dave" environment | Custom benchmark not yet built | Use `--env cartpole` for demoware; implement the custom environment |
| `RuntimeError: CUDA out of memory` | Too many parameters for GPU | Reduce `d_model`, `encoder_dim`, or `d_embedding`; use CPU (`device='cpu'`) |
| Loss NaN in first steps | bf16 numerical instability in Q-head | Verify float32 is being used (`cfg.dtype = "float32"`) |
| Expansion not triggered | Cooldown guard blocking | Check `expansion_cooldown` (default 1,000 steps); verify env wrapper has `max_expansions > 0` |
| Phase transitions not happening | `freeze_q_head_steps` or `freeze_encoder_steps` too high | Reduce them; the defaults (5,000 / 10,000) are for full-scale training |
| `ValueError: Unknown q_form` | Typo in config | Use `"factorized"` or `"dueling_factorized"` |
| Old-action Q-values change in Phase 1 | Gradient hook on embedding table not applied | Verify `freeze_old_embeddings=True`; check `_freeze_mask` values |
| All Q-values near-identical | Embedding space collapse | Enable auxiliary dynamics loss; increase `d_embedding`; reduce `k_nn` |
| Agent never selects new action | Optimistic bonus too low or decayed too fast | Increase `optimistic_init_bonus` or reduce `optimistic_bonus_decay` |
| Q-values diverge after expansion | No gradient clipping or learning rate too high | Verify `clip_grad_norm_(..., 10.0)` is active in the update step |
| `ExpandingActionWrapper` action space not growing | Step counter or expansion threshold misconfiguration | Verify `expand_at_step` is set correctly; the counter is cumulative across episodes |
