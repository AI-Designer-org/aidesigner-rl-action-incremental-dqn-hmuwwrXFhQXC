#!/usr/bin/env python3
"""Layer 3 — Ablation Runner for Action-Space Incremental RL.

Runs single-field ModelConfig changes from the architect's suggested ablated
list and compares against the baseline configuration.

Usage:
    cd /artifacts/j_hmuwwrXFhQXC/work/validator
    PYTHONPATH=/artifacts/j_hmuwwrXFhQXC/work/coder:$PYTHONPATH python run_ablations.py

This runner uses a minimal synthetic environment (no Gymnasium dependency)
to measure relative differences between ablation conditions. For production
ablations, replace `eval_fn` with a full training run on the custom benchmark.
"""

import sys
import time
import json
from copy import deepcopy
from dataclasses import replace, fields as dataclass_fields
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CODER_DIR = "/artifacts/j_hmuwwrXFhQXC/work/coder"
sys.path.insert(0, CODER_DIR)

from model_config import ModelConfig
from agent import ActionIncrementalDQN
from networks import count_params


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic evaluation protocol
# ═══════════════════════════════════════════════════════════════════════════

def quick_eval(agent: ActionIncrementalDQN, n_eval_steps: int = 100) -> dict:
    """Run a fast synthetic evaluation of the agent.

    Uses random states and measures:
      - Q-value statistics (mean, std, min, max)
      - Q-value spread across actions (potential for discrimination)
      - TD loss convergence over synthetic batch
      - Parameter count and gradient health

    This is a PROXY metric — not a substitute for full training on
    the custom benchmark. Use this for fast ablation comparisons.
    """
    cfg = agent.cfg
    metrics = {}

    # ── Q-value statistics on random states ────────────────────────────
    q_means = []
    q_stds = []
    for _ in range(n_eval_steps):
        s = np.random.randn(cfg.obs_dim).astype(np.float32)
        q = agent.get_q_for_actions(s)
        q_means.append(q.mean())
        q_stds.append(q.std())

    metrics["q_mean"] = float(np.mean(q_means))
    metrics["q_std"] = float(np.mean(q_stds))
    metrics["q_range"] = float(np.mean(
        [q.max() - q.min() for q in [agent.get_q_for_actions(
            np.random.randn(cfg.obs_dim).astype(np.float32)
        ) for _ in range(20)]]
    ))

    # ── TD loss on synthetic batch ─────────────────────────────────────
    # Fill buffer and measure loss
    for _ in range(cfg.batch_size * 2):
        agent.replay_buffer.push(
            np.random.randn(cfg.obs_dim).astype(np.float32),
            np.random.randint(0, agent.action_count),
            float(np.random.randn()),
            np.random.randn(cfg.obs_dim).astype(np.float32),
            bool(np.random.random() < 0.1),
        )

    td_losses = []
    aux_losses = []
    for _ in range(10):
        if len(agent.replay_buffer) >= cfg.batch_size:
            batch = agent.replay_buffer.sample(cfg.batch_size)
            m = agent.update(batch)
            td_losses.append(m["td_loss"])
            aux_losses.append(m.get("aux_loss", 0.0))

    metrics["td_loss_mean"] = float(np.mean(td_losses)) if td_losses else -1.0
    metrics["td_loss_std"] = float(np.std(td_losses)) if td_losses else 0.0
    metrics["aux_loss_mean"] = float(np.mean(aux_losses)) if aux_losses else 0.0

    # ── Expansion handling ─────────────────────────────────────────────
    class MockEnv:
        def __init__(self):
            self.action_space_n = agent.action_count + 1

    try:
        agent.handle_action_expansion(MockEnv())
        q_post = agent.get_q_for_actions(
            np.random.randn(cfg.obs_dim).astype(np.float32)
        )
        metrics["expansion_success"] = 1.0
        metrics["q_new_action"] = float(q_post[-1])
    except Exception as e:
        metrics["expansion_success"] = 0.0
        metrics["expansion_error"] = str(e)

    # ── Parameter count ────────────────────────────────────────────────
    metrics["n_params"] = sum(
        p.numel() for p in agent.encoder.parameters()
    ) + sum(
        p.numel() for p in agent.action_embeddings.parameters()
    ) + sum(
        p.numel() for p in agent.q_head.parameters()
    )

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Ablation definitions
# ═══════════════════════════════════════════════════════════════════════════

def _baseline_cfg() -> ModelConfig:
    return ModelConfig(
        obs_dim=64,
        n_actions_init=3,
        encoder_dim=256,
        d_model=256,
        d_embedding=64,
        q_hidden_dim=128,
        q_n_layers=2,
        k_nn=3,
        embedding_similarity="cosine",
        q_form="factorized",
        optimistic_init_bonus=2.0,
        freeze_encoder_steps=10_000,
        freeze_old_embeddings=True,
        freeze_q_head_steps=5_000,
        use_auxiliary_dynamics_loss=True,
        target_tau=1.0,
        buffer_capacity=100_000,
        batch_size=64,
    )


def get_ablations() -> Dict[str, ModelConfig]:
    """Return dict of ablation_name -> modified ModelConfig.

    Each ablation changes exactly ONE field relative to baseline.
    """
    base = _baseline_cfg()

    ablations = {
        "baseline": base,
    }

    # Tier 1 — Core hypotheses (turn these off first if method fails)
    ablations["no_knn_init"] = replace(base, k_nn=0)
    ablations["no_optimistic_bonus"] = replace(base, optimistic_init_bonus=0.0)

    # Tier 2 — Architectural decisions
    ablations["no_freeze_encoder"] = replace(base, freeze_encoder_steps=0)
    ablations["dueling_q"] = replace(base, q_form="dueling_factorized")

    # Tier 3 — Refinements
    ablations["no_aux_loss"] = replace(base, use_auxiliary_dynamics_loss=False)
    ablations["no_old_embedding_freeze"] = replace(base, freeze_old_embeddings=False)
    ablations["small_embedding"] = replace(base, d_embedding=8)
    ablations["large_embedding"] = replace(base, d_embedding=256)
    ablations["euclidean_sim"] = replace(base, embedding_similarity="euclidean")

    # Sensitivity ablations
    ablations["high_bonus"] = replace(base, optimistic_init_bonus=10.0)
    ablations["short_freeze"] = replace(base, freeze_encoder_steps=100)
    ablations["long_freeze"] = replace(base, freeze_encoder_steps=100_000)
    ablations["small_q"] = replace(base, q_hidden_dim=16)
    ablations["deep_q"] = replace(base, q_n_layers=4)

    return ablations


def print_ablation_diff(name: str, baseline_cfg: ModelConfig, ablated_cfg: ModelConfig):
    """Print which fields differ between baseline and ablation."""
    diffs = []
    for f in dataclass_fields(ModelConfig):
        bv = getattr(baseline_cfg, f.name)
        av = getattr(ablated_cfg, f.name)
        if bv != av:
            diffs.append(f"  {f.name}: {bv} -> {av}")

    print(f"\n{'=' * 70}")
    print(f"  Ablation: {name}")
    print(f"{'=' * 70}")
    for d in diffs:
        print(d)
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Main ablation runner
# ═══════════════════════════════════════════════════════════════════════════

def run_ablations():
    """Run all ablations and collect metrics."""
    ablations = get_ablations()
    baseline_cfg = ablations["baseline"]

    results: Dict[str, dict] = {}
    baseline_metrics = None

    print("=" * 70)
    print("  Action-Space Incremental RL — Ablation Runner")
    print("=" * 70)

    for name, cfg in ablations.items():
        print_ablation_diff(name, baseline_cfg, cfg)

        # Create fresh agent
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        agent = ActionIncrementalDQN(cfg, n_actions=cfg.n_actions_init)

        # Run evaluation
        try:
            metrics = quick_eval(agent, n_eval_steps=50)
            results[name] = metrics

            baseline_str = ""
            if name == "baseline":
                baseline_metrics = metrics
                baseline_str = "  [BASELINE]"
            elif baseline_metrics is not None:
                # Compute relative difference key metrics
                for key in ["q_mean", "q_std", "td_loss_mean", "expansion_success"]:
                    if key in metrics and key in baseline_metrics and baseline_metrics[key] != 0:
                        rel_diff = (metrics[key] - baseline_metrics[key]) / abs(baseline_metrics[key])
                        if abs(rel_diff) > 0.1:  # Only flag significant differences
                            baseline_str += f"  {key}: {rel_diff:+.1%} vs baseline"

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"error": str(e)}

        # Print summary
        print(f"  q_mean={metrics.get('q_mean', 'N/A'):.4f}  "
              f"q_std={metrics.get('q_std', 'N/A'):.4f}  "
              f"td_loss={metrics.get('td_loss_mean', 'N/A'):.4f}  "
              f"expansion={metrics.get('expansion_success', 'N/A')}  "
              f"params={metrics.get('n_params', 'N/A')}")
        if baseline_str:
            print(baseline_str)

    # ── Summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Ablation Summary")
    print("=" * 70)
    print(f"{'Name':<25s} {'q_mean':>8s} {'q_std':>8s} {'td_loss':>10s} {'expand':>8s}")
    print("-" * 70)
    for name in results:
        m = results[name]
        if "error" not in m:
            print(f"{name:<25s} {m.get('q_mean', 0):>8.4f} "
                  f"{m.get('q_std', 0):>8.4f} "
                  f"{m.get('td_loss_mean', 0):>10.4f} "
                  f"{m.get('expansion_success', 0):>8.1f}")

    # Save results
    output_path = "/artifacts/j_hmuwwrXFhQXC/work/validator/ablation_results.json"
    with open(output_path, "w") as f:
        # Convert any non-serializable values
        serializable = {}
        for name, m in results.items():
            serializable[name] = {
                k: (str(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in m.items()
            }
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_ablations()
