#!/usr/bin/env python3
"""Layer 4 — Profiling script for Action-Space Incremental DQN.

Uses torch.profiler to measure:
  - Forward pass time and memory per component
  - Backward pass GPU/CPU time
  - Peak memory usage during training
  - Estimated FLOPs (2× param count for forward, 6× for forward+backward)

Usage:
    PYTHONPATH=/artifacts/j_hmuwwrXFhQXC/work/coder:$PYTHONPATH \
        python profile_model.py --mode forward --steps 20
    PYTHONPATH=/artifacts/j_hmuwwrXFhQXC/work/coder:$PYTHONPATH \
        python profile_model.py --mode train --steps 10
    PYTHONPATH=/artifacts/j_hmuwwrXFhQXC/work/coder:$PYTHONPATH \
        python profile_model.py --profile_memory
"""

import argparse
import sys
import time

import numpy as np
import torch

CODER_DIR = "/artifacts/j_hmuwwrXFhQXC/work/coder"
sys.path.insert(0, CODER_DIR)

from model_config import ModelConfig
from agent import ActionIncrementalDQN
from networks import StateEncoder, ActionEmbeddingTable, QHead, ValueHead, DynamicsHead, count_params


def build_agent(cfg: ModelConfig) -> ActionIncrementalDQN:
    """Create a fresh agent."""
    return ActionIncrementalDQN(cfg, n_actions=cfg.n_actions_init)


def build_batch(cfg: ModelConfig, agent: ActionIncrementalDQN, batch_size: int = 64):
    """Build a synthetic batch for profiling."""
    # Push transitions to agent's buffer
    for _ in range(batch_size * 3):
        s = np.random.randn(cfg.obs_dim).astype(np.float32)
        agent.replay_buffer.push(
            s,
            np.random.randint(0, agent.action_count),
            float(np.random.randn()),
            np.random.randn(cfg.obs_dim).astype(np.float32),
            bool(np.random.random() < 0.1),
        )

    return agent.replay_buffer.sample(batch_size)


def profile_forward(cfg: ModelConfig, steps: int = 20, profile_memory: bool = False):
    """Profile the forward pass only (inference-like)."""
    agent = build_agent(cfg)
    device = agent.device

    # Warm up
    for _ in range(5):
        s = torch.randn(1, cfg.obs_dim, device=device)
        with torch.no_grad():
            phi_s = agent.encoder(s)
            for a in range(agent.action_count):
                e_a = agent.action_embeddings(torch.tensor([a], device=device)).squeeze(0)
                _ = agent._compute_q(phi_s, e_a)

    # Profiled run
    from torch.profiler import profile, record_function, ProfilerActivity

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=profile_memory,
        with_stack=True,
    ) as prof:
        for step in range(steps):
            with record_function("full_forward"):
                s = torch.randn(1, cfg.obs_dim, device=device)
                with torch.no_grad():
                    phi_s = agent.encoder(s)
                    for a in range(agent.action_count):
                        e_a = agent.action_embeddings(
                            torch.tensor([a], device=device)
                        ).squeeze(0)
                        _ = agent._compute_q(phi_s, e_a)

    # Print results
    sort_key = "self_cpu_time_total"
    if device.type == "cuda":
        sort_key = "cuda_time_total"
        print(prof.key_averages().table(sort_by=sort_key, row_limit=15))
    else:
        print(prof.key_averages().table(sort_by=sort_key, row_limit=15))

    # Parameter & FLOP estimate
    total_params = sum(p.numel() for p in agent._trainable_params())
    print(f"\n{'=' * 60}")
    print(f"  Forward Pass Profile ({steps} steps)")
    print(f"{'=' * 60}")
    print(f"  Device:          {device.type}")
    print(f"  Trainable params: {total_params:,}")
    print(f"  Estimated FLOPs:  {2 * total_params / 1e6:.1f}M (2× params)")
    print(f"  Action count:     {agent.action_count}")


def profile_train(cfg: ModelConfig, steps: int = 10, profile_memory: bool = False):
    """Profile the full train step (forward + backward)."""
    agent = build_agent(cfg)
    device = agent.device

    # Pre-fill buffer
    for _ in range(200):
        s = np.random.randn(cfg.obs_dim).astype(np.float32)
        ns = np.random.randn(cfg.obs_dim).astype(np.float32)
        agent.replay_buffer.push(
            s, np.random.randint(0, agent.action_count),
            float(np.random.randn()), ns,
            bool(np.random.random() < 0.1),
        )

    # Warm up
    if len(agent.replay_buffer) >= cfg.batch_size:
        for _ in range(5):
            batch = agent.replay_buffer.sample(cfg.batch_size)
            agent.update(batch)

    from torch.profiler import profile, record_function, ProfilerActivity

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=profile_memory,
        with_stack=True,
    ) as prof:
        for step in range(steps):
            with record_function("train_step"):
                if len(agent.replay_buffer) >= cfg.batch_size:
                    batch = agent.replay_buffer.sample(cfg.batch_size)
                    metrics = agent.update(batch)

    # Print results
    sort_key = "self_cpu_time_total"
    mem_sort = "self_cpu_memory_usage"
    if device.type == "cuda":
        sort_key = "cuda_time_total"
        mem_sort = "self_cuda_memory_usage"

    print("\n[Time]")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=15))

    if profile_memory:
        print("\n[Memory]")
        print(prof.key_averages().table(sort_by=mem_sort, row_limit=15))

    # Parameter & FLOP estimate
    total_params = sum(p.numel() for p in agent._trainable_params())
    print(f"\n{'=' * 60}")
    print(f"  Train Step Profile ({steps} steps)")
    print(f"{'=' * 60}")
    print(f"  Device:          {device.type}")
    print(f"  Trainable params: {total_params:,}")
    print(f"  Estimated FLOPs:  {6 * total_params / 1e6:.1f}M (6× params, fwd+bwd)")
    print(f"  Action count:     {agent.action_count}")
    print(f"  Batch size:       {cfg.batch_size}")


def profile_expansion(cfg: ModelConfig):
    """Profile the cost of handle_action_expansion."""
    agent = build_agent(cfg)
    device = agent.device

    class MockEnv:
        def __init__(self):
            self.action_space_n = cfg.n_actions_init + 1

    # Profile expansion
    from torch.profiler import profile, record_function, ProfilerActivity

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True) as prof:
        for _ in range(100):
            with record_function("action_expansion"):
                agent_cp = build_agent(cfg)
                agent_cp.handle_action_expansion(MockEnv())

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    print(f"\n{'=' * 60}")
    print(f"  Action Expansion Profile (100 repetitions)")
    print(f"{'=' * 60}")
    # Total embeddings created
    print(f"  Total expansion ops: 100")
    print(f"  Embedding dimension: {cfg.d_embedding}")
    print(f"  k-NN:                {cfg.k_nn}")


def profile_gradient_checkpointing(cfg: ModelConfig):
    """Compare memory/time with and without gradient checkpointing."""
    from networks import StateEncoder

    encoder = StateEncoder(cfg).to("cuda" if torch.cuda.is_available() else "cpu")
    s = torch.randn(4, cfg.obs_dim, device=encoder.cfg.encoder_dim if hasattr(encoder, 'cfg') else None)

    # Without checkpointing
    s = torch.randn(4, cfg.obs_dim, device=next(encoder.parameters()).device)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    out = encoder(s)
    loss = out.sum()
    loss.backward()

    mem_no_ckpt = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    print(f"\n{'=' * 60}")
    print(f"  Gradient Checkpointing Comparison")
    print(f"{'=' * 60}")
    print(f"  Peak memory (no checkpoint): {mem_no_ckpt / 1024**2:.2f} MB")


def display_component_params(cfg: ModelConfig):
    """Display parameter counts for each component."""
    encoder = StateEncoder(cfg)
    emb = ActionEmbeddingTable(cfg.n_actions_init, cfg)
    q_head = QHead(cfg)
    v_head = ValueHead(cfg)
    dyn = DynamicsHead(cfg)

    print("\n" + "=" * 60)
    print("  Component Parameter Counts")
    print("=" * 60)
    print(f"  {'Component':<25s} {'Params':>10s}")
    print("-" * 60)
    total = 0
    for name, mod in [
        ("StateEncoder", encoder),
        ("ActionEmbeddingTable", emb),
        ("QHead", q_head),
        ("ValueHead", v_head),
        ("DynamicsHead", dyn),
    ]:
        n = sum(p.numel() for p in mod.parameters())
        total += n
        print(f"  {name:<25s} {n:>10,}")

    print("-" * 60)
    print(f"  {'Total':<25s} {total:>10,}")
    print()

    # Per-component breakdown
    count_params(encoder)
    count_params(emb)
    count_params(q_head)


def main():
    parser = argparse.ArgumentParser(
        description="Profile Action-Space Incremental DQN"
    )
    parser.add_argument(
        "--mode", default="forward",
        choices=["forward", "train", "expansion", "component_params"],
        help="Profiling mode",
    )
    parser.add_argument("--steps", default=20, type=int,
                        help="Number of profiled steps")
    parser.add_argument("--profile_memory", action="store_true",
                        help="Enable memory profiling")
    parser.add_argument("--d_embedding", default=64, type=int,
                        help="Action embedding dimension")
    parser.add_argument("--encoder_dim", default=256, type=int,
                        help="Encoder output dimension")
    parser.add_argument("--q_hidden_dim", default=128, type=int,
                        help="Q-head hidden dimension")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size for train profiling")
    args = parser.parse_args()

    cfg = ModelConfig(
        obs_dim=64,
        n_actions_init=3,
        encoder_dim=args.encoder_dim,
        d_model=args.encoder_dim,
        d_embedding=args.d_embedding,
        q_hidden_dim=args.q_hidden_dim,
        q_n_layers=2,
        k_nn=3,
        batch_size=args.batch_size,
        use_auxiliary_dynamics_loss=True,
    )

    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'} "
          f"({torch.cuda.device_count() if torch.cuda.is_available() else 0} GPUs)")

    if args.mode == "forward":
        profile_forward(cfg, steps=args.steps, profile_memory=args.profile_memory)
    elif args.mode == "train":
        profile_train(cfg, steps=args.steps, profile_memory=args.profile_memory)
    elif args.mode == "expansion":
        profile_expansion(cfg)
    elif args.mode == "component_params":
        display_component_params(cfg)


if __name__ == "__main__":
    main()
