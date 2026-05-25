#!/usr/bin/env python3
"""Smoke test for Action-Space Incremental RL.

Tests:
  1. ModelConfig instantiation
  2. All network components (StateEncoder, ActionEmbeddingTable, QHead,
     ValueHead, DuelingFactorizedQ, DynamicsHead) forward pass shapes
  3. ActionEmbeddingTable expansion (add_embedding)
  4. ActionEmbeddingTable gradient freezing mask
  5. Full ActionIncrementalDQN agent:
       - select_action with epsilon-greedy
       - update (TD learning step)
       - handle_action_expansion
       - freeze schedule (Phase 1 -> 2 -> 3)
       - get_q_for_actions
  6. ReplayBuffer push/sample
  7. Parameter count
  8. Both q_form variants (factorized and dueling_factorized)

Run with:
    python smoke_test.py
"""
import sys
import os

import numpy as np
import torch

from model_config import ModelConfig
from networks import (
    StateEncoder,
    ActionEmbeddingTable,
    QHead,
    ValueHead,
    DuelingFactorizedQ,
    DynamicsHead,
    count_params,
    BaseOperator,
)
from agent import ActionIncrementalDQN, ReplayBuffer, Transition
from env_wrapper import ExpandingActionWrapper


def test_model_config():
    """Test that ModelConfig can be created with defaults and overrides."""
    print("[test] ModelConfig...", end=" ")

    cfg = ModelConfig()
    assert cfg.obs_type == "vector"
    assert cfg.n_actions_init == 3
    assert cfg.q_form == "factorized"
    assert cfg.d_embedding == 64
    print("OK")

    # Test image config
    cfg_img = ModelConfig(obs_type="image", obs_height=84, obs_width=84)
    assert cfg_img.obs_type == "image"
    print("[test] ModelConfig (image variant)... OK")


def test_state_encoder_vector():
    """Test StateEncoder with vector observations."""
    print("[test] StateEncoder (vector)...", end=" ")

    cfg = ModelConfig(obs_type="vector", obs_dim=64, encoder_dim=256, d_model=256)
    encoder = StateEncoder(cfg)

    B = 4
    s = torch.randn(B, cfg.obs_dim)                                          # (B, obs_dim)
    phi_s = encoder(s)                                                       # (B, encoder_dim)

    assert phi_s.shape == (B, cfg.encoder_dim), f"Bad shape: {phi_s.shape}"
    assert not torch.isnan(phi_s).any(), "NaN in encoder output"
    print(f"OK ({phi_s.shape})")


def test_state_encoder_image():
    """Test StateEncoder with image observations."""
    print("[test] StateEncoder (image)...", end=" ")

    cfg = ModelConfig(obs_type="image", obs_channels=3, obs_height=84, obs_width=84,
                      encoder_dim=256)
    encoder = StateEncoder(cfg)

    B = 2
    s = torch.randn(B, cfg.obs_channels, cfg.obs_height, cfg.obs_width)      # (B, C, H, W)
    phi_s = encoder(s)                                                       # (B, encoder_dim)

    assert phi_s.shape == (B, cfg.encoder_dim), f"Bad shape: {phi_s.shape}"
    assert not torch.isnan(phi_s).any(), "NaN in encoder output"
    print(f"OK ({phi_s.shape})")


def test_action_embedding_table():
    """Test ActionEmbeddingTable forward, expansion, and freeze mask."""
    print("[test] ActionEmbeddingTable...", end=" ")

    cfg = ModelConfig(d_embedding=64)
    n_actions = 3

    # Create table
    emb = ActionEmbeddingTable(n_actions, cfg)
    assert emb.weight.shape == (n_actions, cfg.d_embedding)

    # Forward: single index
    indices = torch.tensor([0, 1, 2])
    e = emb(indices)                                                         # (3, d_emb)
    assert e.shape == (3, cfg.d_embedding)

    # Forward: batched
    batch_indices = torch.tensor([0, 0, 1, 2, 1])                            # (5,)
    e_batch = emb(batch_indices)                                             # (5, d_emb)
    assert e_batch.shape == (5, cfg.d_embedding)
    print("forward OK,", end=" ")

    # Expansion: add a new embedding
    emb.add_embedding(n_known=3, k=2, similarity="cosine")
    assert emb.weight.shape == (4, cfg.d_embedding), f"Bad shape after expansion: {emb.weight.shape}"
    assert emb.n_actions == 4
    print("expand OK,", end=" ")

    # Check that the new embedding is non-zero and finite
    new_e = emb(torch.tensor([3]))                                            # (1, d_emb)
    assert not torch.isnan(new_e).any(), "NaN in new embedding"
    print("init OK,", end=" ")

    # Freeze mask: freeze old rows
    emb.set_rows_frozen(torch.tensor([0, 1]), frozen=True)
    assert emb._freeze_mask[0].item() == 0.0
    assert emb._freeze_mask[1].item() == 0.0
    assert emb._freeze_mask[2].item() == 1.0  # unfrozen
    print("freeze OK,", end=" ")

    # Unfreeze all
    emb.unfreeze_all()
    assert emb._freeze_mask.sum().item() == 4.0
    print("unfreeze OK")

    # Euclidean similarity
    emb2 = ActionEmbeddingTable(3, cfg)
    emb2.add_embedding(n_known=3, k=2, similarity="euclidean")
    assert emb2.weight.shape == (4, cfg.d_embedding)
    print("[test] ActionEmbeddingTable (euclidean)... OK")


def test_q_head():
    """Test QHead forward pass."""
    print("[test] QHead...", end=" ")

    cfg = ModelConfig(encoder_dim=256, d_embedding=64,
                      q_hidden_dim=128, q_n_layers=2)
    q_head = QHead(cfg)

    B = 4
    phi_s = torch.randn(B, cfg.encoder_dim)                                   # (B, d_enc)
    e_a = torch.randn(cfg.d_embedding)                                        # (d_emb,)

    # Single action (broadcast)
    q = q_head(phi_s, e_a)                                                    # (B, 1)
    assert q.shape == (B, 1), f"Bad shape: {q.shape}"

    # Batched actions
    e_batch = torch.randn(B, cfg.d_embedding)                                 # (B, d_emb)
    q2 = q_head(phi_s, e_batch)                                               # (B, 1)
    assert q2.shape == (B, 1)
    assert not torch.isnan(q).any(), "NaN in Q-head output"
    print(f"OK ({q.shape})")


def test_value_head():
    """Test ValueHead forward pass."""
    print("[test] ValueHead...", end=" ")

    cfg = ModelConfig(encoder_dim=256, q_hidden_dim=128)
    v_head = ValueHead(cfg)

    B = 4
    phi_s = torch.randn(B, cfg.encoder_dim)                                   # (B, d_enc)
    V = v_head(phi_s)                                                         # (B, 1)

    assert V.shape == (B, 1), f"Bad shape: {V.shape}"
    print(f"OK ({V.shape})")


def test_dueling_factorized_q():
    """Test DuelingFactorizedQ forward pass with and without centering."""
    print("[test] DuelingFactorizedQ...", end=" ")

    cfg = ModelConfig(encoder_dim=256, d_embedding=64,
                      q_hidden_dim=128, q_n_layers=2)
    dq = DuelingFactorizedQ(cfg)

    B, n_act = 4, 3
    phi_s = torch.randn(B, cfg.encoder_dim)                                   # (B, d_enc)
    e_a = torch.randn(cfg.d_embedding)                                        # (d_emb,)
    all_embeds = torch.randn(n_act, cfg.d_embedding)                          # (n_act, d_emb)

    # Without centering
    q1 = dq(phi_s, e_a)
    assert q1.shape == (B, 1), f"Bad shape without centering: {q1.shape}"

    # With centering
    q2 = dq(phi_s, e_a, all_embeddings=all_embeds, n_known=2)
    assert q2.shape == (B, 1), f"Bad shape with centering: {q2.shape}"

    assert not torch.isnan(q2).any(), "NaN in dueling Q output"
    print(f"OK ({q1.shape}, {q2.shape})")


def test_dynamics_head():
    """Test DynamicsHead forward pass."""
    print("[test] DynamicsHead...", end=" ")

    cfg = ModelConfig(encoder_dim=256, d_embedding=64, q_hidden_dim=128)
    dyn = DynamicsHead(cfg)

    B = 4
    phi_s = torch.randn(B, cfg.encoder_dim)                                   # (B, d_enc)
    e_a = torch.randn(cfg.d_embedding)                                        # (d_emb,)

    delta = dyn(phi_s, e_a)                                                   # (B, d_enc)
    assert delta.shape == (B, cfg.encoder_dim), f"Bad shape: {delta.shape}"
    assert not torch.isnan(delta).any(), "NaN in dynamics head output"
    print(f"OK ({delta.shape})")


def test_base_operator():
    """Test that all components inherit from BaseOperator."""
    print("[test] BaseOperator inheritance...", end=" ")

    cfg = ModelConfig()
    assert isinstance(StateEncoder(cfg), BaseOperator)
    assert isinstance(ActionEmbeddingTable(3, cfg), BaseOperator)
    assert isinstance(QHead(cfg), BaseOperator)
    assert isinstance(ValueHead(cfg), BaseOperator)
    assert isinstance(DuelingFactorizedQ(cfg), BaseOperator)
    assert isinstance(DynamicsHead(cfg), BaseOperator)
    print("OK")


def test_replay_buffer():
    """Test ReplayBuffer push/sample."""
    print("[test] ReplayBuffer...", end=" ")

    buf = ReplayBuffer(capacity=1000)

    # Push transitions
    for i in range(100):
        buf.push(
            state=np.random.randn(64).astype(np.float32),
            action=i % 3,
            reward=float(i),
            next_state=np.random.randn(64).astype(np.float32),
            done=bool(i % 5 == 0),
        )

    assert len(buf) == 100

    # Sample a batch
    batch = buf.sample(batch_size=32)
    assert batch.state.shape == (32, 64), f"Bad state shape: {batch.state.shape}"
    assert batch.action.shape == (32,), f"Bad action shape: {batch.action.shape}"
    assert batch.reward.shape == (32,), f"Bad reward shape: {batch.reward.shape}"
    assert batch.next_state.shape == (32, 64)
    assert batch.done.shape == (32,)

    # FIFO eviction
    for i in range(1000):
        buf.push(
            state=np.random.randn(64).astype(np.float32),
            action=0, reward=0.0,
            next_state=np.random.randn(64).astype(np.float32),
            done=False,
        )
    assert len(buf) == 1000  # at capacity
    print("OK")


def _make_agent(cfg: ModelConfig):
    """Create a minimal agent for smoke testing."""
    return ActionIncrementalDQN(cfg, n_actions=cfg.n_actions_init)


def test_agent_initialization():
    """Test ActionIncrementalDQN initialization."""
    print("[test] Agent init (factorized)...", end=" ")

    cfg = ModelConfig(obs_dim=64, n_actions_init=3, d_embedding=16, encoder_dim=64,
                      d_model=64, q_hidden_dim=32)
    agent = _make_agent(cfg)

    assert agent.action_count == 3
    assert agent.freeze_state == "none"
    assert len(agent.new_action_indices) == 0
    assert len(agent.action_bonuses) == 0
    print("OK")

    print("[test] Agent init (dueling_factorized)...", end=" ")
    cfg2 = ModelConfig(obs_dim=64, n_actions_init=3, q_form="dueling_factorized",
                       d_embedding=16, encoder_dim=64, d_model=64, q_hidden_dim=32)
    agent2 = _make_agent(cfg2)
    assert agent2.value_head is not None
    print("OK")


def test_agent_select_action():
    """Test select_action with epsilon-greedy."""
    print("[test] Agent select_action...", end=" ")

    cfg = ModelConfig(obs_dim=64, n_actions_init=3, d_embedding=16, encoder_dim=64,
                      d_model=64, q_hidden_dim=32)
    agent = _make_agent(cfg)

    # Greedy (epsilon=0)
    s = np.random.randn(64).astype(np.float32)
    a = agent.select_action(s, epsilon=0.0)
    assert isinstance(a, int)
    assert 0 <= a < agent.action_count

    # Random (epsilon=1.0)
    actions = [agent.select_action(s, epsilon=1.0) for _ in range(50)]
    assert all(0 <= a < agent.action_count for a in actions)
    # With 3 actions and 50 random draws, we should see some variety
    assert len(set(actions)) > 1, "All random actions were the same!"
    print(f"OK (greedy={a}, random_samples={len(set(actions))}/3)")


def test_agent_update():
    """Test a single TD update step."""
    print("[test] Agent TD update...", end=" ")

    cfg = ModelConfig(obs_dim=64, n_actions_init=3, d_embedding=16, encoder_dim=64,
                      d_model=64, q_hidden_dim=32, buffer_capacity=1000, batch_size=8)
    agent = _make_agent(cfg)

    # Fill replay buffer
    for _ in range(100):
        s = np.random.randn(64).astype(np.float32)
        ns = np.random.randn(64).astype(np.float32)
        agent.replay_buffer.push(s, int(np.random.randint(0, 3)), 1.0,
                                 ns, False)

    # Run update
    batch = agent.replay_buffer.sample(cfg.batch_size)
    metrics = agent.update(batch)

    assert "td_loss" in metrics
    assert "aux_loss" in metrics
    assert metrics["td_loss"] >= 0
    assert not np.isnan(metrics["td_loss"]), "NaN in TD loss"
    print(f"OK (td_loss={metrics['td_loss']:.4f}, aux_loss={metrics['aux_loss']:.4f})")


def test_agent_update_dueling():
    """Test TD update with dueling factorized Q."""
    print("[test] Agent TD update (dueling)...", end=" ")

    cfg = ModelConfig(obs_dim=64, n_actions_init=3, q_form="dueling_factorized",
                      d_embedding=16, encoder_dim=64, d_model=64,
                      q_hidden_dim=32, buffer_capacity=1000, batch_size=8)
    agent = ActionIncrementalDQN(cfg, n_actions=3)

    for _ in range(100):
        s = np.random.randn(64).astype(np.float32)
        ns = np.random.randn(64).astype(np.float32)
        agent.replay_buffer.push(s, int(np.random.randint(0, 3)), 1.0,
                                 ns, False)

    batch = agent.replay_buffer.sample(cfg.batch_size)
    metrics = agent.update(batch)

    assert "td_loss" in metrics
    assert not np.isnan(metrics["td_loss"]), "NaN in TD loss (dueling)"
    print(f"OK (td_loss={metrics['td_loss']:.4f})")


def test_agent_expansion():
    """Test handle_action_expansion end-to-end."""
    print("[test] Agent action expansion...", end=" ")

    cfg = ModelConfig(obs_dim=64, n_actions_init=3, d_embedding=16, encoder_dim=64,
                      d_model=64, q_hidden_dim=32, k_nn=2,
                      freeze_old_embeddings=True)
    agent = _make_agent(cfg)

    initial_count = agent.action_count
    assert initial_count == 3

    # Mock environment with expanded action space
    class MockEnv:
        def __init__(self):
            self.action_space_n = 4

    mock_env = MockEnv()
    agent.handle_action_expansion(mock_env)

    # Check expansion state
    assert agent.action_count == 4, f"Expected 4, got {agent.action_count}"
    assert 3 in agent.new_action_indices
    assert agent.freeze_state == "phase1"
    assert agent.action_bonuses.get(3, 0.0) == cfg.optimistic_init_bonus

    # Check embedding table grew
    assert agent.action_embeddings.weight.shape[0] == 4
    assert agent.target_action_embeddings.weight.shape[0] == 4

    # Check freeze mask: old rows (0,1,2) should be frozen, new row (3) trainable
    assert agent.action_embeddings._freeze_mask[0].item() == 0.0, "Old row 0 not frozen"
    assert agent.action_embeddings._freeze_mask[1].item() == 0.0, "Old row 1 not frozen"
    assert agent.action_embeddings._freeze_mask[2].item() == 0.0, "Old row 2 not frozen"
    assert agent.action_embeddings._freeze_mask[3].item() == 1.0, "New row should be trainable"

    # Check forward pass still works after expansion
    s = np.random.randn(64).astype(np.float32)
    a = agent.select_action(s, epsilon=0.0)
    assert 0 <= a < 4
    print(f"OK (|A|: {initial_count} -> {agent.action_count})")


def test_agent_expansion_dueling():
    """Test action expansion with dueling factorized Q."""
    print("[test] Agent action expansion (dueling)...", end=" ")

    cfg = ModelConfig(obs_dim=64, n_actions_init=3, q_form="dueling_factorized",
                      d_embedding=16, encoder_dim=64, d_model=64,
                      q_hidden_dim=32, k_nn=2)
    agent = ActionIncrementalDQN(cfg, n_actions=3)

    class MockEnv:
        def __init__(self):
            self.action_space_n = 4

    s = np.random.randn(64).astype(np.float32)

    # Q-values before expansion
    q_before = agent.get_q_for_actions(s)
    assert len(q_before) == 3

    agent.handle_action_expansion(MockEnv())
    assert agent.action_count == 4

    # After expansion: should compute Q for all 4 actions
    q_after = agent.get_q_for_actions(s)
    assert len(q_after) == 4, f"Expected 4 Q-values, got {len(q_after)}"

    # Forward pass
    a = agent.select_action(s, epsilon=0.0)
    assert 0 <= a < 4

    # TD update
    for _ in range(50):
        agent.replay_buffer.push(s, 0, 1.0, s, False)
    if len(agent.replay_buffer) >= cfg.batch_size:
        batch = agent.replay_buffer.sample(cfg.batch_size)
        metrics = agent.update(batch)
        assert not np.isnan(metrics["td_loss"])
    print("OK")


def test_freeze_schedule():
    """Test progressive freeze schedule (Phase 1 -> 2 -> 3)."""
    print("[test] Freeze schedule...", end=" ")

    cfg = ModelConfig(obs_dim=64, n_actions_init=3, d_embedding=16, encoder_dim=64,
                      d_model=64, q_hidden_dim=32,
                      freeze_q_head_steps=10, freeze_encoder_steps=20)
    agent = _make_agent(cfg)

    class MockEnv:
        def __init__(self):
            self.action_space_n = 4

    # Expand (enters Phase 1)
    agent.handle_action_expansion(MockEnv())
    assert agent.freeze_state == "phase1"

    # Check encoder is frozen
    encoder_trainable = sum(p.requires_grad for p in agent.encoder.parameters())
    assert encoder_trainable == 0, "Encoder should be frozen in Phase 1"

    # Check Q-head is frozen
    q_head_trainable = sum(p.requires_grad for p in agent.q_head.parameters())
    assert q_head_trainable == 0, "Q-head should be frozen in Phase 1"

    # Simulate enough steps for Phase 2
    agent.env_step = agent.expansion_step + cfg.freeze_q_head_steps + 1
    agent.maybe_update_freeze_schedule()
    assert agent.freeze_state == "phase2", f"Expected phase2, got {agent.freeze_state}"

    # Check Q-head is unfrozen
    q_head_trainable = sum(p.requires_grad for p in agent.q_head.parameters())
    assert q_head_trainable > 0, "Q-head should be unfrozen in Phase 2"

    # Check encoder is still frozen
    encoder_trainable = sum(p.requires_grad for p in agent.encoder.parameters())
    assert encoder_trainable == 0, "Encoder should still be frozen in Phase 2"

    # Simulate enough steps for Phase 3
    agent.env_step = agent.expansion_step + cfg.freeze_encoder_steps + 1
    agent.maybe_update_freeze_schedule()
    assert agent.freeze_state == "phase3", f"Expected phase3, got {agent.freeze_state}"

    # Check encoder is unfrozen
    encoder_trainable = sum(p.requires_grad for p in agent.encoder.parameters())
    assert encoder_trainable > 0, "Encoder should be unfrozen in Phase 3"

    # Check learning rate reduction
    for g in agent.optimizer.param_groups:
        assert g["lr"] == cfg.learning_rate * 0.1, \
            f"Expected LR {cfg.learning_rate * 0.1}, got {g['lr']}"

    print("OK (Phase 1 -> 2 -> 3)")


def test_get_q_for_actions():
    """Test get_q_for_actions utility."""
    print("[test] get_q_for_actions...", end=" ")

    cfg = ModelConfig(obs_dim=64, n_actions_init=3, d_embedding=16, encoder_dim=64,
                      d_model=64, q_hidden_dim=32)
    agent = _make_agent(cfg)

    s = np.random.randn(64).astype(np.float32)
    q_vals = agent.get_q_for_actions(s)

    assert isinstance(q_vals, np.ndarray)
    assert q_vals.shape == (3,), f"Bad shape: {q_vals.shape}"
    assert not np.isnan(q_vals).any(), "NaN in Q-values"
    print(f"OK ({q_vals.shape})")


def test_env_wrapper():
    """Test ExpandingActionWrapper."""
    print("[test] ExpandingActionWrapper...", end=" ")

    import gymnasium as gym
    env = gym.make("CartPole-v1")
    env = ExpandingActionWrapper(env, expand_at_step=50, max_expansions=1)
    env.reset()

    initial_n = env.action_space.n
    assert initial_n == 2, f"CartPole should have 2 actions, got {initial_n}"

    # Step to trigger expansion
    for _ in range(55):
        s, r, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            env.reset()

    assert env.has_expanded, "Expansion should have occurred"
    assert env.action_space.n == 3, f"Expected 3 actions after expansion, got {env.action_space.n}"
    print(f"OK ({initial_n} -> {env.action_space.n})")


def test_parameter_counts():
    """Test count_params helper and verify parameter counts."""
    print("[test] Parameter counts...")

    cfg = ModelConfig(obs_dim=64, encoder_dim=256, d_embedding=64,
                      d_model=256, q_hidden_dim=128, n_encoder_layers=3,
                      q_n_layers=2)
    encoder = StateEncoder(cfg)
    emb = ActionEmbeddingTable(3, cfg)
    q_head = QHead(cfg)

    print(f"  Encoder:    {encoder.count_parameters():>8,} params")
    print(f"  Embedding:  {emb.count_parameters():>8,} params")
    print(f"  QHead:      {q_head.count_parameters():>8,} params")

    # Verify they all have positive parameter counts
    assert encoder.count_parameters() > 0
    assert emb.count_parameters() > 0
    assert q_head.count_parameters() > 0

    # Full agent
    agent = ActionIncrementalDQN(cfg, n_actions=3)
    count_params(agent.encoder)
    print("  [OK] All components have positive parameters")


def test_gradient_flow():
    """Test that gradients flow through the full computation graph."""
    print("[test] Gradient flow...", end=" ")

    cfg = ModelConfig(obs_dim=64, n_actions_init=3, d_embedding=16, encoder_dim=64,
                      d_model=64, q_hidden_dim=32, buffer_capacity=100, batch_size=4)
    agent = _make_agent(cfg)

    # Fill buffer
    for _ in range(20):
        s = np.random.randn(64).astype(np.float32)
        ns = np.random.randn(64).astype(np.float32)
        agent.replay_buffer.push(s, int(np.random.randint(0, 3)), 1.0, ns, False)

    # Run update
    batch = agent.replay_buffer.sample(cfg.batch_size)
    metrics = agent.update(batch)

    # Check all modules have gradients
    for name, module in [
        ("encoder", agent.encoder),
        ("action_embeddings", agent.action_embeddings),
        ("q_head", agent.q_head),
    ]:
        has_grad = any(p.grad is not None for p in module.parameters())
        assert has_grad, f"No gradients in {name}"

    print("OK (gradients flow through all modules)")


def test_batch_q_values():
    """Test batch-computed Q-values for all actions."""
    print("[test] Batch Q-values...", end=" ")

    cfg = ModelConfig(obs_dim=64, n_actions_init=4, d_embedding=16, encoder_dim=64,
                      d_model=64, q_hidden_dim=32)
    agent = _make_agent(cfg)

    s = np.random.randn(64).astype(np.float32)
    s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)                  # (1, obs_dim)

    with torch.no_grad():
        phi_s = agent.encoder(s_t)                                           # (1, d_enc)
        q_all = agent._batch_q_values(phi_s)                                 # (action_count,)

    assert q_all.shape == (4,), f"Expected (4,), got {q_all.shape}"
    assert not torch.isnan(q_all).any()

    print(f"OK ({q_all.shape})")


def test_embedding_gradient_hook():
    """Test that the gradient freeze mask works during backward."""
    print("[test] Embedding gradient hook...", end=" ")

    cfg = ModelConfig(d_embedding=16)
    emb = ActionEmbeddingTable(4, cfg)

    # Freeze rows 0 and 1
    emb.set_rows_frozen(torch.tensor([0, 1]), frozen=True)
    assert emb._freeze_mask[0, 0].item() == 0.0
    assert emb._freeze_mask[3, 0].item() == 1.0

    # Forward + backward
    x = emb(torch.tensor([0, 1, 2, 3]))                                      # (4, d_emb)
    loss = x.sum()
    loss.backward()

    # Check gradients are zeroed for frozen rows
    assert emb.weight.grad is not None
    assert emb.weight.grad[0].abs().sum().item() == 0.0, "Row 0 grad should be zero"
    assert emb.weight.grad[1].abs().sum().item() == 0.0, "Row 1 grad should be zero"
    assert emb.weight.grad[2].abs().sum().item() > 0.0, "Row 2 grad should be non-zero"
    assert emb.weight.grad[3].abs().sum().item() > 0.0, "Row 3 grad should be non-zero"

    print("OK (gradients correctly masked)")


def test_cooldown_guard():
    """Test that expansion_cooldown prevents rapid successive expansions."""
    print("[test] Expansion cooldown...", end=" ")

    cfg = ModelConfig(obs_dim=64, n_actions_init=3, d_embedding=16, encoder_dim=64,
                      d_model=64, q_hidden_dim=32, expansion_cooldown=100)
    agent = _make_agent(cfg)

    # First expansion
    class MockEnv:
        def __init__(self):
            self.action_space_n = 4

    agent.handle_action_expansion(MockEnv())
    assert agent.action_count == 4

    # Immediately try another expansion (should be blocked by cooldown)
    class MockEnv2:
        def __init__(self):
            self.action_space_n = 5

    # The agent checks cooldown before calling handle_action_expansion
    steps_since = agent.env_step - agent.expansion_step
    cooldown_ok = steps_since >= cfg.expansion_cooldown
    assert not cooldown_ok, "Cooldown should block immediate re-expansion"

    # After enough steps, cooldown passes
    agent.env_step = agent.expansion_step + cfg.expansion_cooldown + 1
    steps_since = agent.env_step - agent.expansion_step
    cooldown_ok = steps_since >= cfg.expansion_cooldown
    assert cooldown_ok, "Cooldown should pass after enough steps"

    print("OK")


def test_agent_without_aux_loss():
    """Test agent without auxiliary dynamics loss."""
    print("[test] Agent without aux loss...", end=" ")

    cfg = ModelConfig(obs_dim=64, n_actions_init=3, d_embedding=16, encoder_dim=64,
                      d_model=64, q_hidden_dim=32, use_auxiliary_dynamics_loss=False,
                      buffer_capacity=100, batch_size=8)
    agent = ActionIncrementalDQN(cfg, n_actions=3)

    assert agent.aux_head is None

    for _ in range(20):
        agent.replay_buffer.push(np.random.randn(64).astype(np.float32),
                                 0, 1.0,
                                 np.random.randn(64).astype(np.float32),
                                 False)

    batch = agent.replay_buffer.sample(cfg.batch_size)
    metrics = agent.update(batch)
    assert metrics["aux_loss"] == 0.0
    print("OK")


def test_target_sync():
    """Test target network synchronisation."""
    print("[test] Target network sync...", end=" ")

    cfg = ModelConfig(obs_dim=64, n_actions_init=3, d_embedding=16, encoder_dim=64,
                      d_model=64, q_hidden_dim=32)
    agent = _make_agent(cfg)

    # Modify online networks
    with torch.no_grad():
        agent.encoder.net[0].weight.data += 1.0

    # Before sync, targets differ
    online_w = agent.encoder.net[0].weight.data
    target_w = agent.target_encoder.net[0].weight.data
    assert not torch.allclose(online_w, target_w), "Target should differ before sync"

    # Sync
    agent._sync_target()

    # After sync, targets match
    target_w2 = agent.target_encoder.net[0].weight.data
    assert torch.allclose(online_w, target_w2), "Target should match after sync"
    print("OK")


def test_exploration_bonus_decay():
    """Test exploration bonus decay mechanism."""
    print("[test] Exploration bonus decay...", end=" ")

    cfg = ModelConfig(obs_dim=64, n_actions_init=3, d_embedding=16, encoder_dim=64,
                      d_model=64, q_hidden_dim=32,
                      optimistic_init_bonus=2.0, optimistic_bonus_decay=0.5,
                      optimistic_bonus_min=0.1)
    agent = _make_agent(cfg)

    class MockEnv:
        def __init__(self):
            self.action_space_n = 4

    agent.handle_action_expansion(MockEnv())

    # Initially bonus is at max
    assert agent.action_bonuses[3] == 2.0

    # Simulate visits and decay
    agent.action_visit_counts[3] = 10
    for _ in range(5):
        agent.decay_exploration_bonuses()

    # Bonus should have decayed
    assert agent.action_bonuses[3] < 2.0
    assert agent.action_bonuses[3] >= 0.1  # clipped to min
    print(f"OK (bonus={agent.action_bonuses[3]:.3f})")


def test_epsilon_schedule():
    """Test epsilon annealing with expansion boost."""
    from train import compute_epsilon

    print("[test] Epsilon schedule...", end=" ")

    cfg = ModelConfig(epsilon_init=1.0, epsilon_min=0.05,
                      epsilon_decay_steps=1000, expansion_cooldown=200)

    # Early in training, epsilon near 1.0
    eps = compute_epsilon(cfg, env_step=10, last_expansion_step=0)
    assert eps > 0.9, f"Expected high epsilon early, got {eps}"

    # Late in training, epsilon near min
    eps = compute_epsilon(cfg, env_step=10000, last_expansion_step=0)
    assert abs(eps - 0.05) < 1e-6, f"Expected min epsilon, got {eps}"

    # After expansion, epsilon boosted
    eps = compute_epsilon(cfg, env_step=5000, last_expansion_step=4900)
    assert eps >= 0.5, f"Expected epsilon boost after expansion, got {eps}"

    # After cooldown, epsilon returns to normal
    eps = compute_epsilon(cfg, env_step=5200, last_expansion_step=4900)
    assert eps < 0.5, f"Expected normal epsilon after cooldown, got {eps}"

    print("OK")


def test_forward_after_multiple_expansions():
    """Test that the agent works correctly after multiple expansions."""
    print("[test] Multiple expansions...", end=" ")

    cfg = ModelConfig(obs_dim=64, n_actions_init=2, d_embedding=16, encoder_dim=64,
                      d_model=64, q_hidden_dim=32, k_nn=1, expansion_cooldown=0)
    agent = ActionIncrementalDQN(cfg, n_actions=2)

    class ExpandingMockEnv:
        def __init__(self):
            self.n = 2

        @property
        def action_space_n(self):
            return self.n

    env = ExpandingMockEnv()

    # Perform 3 expansions (2 -> 3 -> 4 -> 5)
    for expected in [3, 4, 5]:
        env.n = expected
        agent.handle_action_expansion(env)
        assert agent.action_count == expected, \
            f"Expected |A|={expected}, got {agent.action_count}"

    # Forward pass with 5 actions
    s = np.random.randn(64).astype(np.float32)
    a = agent.select_action(s, epsilon=0.0)
    assert 0 <= a < 5

    # Q-values for all 5 actions
    q_vals = agent.get_q_for_actions(s)
    assert len(q_vals) == 5, f"Expected 5 Q-values, got {len(q_vals)}"

    # TD update still works
    for _ in range(20):
        agent.replay_buffer.push(s, 0, 1.0, s, False)
    if len(agent.replay_buffer) >= cfg.batch_size:
        batch = agent.replay_buffer.sample(cfg.batch_size)
        metrics = agent.update(batch)
        assert not np.isnan(metrics["td_loss"])

    print("OK (2 -> 5 actions)")


# ── Run all tests ────────────────────────────────────────────────────────────

def main():
    """Run all smoke tests."""
    print("=" * 70)
    print("Action-Space Incremental RL — Smoke Test Suite")
    print("=" * 70)
    print()

    tests = [
        ("Config", test_model_config),
        ("Network components", lambda: None),  # placeholder
        ("  BaseOperator inheritance", test_base_operator),
        ("  StateEncoder (vector)", test_state_encoder_vector),
        ("  StateEncoder (image)", test_state_encoder_image),
        ("  ActionEmbeddingTable", test_action_embedding_table),
        ("  QHead", test_q_head),
        ("  ValueHead", test_value_head),
        ("  DuelingFactorizedQ", test_dueling_factorized_q),
        ("  DynamicsHead", test_dynamics_head),
        ("  Embedding gradient hook", test_embedding_gradient_hook),
        ("ReplayBuffer", test_replay_buffer),
        ("Agent", lambda: None),
        ("  Init (factorized + dueling)", test_agent_initialization),
        ("  select_action", test_agent_select_action),
        ("  TD update (factorized)", test_agent_update),
        ("  TD update (dueling)", test_agent_update_dueling),
        ("  Action expansion", test_agent_expansion),
        ("  Action expansion (dueling)", test_agent_expansion_dueling),
        ("  Freeze schedule", test_freeze_schedule),
        ("  Batch Q-values", test_batch_q_values),
        ("  Target network sync", test_target_sync),
        ("  Without aux loss", test_agent_without_aux_loss),
        ("  Exploration bonus decay", test_exploration_bonus_decay),
        ("  get_q_for_actions", test_get_q_for_actions),
        ("  Multiple expansions", test_forward_after_multiple_expansions),
        ("Environment wrapper", test_env_wrapper),
        ("Training helpers", lambda: None),
        ("  Epsilon schedule", test_epsilon_schedule),
        ("  Expansion cooldown", test_cooldown_guard),
        ("Parameter counts", test_parameter_counts),
        ("Gradient flow", test_gradient_flow),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        if test_fn.__name__ == "lambda":
            # Section header
            print(f"\n--- {name} ---")
            continue

        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 70)
    total = passed + failed
    print(f"Results: {passed}/{total} passed", end="")
    if failed > 0:
        print(f", {failed} FAILED", file=sys.stderr)
        sys.exit(1)
    else:
        print(" — ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
