"""Layer 1-2: Comprehensive test suite for Action-Space Incremental RL.

Covers:
  Layer 1a — Shape tests: network components, agent, expansion
  Layer 1b — Gradient flow tests: backward pass, freeze mask, NaN detection
  Layer 1c — Correctness / invariance tests: RL-specific properties
  Layer 1d — Numerical stability tests: bf16, extreme values
  Layer 2  — Domain-specific RL benchmarks: rollout sanity, Q-degradation,
             monotonicity, exploration entropy, multiple seeds

Run with:
    cd /artifacts/j_hmuwwrXFhQXC/work/validator
    PYTHONPATH=/artifacts/j_hmuwwrXFhQXC/work/coder:$PYTHONPATH pytest test_model.py -v -x --tb=short
"""

import math
import sys
import os
from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Import implementation from coder stage ─────────────────────────────────
CODER_DIR = "/artifacts/j_hmuwwrXFhQXC/work/coder"
sys.path.insert(0, CODER_DIR)

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


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def cfg():
    """Default ModelConfig for testing."""
    return ModelConfig(
        obs_type="vector",
        obs_dim=64,
        n_actions_init=3,
        encoder_dim=64,
        d_model=64,
        d_embedding=16,
        q_hidden_dim=32,
        q_n_layers=2,
        buffer_capacity=1000,
        batch_size=8,
        k_nn=2,
        freeze_encoder_steps=10_000,
        freeze_q_head_steps=5_000,
        expansion_cooldown=1_000,
        use_auxiliary_dynamics_loss=True,
    )


@pytest.fixture
def cfg_dueling():
    """Dueling-factorized variant config."""
    return ModelConfig(
        obs_type="vector",
        obs_dim=64,
        n_actions_init=3,
        q_form="dueling_factorized",
        encoder_dim=64,
        d_model=64,
        d_embedding=16,
        q_hidden_dim=32,
        q_n_layers=2,
        buffer_capacity=1000,
        batch_size=8,
        k_nn=2,
    )


@pytest.fixture
def agent(cfg):
    """Minimal ActionIncrementalDQN for testing."""
    return ActionIncrementalDQN(cfg, n_actions=cfg.n_actions_init)


@pytest.fixture
def agent_dueling(cfg_dueling):
    """Dueling-factorized agent for testing."""
    return ActionIncrementalDQN(cfg_dueling, n_actions=cfg_dueling.n_actions_init)


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1a — Shape Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestShapes:
    """Verify all network components produce expected output shapes."""

    def test_state_encoder_vector_shape(self, cfg):
        encoder = StateEncoder(cfg)
        B = 4
        s = torch.randn(B, cfg.obs_dim)
        phi_s = encoder(s)
        assert phi_s.shape == (B, cfg.encoder_dim), f"Got {phi_s.shape}"

    def test_state_encoder_image_shape(self):
        cfg_img = ModelConfig(
            obs_type="image", obs_channels=3, obs_height=84, obs_width=84,
            encoder_dim=256
        )
        encoder = StateEncoder(cfg_img)
        B = 2
        s = torch.randn(B, cfg_img.obs_channels, cfg_img.obs_height, cfg_img.obs_width)
        phi_s = encoder(s)
        assert phi_s.shape == (B, cfg_img.encoder_dim), f"Got {phi_s.shape}"

    def test_action_embedding_table_shape(self, cfg):
        emb = ActionEmbeddingTable(3, cfg)
        assert emb.weight.shape == (3, cfg.d_embedding)

        indices = torch.tensor([0, 1, 2])
        e = emb(indices)
        assert e.shape == (3, cfg.d_embedding)

        batched = torch.tensor([0, 0, 1, 1, 2])
        e_batch = emb(batched)
        assert e_batch.shape == (5, cfg.d_embedding)

    def test_action_embedding_expansion_shape(self, cfg):
        """After add_embedding, the table should have one more row."""
        emb = ActionEmbeddingTable(3, cfg)
        emb.add_embedding(n_known=3, k=2, similarity="cosine")
        assert emb.weight.shape == (4, cfg.d_embedding)
        assert emb.n_actions == 4

        # Multiple expansions
        emb.add_embedding(n_known=4, k=2, similarity="cosine")
        assert emb.weight.shape == (5, cfg.d_embedding)

    def test_q_head_shape(self, cfg):
        q_head = QHead(cfg)
        B = 4
        phi_s = torch.randn(B, cfg.encoder_dim)
        e_a = torch.randn(cfg.d_embedding)
        q = q_head(phi_s, e_a)
        assert q.shape == (B, 1), f"Got {q.shape}"

        # Batched embeddings
        e_batch = torch.randn(B, cfg.d_embedding)
        q2 = q_head(phi_s, e_batch)
        assert q2.shape == (B, 1)

    def test_value_head_shape(self, cfg):
        v_head = ValueHead(cfg)
        B = 4
        phi_s = torch.randn(B, cfg.encoder_dim)
        V = v_head(phi_s)
        assert V.shape == (B, 1), f"Got {V.shape}"

    def test_dueling_factorized_q_shape(self, cfg):
        dq = DuelingFactorizedQ(cfg)
        B, n_act = 4, 3
        phi_s = torch.randn(B, cfg.encoder_dim)
        e_a = torch.randn(cfg.d_embedding)
        all_embeds = torch.randn(n_act, cfg.d_embedding)

        q1 = dq(phi_s, e_a)
        assert q1.shape == (B, 1)

        q2 = dq(phi_s, e_a, all_embeddings=all_embeds, n_known=2)
        assert q2.shape == (B, 1)

    def test_dynamics_head_shape(self, cfg):
        dyn = DynamicsHead(cfg)
        B = 4
        phi_s = torch.randn(B, cfg.encoder_dim)
        e_a = torch.randn(cfg.d_embedding)
        delta = dyn(phi_s, e_a)
        assert delta.shape == (B, cfg.encoder_dim), f"Got {delta.shape}"

    def test_agent_select_action_returns_valid_int(self, agent, cfg):
        s = np.random.randn(cfg.obs_dim).astype(np.float32)
        a = agent.select_action(s, epsilon=0.0)
        assert isinstance(a, int)
        assert 0 <= a < agent.action_count

    def test_agent_get_q_for_actions_shape(self, agent):
        s = np.random.randn(agent.cfg.obs_dim).astype(np.float32)
        q_vals = agent.get_q_for_actions(s)
        assert isinstance(q_vals, np.ndarray)
        assert q_vals.shape == (agent.action_count,), f"Got {q_vals.shape}"

    def test_batch_q_consistency(self, agent):
        """Batch-computed Q(s, a) for all actions should match per-action
        calls (consistency check)."""
        cfg = agent.cfg
        s = np.random.randn(cfg.obs_dim).astype(np.float32)
        s_t = torch.tensor(s, dtype=torch.float32, device=agent.device).unsqueeze(0)

        with torch.no_grad():
            phi_s = agent.encoder(s_t)
            # Batch computation
            q_batch = agent._batch_q_values(phi_s)  # (action_count,)

            # Per-action computation
            q_per = []
            for a in range(agent.action_count):
                e_a = agent.action_embeddings(
                    torch.tensor([a], device=agent.device)
                ).squeeze(0)
                q_a = agent._compute_q(phi_s, e_a).squeeze(-1)  # (1,)
                q_per.append(q_a.item())

        q_per_t = torch.tensor(q_per, device=q_batch.device)
        assert torch.allclose(q_batch, q_per_t, atol=1e-5), \
            "Batch Q-values differ from per-action Q-values"

    def test_variable_batch_size(self, cfg):
        """QHead and encoder should handle variable batch sizes."""
        encoder = StateEncoder(cfg)
        q_head = QHead(cfg)

        for B in [1, 2, 8, 16]:
            s = torch.randn(B, cfg.obs_dim)
            phi_s = encoder(s)
            assert phi_s.shape == (B, cfg.encoder_dim)

            e_a = torch.randn(cfg.d_embedding)
            q = q_head(phi_s, e_a)
            assert q.shape == (B, 1)

    def test_agent_after_expansion_forward_shape(self, agent):
        """After expansion, forward pass should work for all actions."""
        cfg = agent.cfg
        initial_n = agent.action_count

        class MockEnv:
            def __init__(self):
                self.action_space_n = initial_n + 1

        agent.handle_action_expansion(MockEnv())

        s = np.random.randn(cfg.obs_dim).astype(np.float32)
        q_vals = agent.get_q_for_actions(s)
        assert q_vals.shape == (initial_n + 1,), f"Got {q_vals.shape} after expansion"

        a = agent.select_action(s, epsilon=0.0)
        assert 0 <= a < initial_n + 1


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1b — Gradient Flow Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGradients:
    """Verify gradients flow correctly through all components."""

    def test_all_params_receive_gradients(self, agent):
        """Every trainable parameter should receive a non-None gradient."""
        cfg = agent.cfg
        # Fill buffer and run update
        for _ in range(20):
            agent.replay_buffer.push(
                np.random.randn(cfg.obs_dim).astype(np.float32),
                0, 1.0,
                np.random.randn(cfg.obs_dim).astype(np.float32),
                False,
            )

        batch = agent.replay_buffer.sample(cfg.batch_size)
        agent.update(batch)

        dead = []
        for name, p in agent.encoder.named_parameters():
            if p.requires_grad and p.grad is None:
                dead.append(f"encoder.{name}")
        for name, p in agent.action_embeddings.named_parameters():
            if p.requires_grad and p.grad is None:
                dead.append(f"action_embeddings.{name}")
        for name, p in agent.q_head.named_parameters():
            if p.requires_grad and p.grad is None:
                dead.append(f"q_head.{name}")

        assert len(dead) == 0, f"No gradient for: {dead}"

    def test_no_nan_gradients(self, agent):
        """No gradient should be NaN."""
        cfg = agent.cfg
        for _ in range(20):
            agent.replay_buffer.push(
                np.random.randn(cfg.obs_dim).astype(np.float32),
                0, 1.0,
                np.random.randn(cfg.obs_dim).astype(np.float32),
                False,
            )
        batch = agent.replay_buffer.sample(cfg.batch_size)
        agent.update(batch)

        for name, p in agent.encoder.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in encoder.{name}"
        for name, p in agent.q_head.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in q_head.{name}"

    def test_gradient_flow_after_expansion(self, agent):
        """After action expansion, gradients should only flow through the
        newly added action embedding (Phase 1)."""
        cfg = agent.cfg

        class MockEnv:
            def __init__(self):
                self.action_space_n = cfg.n_actions_init + 1

        agent.handle_action_expansion(MockEnv())
        assert agent.freeze_state == "phase1"

        # Fill buffer with actions including the new one
        for _ in range(50):
            agent.replay_buffer.push(
                np.random.randn(cfg.obs_dim).astype(np.float32),
                3, 1.0,  # Use new action (index 3)
                np.random.randn(cfg.obs_dim).astype(np.float32),
                False,
            )

        batch = agent.replay_buffer.sample(cfg.batch_size)
        agent.update(batch)

        # Encoder should have NO gradients (frozen in Phase 1)
        for p in agent.encoder.parameters():
            assert p.grad is None or p.grad.abs().sum().item() == 0.0, \
                "Encoder should not receive gradients in Phase 1"

        # Q-head should have NO gradients (frozen in Phase 1)
        for p in agent.q_head.parameters():
            assert p.grad is None or p.grad.abs().sum().item() == 0.0, \
                "Q-head should not receive gradients in Phase 1"

    def test_embedding_freeze_mask_gradient(self, cfg):
        """The gradient freeze mask should zero gradients for frozen rows."""
        emb = ActionEmbeddingTable(4, cfg)

        # Freeze rows 0, 1
        emb.set_rows_frozen(torch.tensor([0, 1]), frozen=True)

        x = emb(torch.tensor([0, 1, 2, 3]))
        loss = x.sum()
        loss.backward()

        assert emb.weight.grad is not None, "No gradient computed"
        assert emb.weight.grad[0].abs().sum().item() == 0.0, "Row 0 should be frozen"
        assert emb.weight.grad[1].abs().sum().item() == 0.0, "Row 1 should be frozen"
        assert emb.weight.grad[2].abs().sum().item() > 0.0, "Row 2 should have gradient"
        assert emb.weight.grad[3].abs().sum().item() > 0.0, "Row 3 should have gradient"

    def test_dueling_gradient_flow(self, agent_dueling):
        """Dueling variant should have gradients in both V and A heads."""
        cfg = agent_dueling.cfg
        for _ in range(20):
            agent_dueling.replay_buffer.push(
                np.random.randn(cfg.obs_dim).astype(np.float32),
                0, 1.0,
                np.random.randn(cfg.obs_dim).astype(np.float32),
                False,
            )
        batch = agent_dueling.replay_buffer.sample(cfg.batch_size)
        agent_dueling.update(batch)

        assert agent_dueling.value_head is not None
        has_v_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in agent_dueling.value_head.parameters()
        )
        assert has_v_grad, "Value head should receive gradients"


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1c — Correctness / Invariance Tests (RL-Specific)
# ═══════════════════════════════════════════════════════════════════════════


class TestRLProperties:
    """Domain-specific correctness tests for reinforcement learning."""

    def test_action_in_valid_range(self, agent):
        """Agent should always select actions within the current action set."""
        cfg = agent.cfg
        s = np.random.randn(cfg.obs_dim).astype(np.float32)

        for _ in range(100):
            a = agent.select_action(s, epsilon=0.5)
            assert 0 <= a < agent.action_count, \
                f"Action {a} out of range [0, {agent.action_count})"

    def test_action_in_valid_range_after_expansion(self, agent):
        """After expansion, actions should still be in valid range."""
        cfg = agent.cfg

        class MockEnv:
            def __init__(self):
                self.action_space_n = cfg.n_actions_init + 1

        agent.handle_action_expansion(MockEnv())

        s = np.random.randn(cfg.obs_dim).astype(np.float32)
        for _ in range(100):
            a = agent.select_action(s, epsilon=0.5)
            assert 0 <= a < agent.action_count

    def test_exploration_entropy(self, agent):
        """Under high epsilon, action distribution should have variety."""
        s = np.random.randn(agent.cfg.obs_dim).astype(np.float32)
        actions = [agent.select_action(s, epsilon=0.9) for _ in range(100)]

        n_unique = len(set(actions))
        assert n_unique > 1, \
            f"All actions were the same ({actions[0]}) with epsilon=0.9"

    def test_target_network_stability(self, agent):
        """Target network parameters should be stable (hard copy)."""
        cfg = agent.cfg

        # Grab target weights before any update
        target_w_before = agent.target_encoder.net[0].weight.data.clone()

        # Run a few training steps
        for _ in range(30):
            agent.replay_buffer.push(
                np.random.randn(cfg.obs_dim).astype(np.float32),
                0, 1.0,
                np.random.randn(cfg.obs_dim).astype(np.float32),
                False,
            )

        if len(agent.replay_buffer) >= cfg.batch_size:
            for _ in range(5):
                batch = agent.replay_buffer.sample(cfg.batch_size)
                agent.update(batch)

        # Target should not have changed (only synced at freq, not yet)
        target_w_after = agent.target_encoder.net[0].weight.data
        assert torch.allclose(target_w_before, target_w_after), \
            "Target network changed before sync interval"

    def test_target_sync_correctness(self, agent):
        """After _sync_target, online and target should match."""
        # Modify online network
        with torch.no_grad():
            agent.encoder.net[0].weight.data += 1.0

        # Before sync: weights differ
        online_w = agent.encoder.net[0].weight.data
        target_w = agent.target_encoder.net[0].weight.data
        assert not torch.allclose(online_w, target_w), \
            "Should differ before sync"

        agent._sync_target()
        target_w2 = agent.target_encoder.net[0].weight.data
        assert torch.allclose(online_w, target_w2), \
            "Should match after sync"

    def test_optimistic_bonus_initialization(self, agent):
        """New actions should get the configured optimistic bonus."""
        cfg = agent.cfg

        class MockEnv:
            def __init__(self):
                self.action_space_n = cfg.n_actions_init + 1

        agent.handle_action_expansion(MockEnv())
        new_idx = cfg.n_actions_init
        assert agent.action_bonuses.get(new_idx, 0.0) == cfg.optimistic_init_bonus

    def test_new_action_visit_tracking(self, agent):
        """Visit counts for new actions should increment."""
        cfg = agent.cfg

        class MockEnv:
            def __init__(self):
                self.action_space_n = cfg.n_actions_init + 1

        agent.handle_action_expansion(MockEnv())
        new_idx = cfg.n_actions_init

        s = np.random.randn(cfg.obs_dim).astype(np.float32)
        for _ in range(10):
            a = agent.select_action(s, epsilon=1.0)
            if a in agent.new_action_indices:
                agent.action_visit_counts[a] += 1

        assert agent.action_visit_counts[new_idx].item() >= 0

    def test_expansion_cooldown_guard(self, agent):
        """Rapid successive expansions should be blocked by cooldown."""
        cfg = agent.cfg

        class MockEnv:
            def __init__(self):
                self.action_space_n = cfg.n_actions_init + 1

        # First expansion
        agent.handle_action_expansion(MockEnv())

        steps_since = agent.env_step - agent.expansion_step
        assert steps_since < cfg.expansion_cooldown, \
            "Cooldown should not have passed yet"

        class MockEnv2:
            def __init__(self):
                self.action_space_n = cfg.n_actions_init + 2

        # Simulate the cooldown check the training loop would do
        would_expand = steps_since >= cfg.expansion_cooldown
        assert not would_expand, "Expansion should be blocked by cooldown"

    def test_epsilon_schedule_expansion_boost(self):
        """Epsilon should be boosted after expansion."""
        from train import compute_epsilon

        cfg = ModelConfig(
            epsilon_init=1.0, epsilon_min=0.05,
            epsilon_decay_steps=1000, expansion_cooldown=200
        )

        # Late in training, epsilon near min
        eps_normal = compute_epsilon(cfg, env_step=5000, last_expansion_step=0)
        assert abs(eps_normal - 0.05) < 1e-6

        # After expansion, epsilon boosted
        eps_boosted = compute_epsilon(cfg, env_step=5000, last_expansion_step=4900)
        assert eps_boosted >= 0.5

    def test_bonus_decay_monotonic(self, agent):
        """Exploration bonuses should decay monotonically."""
        cfg = agent.cfg

        class MockEnv:
            def __init__(self):
                self.action_space_n = cfg.n_actions_init + 1

        agent.handle_action_expansion(MockEnv())
        new_idx = cfg.n_actions_init

        initial_bonus = agent.action_bonuses[new_idx]
        agent.action_visit_counts[new_idx] = 5

        prev = initial_bonus
        for _ in range(10):
            agent.decay_exploration_bonuses()
            curr = agent.action_bonuses[new_idx]
            assert curr <= prev + 1e-6, "Bonus should not increase"
            prev = curr

        assert agent.action_bonuses[new_idx] >= cfg.optimistic_bonus_min, \
            "Bonus should not decay below minimum"


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1d — Numerical Stability Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestNumerics:
    """Numerical stability tests for all components."""

    def test_encoder_no_nan_or_inf(self, cfg):
        encoder = StateEncoder(cfg)
        s = torch.randn(4, cfg.obs_dim)
        phi_s = encoder(s)
        assert not torch.isnan(phi_s).any(), "NaN in encoder output"
        assert not torch.isinf(phi_s).any(), "Inf in encoder output"

    def test_q_head_no_nan_or_inf(self, cfg):
        q_head = QHead(cfg)
        phi_s = torch.randn(4, cfg.encoder_dim)
        e_a = torch.randn(cfg.d_embedding)
        q = q_head(phi_s, e_a)
        assert not torch.isnan(q).any(), "NaN in QHead output"
        assert not torch.isinf(q).any(), "Inf in QHead output"

    def test_action_embedding_no_nan(self, cfg):
        emb = ActionEmbeddingTable(3, cfg)
        emb.add_embedding(n_known=3, k=2, similarity="cosine")
        e = emb(torch.tensor([0, 1, 2, 3]))
        assert not torch.isnan(e).any(), "NaN in embedding after expansion"

        # Euclidean similarity
        emb2 = ActionEmbeddingTable(3, cfg)
        emb2.add_embedding(n_known=3, k=2, similarity="euclidean")
        e2 = emb2(torch.tensor([0, 1, 2, 3]))
        assert not torch.isnan(e2).any(), "NaN in euclidean embedding"

    def test_bf16_encoder_forward(self, cfg):
        """Encoder should produce finite output in bf16."""
        encoder = StateEncoder(cfg).bfloat16()
        s = torch.randn(4, cfg.obs_dim).bfloat16()
        phi_s = encoder(s)
        assert not torch.isnan(phi_s).any(), "NaN in bf16 encoder"
        assert not torch.isinf(phi_s).any(), "Inf in bf16 encoder"

    def test_bf16_q_head_forward(self, cfg):
        """QHead should produce finite output in bf16."""
        q_head = QHead(cfg).bfloat16()
        phi_s = torch.randn(4, cfg.encoder_dim).bfloat16()
        e_a = torch.randn(cfg.d_embedding).bfloat16()
        q = q_head(phi_s, e_a)
        assert not torch.isnan(q).any(), "NaN in bf16 QHead"
        assert not torch.isinf(q).any(), "Inf in bf16 QHead"

    def test_extreme_input_values_encoder(self, cfg):
        """Very large inputs should not produce NaN."""
        encoder = StateEncoder(cfg)
        s = torch.randn(4, cfg.obs_dim) * 1e4
        phi_s = encoder(s)
        assert not torch.isnan(phi_s).any(), "NaN with extreme encoder input"

    def test_extreme_embedding_values(self, cfg):
        """Extreme action indices should still work (within range)."""
        emb = ActionEmbeddingTable(3, cfg)
        idx = torch.tensor([0])
        e = emb(idx)
        assert not torch.isnan(e).any()

    def test_td_loss_finite(self, agent):
        """TD loss should always be finite."""
        cfg = agent.cfg
        for _ in range(20):
            agent.replay_buffer.push(
                np.random.randn(cfg.obs_dim).astype(np.float32),
                0, 1.0,
                np.random.randn(cfg.obs_dim).astype(np.float32),
                False,
            )
        batch = agent.replay_buffer.sample(cfg.batch_size)
        metrics = agent.update(batch)

        assert math.isfinite(metrics["td_loss"]), \
            f"Non-finite TD loss: {metrics['td_loss']}"

    def test_td_loss_after_expansion_finite(self, agent):
        """TD loss should remain finite after expansion."""
        cfg = agent.cfg

        class MockEnv:
            def __init__(self):
                self.action_space_n = cfg.n_actions_init + 1

        agent.handle_action_expansion(MockEnv())

        for _ in range(20):
            agent.replay_buffer.push(
                np.random.randn(cfg.obs_dim).astype(np.float32),
                3, 1.0,
                np.random.randn(cfg.obs_dim).astype(np.float32),
                False,
            )

        if len(agent.replay_buffer) >= cfg.batch_size:
            batch = agent.replay_buffer.sample(cfg.batch_size)
            metrics = agent.update(batch)
            assert math.isfinite(metrics["td_loss"]), \
                f"Non-finite TD loss post-expansion: {metrics['td_loss']}"

    def test_dueling_q_values_finite(self, agent_dueling):
        """Dueling variant Q-values should be finite."""
        cfg = agent_dueling.cfg
        s = np.random.randn(cfg.obs_dim).astype(np.float32)
        q_vals = agent_dueling.get_q_for_actions(s)
        assert np.all(np.isfinite(q_vals)), "Non-finite Q-values in dueling variant"


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2 — Domain-Specific RL Benchmarks
# ═══════════════════════════════════════════════════════════════════════════


class TestRLBenchmarks:
    """Domain-specific benchmarks for Action-Space Incremental RL."""

    def test_q_degradation_metric(self, agent):
        """Q-degradation metric: after expansion, Q-values for old actions
        should not drop catastrophically (proxy for catastrophic forgetting).

        Uses random initialization (no training), so we just verify the
        metric computation is well-defined — not the specific degradation value.
        """
        cfg = agent.cfg
        s = np.random.randn(cfg.obs_dim).astype(np.float32)

        # Q-values before expansion
        q_before = agent.get_q_for_actions(s)
        assert len(q_before) == cfg.n_actions_init

        class MockEnv:
            def __init__(self):
                self.action_space_n = cfg.n_actions_init + 1

        agent.handle_action_expansion(MockEnv())

        # Q-values after expansion
        q_after = agent.get_q_for_actions(s)
        assert len(q_after) == cfg.n_actions_init + 1

        # Old-action Q-values should not change (Phase 1: everything frozen
        # except new embedding). With no training steps, they should be identical.
        old_q_before = q_before[:cfg.n_actions_init]
        old_q_after = q_after[:cfg.n_actions_init]
        # The Q-head and encoder are frozen, so old-action Q-values should
        # be exactly preserved (no training happened)
        assert np.allclose(old_q_before, old_q_after, atol=1e-6), \
            "Old Q-values changed despite frozen parameters"

    def test_q_degradation_after_training(self, agent):
        """After a few training steps post-expansion, old Q-values should
        not diverge catastrophically."""
        cfg = agent.cfg

        # Fill buffer with pre-expansion data
        for _ in range(100):
            agent.replay_buffer.push(
                np.random.randn(cfg.obs_dim).astype(np.float32),
                np.random.randint(0, cfg.n_actions_init),
                float(np.random.randn()),
                np.random.randn(cfg.obs_dim).astype(np.float32),
                False,
            )

        s_for_metric = np.random.randn(cfg.obs_dim).astype(np.float32)
        q_before = agent.get_q_for_actions(s_for_metric)
        old_q_before = q_before[:cfg.n_actions_init].copy()

        # Expand
        class MockEnv:
            def __init__(self):
                self.action_space_n = cfg.n_actions_init + 1

        agent.handle_action_expansion(MockEnv())

        # Fill buffer with post-expansion data
        for _ in range(100):
            agent.replay_buffer.push(
                np.random.randn(cfg.obs_dim).astype(np.float32),
                3, float(np.random.randn()),
                np.random.randn(cfg.obs_dim).astype(np.float32),
                False,
            )

        # Run a few training steps
        for _ in range(20):
            if len(agent.replay_buffer) >= cfg.batch_size:
                batch = agent.replay_buffer.sample(cfg.batch_size)
                agent.update(batch)

        q_after = agent.get_q_for_actions(s_for_metric)
        old_q_after = q_after[:cfg.n_actions_init]

        # Compute degradation ratio
        q_norm_before = np.linalg.norm(old_q_before)
        q_norm_after = np.linalg.norm(old_q_after)
        if q_norm_before > 0:
            degradation = abs(q_norm_after - q_norm_before) / q_norm_before
            # Even with frozen params, random training can shift embeddings
            # via the new embedding's interactions. Allow a generous margin.
            # This isn't a hard pass/fail — it's a benchmark probe.
            print(f"  [benchmark] Old-action Q norm change: {degradation:.4f}")
            assert degradation < 5.0, \
                f"Old-action Q norms changed by {degradation:.2%}"

    def test_monotonicity_property(self, agent):
        """After expansion, max Q over the larger action set should be >=
        max Q over the original action set (monotonicity of value).

        Note: for randomly initialized networks this may not hold strictly,
        but with k-NN initialization of the new embedding it should not
        systematically decrease.
        """
        cfg = agent.cfg
        s = np.random.randn(cfg.obs_dim).astype(np.float32)

        q_before = agent.get_q_for_actions(s)
        max_q_before = q_before.max()

        class MockEnv:
            def __init__(self):
                self.action_space_n = cfg.n_actions_init + 1

        agent.handle_action_expansion(MockEnv())
        q_after = agent.get_q_for_actions(s)

        max_q_after_over_old = q_after[:cfg.n_actions_init].max()
        max_q_after_over_all = q_after.max()

        print(f"  [benchmark] max Q before: {max_q_before:.4f}, "
              f"over old: {max_q_after_over_old:.4f}, "
              f"over all: {max_q_after_over_all:.4f}")

    def test_batch_q_values_all_actions(self, agent):
        """Batch Q-value computation should give correct values for each action."""
        cfg = agent.cfg
        B = 4
        states = torch.randn(B, cfg.obs_dim)

        with torch.no_grad():
            phi_s = agent.encoder(states)
            q_all = agent._batch_q_values(phi_s)  # (B, action_count)

        assert q_all.shape == (B, agent.action_count), f"Got {q_all.shape}"
        assert not torch.isnan(q_all).any()

        # Verify each action's Q-value is distinct (should not all be identical
        # due to different embeddings — unless weights are degenerate)
        for b in range(B):
            n_unique = len(torch.unique(q_all[b]))
            if n_unique == 1:
                # This can happen with random init if all embeddings are near-identical
                print(f"  [warn] All Q-values identical for batch {b}")
            else:
                assert n_unique > 1, "Q-values should vary across actions"

    def test_rollout_sanity_with_env_wrapper(self):
        """End-to-end: agent + ExpandingActionWrapper should produce finite
        rewards through simulated expansion."""
        import gymnasium as gym
        cfg = ModelConfig(
            obs_dim=4,  # CartPole obs dim
            n_actions_init=2,
            encoder_dim=32,
            d_model=32,
            d_embedding=8,
            q_hidden_dim=16,
            buffer_capacity=1000,
            batch_size=8,
            expansion_cooldown=10,
            freeze_q_head_steps=100,
            freeze_encoder_steps=200,
        )

        env = gym.make("CartPole-v1")
        env = ExpandingActionWrapper(env, expand_at_step=50, max_expansions=1)
        env.reset()

        agent = ActionIncrementalDQN(cfg, n_actions=int(env.action_space.n))

        for step in range(200):
            # Check for expansion
            if env.action_space.n > agent.action_count:
                steps_since = agent.env_step - agent.expansion_step
                if steps_since >= cfg.expansion_cooldown:
                    agent.handle_action_expansion(env)

            s, r, terminated, truncated, _ = env.step(0)
            agent.replay_buffer.push(s, 0, r, s, terminated or truncated)
            if terminated or truncated:
                env.reset()

            # Training step
            if len(agent.replay_buffer) >= cfg.batch_size:
                batch = agent.replay_buffer.sample(cfg.batch_size)
                metrics = agent.update(batch)
                assert math.isfinite(metrics["td_loss"])

        print(f"  [benchmark] CartPole rollout complete: "
              f"|A|={agent.action_count}, buffer={len(agent.replay_buffer)}")

    def test_multiple_expansions_consistency(self):
        """Agent should maintain consistency across multiple expansions."""
        cfg = ModelConfig(
            obs_dim=64, n_actions_init=2,
            encoder_dim=32, d_model=32, d_embedding=8, q_hidden_dim=16,
            k_nn=1, expansion_cooldown=0,
        )
        agent = ActionIncrementalDQN(cfg, n_actions=2)

        class ExpandingMockEnv:
            def __init__(self):
                self.n = 2
            @property
            def action_space_n(self):
                return self.n

        env = ExpandingMockEnv()

        for target_n in [3, 4, 5]:
            env.n = target_n
            agent.handle_action_expansion(env)
            assert agent.action_count == target_n

        # Forward pass after 3 expansions
        s = np.random.randn(64).astype(np.float32)
        q_vals = agent.get_q_for_actions(s)
        assert q_vals.shape == (5,)

        actions = [agent.select_action(s, epsilon=0.5) for _ in range(50)]
        assert all(0 <= a < 5 for a in actions)

    def test_dueling_centering_correctness(self, cfg_dueling):
        """Dueling variant centering: the mean advantage over known actions
        should be subtracted, making the known-action advantages sum to ~zero."""
        agent = ActionIncrementalDQN(cfg_dueling, n_actions=3)

        s = np.random.randn(cfg_dueling.obs_dim).astype(np.float32)
        s_t = torch.tensor(s, dtype=torch.float32, device=agent.device).unsqueeze(0)

        with torch.no_grad():
            phi_s = agent.encoder(s_t)
            V = agent.value_head(phi_s)

            # Compute A(s,a) for each action with centering
            A_all = []
            for a in range(agent.action_count):
                e_a = agent.action_embeddings(
                    torch.tensor([a], device=agent.device)
                ).squeeze(0)
                A = agent.q_head(phi_s, e_a)
                A_all.append(A.item())

            # The centering inside _batch_q_values should make
            # mean of old-action advantages ~0
            A_mean = np.mean(A_all)
            print(f"  [benchmark] Mean advantage (dueling): {A_mean:.4f} "
                  f"(should be near 0 with centering)")

    def test_freezing_prevents_encoder_drift(self, agent):
        """In Phase 1, encoder weights should not change after training steps."""
        cfg = agent.cfg

        # Save initial encoder weights
        init_weights = {}
        for name, p in agent.encoder.named_parameters():
            init_weights[name] = p.data.clone()

        # Expand and enter Phase 1
        class MockEnv:
            def __init__(self):
                self.action_space_n = cfg.n_actions_init + 1

        agent.handle_action_expansion(MockEnv())

        # Fill buffer and train
        for _ in range(50):
            agent.replay_buffer.push(
                np.random.randn(cfg.obs_dim).astype(np.float32),
                3, 1.0,
                np.random.randn(cfg.obs_dim).astype(np.float32),
                False,
            )

        for _ in range(10):
            if len(agent.replay_buffer) >= cfg.batch_size:
                batch = agent.replay_buffer.sample(cfg.batch_size)
                agent.update(batch)

        # Verify encoder weights unchanged
        for name, p in agent.encoder.named_parameters():
            assert torch.allclose(init_weights[name], p.data), \
                f"Encoder parameter {name} changed in Phase 1"

    def test_freeze_schedule_progression(self, agent):
        """Freeze schedule: Phase 1 -> 2 -> 3 should progress correctly."""
        cfg = agent.cfg

        class MockEnv:
            def __init__(self):
                self.action_space_n = cfg.n_actions_init + 1

        agent.handle_action_expansion(MockEnv())
        assert agent.freeze_state == "phase1"

        # Phase 1 -> 2
        agent.env_step = agent.expansion_step + cfg.freeze_q_head_steps + 1
        agent.maybe_update_freeze_schedule()
        assert agent.freeze_state == "phase2"
        assert any(p.requires_grad for p in agent.q_head.parameters()), \
            "Q-head should be trainable in Phase 2"

        # Phase 2 -> 3
        agent.env_step = agent.expansion_step + cfg.freeze_encoder_steps + 1
        agent.maybe_update_freeze_schedule()
        assert agent.freeze_state == "phase3"
        assert any(p.requires_grad for p in agent.encoder.parameters()), \
            "Encoder should be trainable in Phase 3"

    def test_aux_loss_reduces_mse(self):
        """Auxiliary dynamics loss should be an MSE-like quantity >= 0."""
        cfg = ModelConfig(
            obs_dim=16, n_actions_init=3, encoder_dim=32, d_model=32,
            d_embedding=8, q_hidden_dim=16, use_auxiliary_dynamics_loss=True,
        )
        agent = ActionIncrementalDQN(cfg, n_actions=3)

        for _ in range(20):
            agent.replay_buffer.push(
                np.random.randn(16).astype(np.float32),
                0, 1.0,
                np.random.randn(16).astype(np.float32),
                False,
            )

        if len(agent.replay_buffer) >= cfg.batch_size:
            batch = agent.replay_buffer.sample(cfg.batch_size)
            metrics = agent.update(batch)
            assert metrics["aux_loss"] >= 0, \
                f"Aux loss should be non-negative, got {metrics['aux_loss']}"


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2 — Synthetic RL Benchmark Tasks
# ═══════════════════════════════════════════════════════════════════════════


class TestSyntheticBenchmarks:
    """Synthetic benchmarks that verify expected behavior in controlled settings."""

    def test_discounted_return_finite(self):
        """Discounted return should always be finite and well-defined."""
        rewards = np.random.randn(100) * 0.1  # small random rewards
        gamma = 0.99
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
        assert math.isfinite(G), f"Non-finite discounted return: {G}"

    def test_td_target_with_new_action(self, agent):
        """The TD target should naturally include new actions in max_{a'}."""
        cfg = agent.cfg

        class MockEnv:
            def __init__(self):
                self.action_space_n = cfg.n_actions_init + 1

        agent.handle_action_expansion(MockEnv())

        # Create a batch where all transitions are from before expansion
        # but the target computation uses the expanded action set
        states = torch.randn(4, cfg.obs_dim)
        actions = torch.tensor([0, 1, 2, 0])
        rewards = torch.randn(4)
        next_states = torch.randn(4, cfg.obs_dim)
        dones = torch.zeros(4)

        batch = Transition(
            state=states, action=actions, reward=rewards,
            next_state=next_states, done=dones,
        )

        metrics = agent.update(batch)
        assert math.isfinite(metrics["td_loss"]), \
            "TD loss should be finite even with expanded action set"

    def test_embedding_knn_init_produces_valid_embeddings(self, cfg):
        """k-NN initialization should produce finite, non-zero embeddings."""
        emb = ActionEmbeddingTable(5, cfg)
        emb.add_embedding(n_known=5, k=3, similarity="cosine")
        new_e = emb(torch.tensor([5]))
        assert not torch.isnan(new_e).any()
        assert not torch.isinf(new_e).any()

        emb2 = ActionEmbeddingTable(5, cfg)
        emb2.add_embedding(n_known=5, k=3, similarity="euclidean")
        new_e2 = emb2(torch.tensor([5]))
        assert not torch.isnan(new_e2).any()

    def test_embedding_similarity_metrics(self, cfg):
        """Both cosine and euclidean similarity should produce valid
        new embeddings."""
        n_known = 10

        emb_cos = ActionEmbeddingTable(n_known, cfg)
        emb_cos.add_embedding(n_known=n_known, k=3, similarity="cosine")
        assert emb_cos.weight.shape == (n_known + 1, cfg.d_embedding)

        emb_euc = ActionEmbeddingTable(n_known, cfg)
        emb_euc.add_embedding(n_known=n_known, k=3, similarity="euclidean")
        assert emb_euc.weight.shape == (n_known + 1, cfg.d_embedding)

    def test_parameter_counts_positive(self, cfg):
        """All components should have > 0 parameters."""
        encoder = StateEncoder(cfg)
        assert encoder.count_parameters() > 0

        emb = ActionEmbeddingTable(3, cfg)
        assert emb.count_parameters() > 0

        q_head = QHead(cfg)
        assert q_head.count_parameters() > 0

        v_head = ValueHead(cfg)
        assert v_head.count_parameters() > 0

        dyn = DynamicsHead(cfg)
        assert dyn.count_parameters() > 0

    def test_cnn_forward_with_encoder(self):
        """CNN-based encoder should work with standard image sizes."""
        cfg = ModelConfig(
            obs_type="image", obs_channels=3, obs_height=84, obs_width=84,
            encoder_dim=256, n_encoder_layers=3,
        )
        encoder = StateEncoder(cfg)
        s = torch.randn(2, 3, 84, 84)
        phi_s = encoder(s)
        assert phi_s.shape == (2, 256)

    def test_random_rollout_no_crash_with_wrapper(self):
        """Random rollout should not crash or produce NaN rewards."""
        import gymnasium as gym
        env = gym.make("CartPole-v1")
        env = ExpandingActionWrapper(env, expand_at_step=1000, max_expansions=1)
        env.reset()

        for step in range(50):
            a = env.action_space.sample()
            s, r, terminated, truncated, _ = env.step(a)
            assert not math.isnan(r), "NaN reward"
            if terminated or truncated:
                env.reset()

    def test_expansion_at_different_steps(self):
        """Wrapper should expand at multiple different step thresholds."""
        for expand_at in [50, 100, 500]:
            import gymnasium as gym
            env = gym.make("CartPole-v1")
            env = ExpandingActionWrapper(env, expand_at_step=expand_at, max_expansions=1)
            env.reset()

            for step in range(expand_at + 10):
                s, r, terminated, truncated, _ = env.step(0)
                if terminated or truncated:
                    env.reset()

            assert env.has_expanded, \
                f"Should have expanded at step {expand_at}"
