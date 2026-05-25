"""Training loop for Action-Space Incremental DQN.

Detects action-space expansions from the environment and integrates
new actions into the agent without retraining from scratch.

Usage:
    python train.py --env dave --n_episodes 5000 --seed 42

Environment interface:
    The environment MUST expose action_space.n that can change mid-training.
    When env.action_space.n > agent.action_count, handle_action_expansion()
    is triggered automatically.
"""
import argparse
import logging
import time
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch

from model_config import ModelConfig
from agent import ActionIncrementalDQN, ReplayBuffer


def build_env(env_name: str, level: int = 1, seed: int = 42):
    """Create environment instance.

    For standard benchmarks (CartPole, etc.), wraps with a mock expanding
    action space. For custom environments (DAVE-analogue), loads directly.

    TODO: Replace with actual custom gymnasium env once implemented.
    """
    if env_name == "dave" or env_name == "custom":
        # Placeholder: custom Gymnasium environment with expanding action sets.
        # Must expose:
        #   env.action_space.n  — changes when new action appears
        #   env.unwrapped.level — current game level
        #   env.action_set      — list of available action IDs
        raise NotImplementedError(
            "Custom DAVE-game environment not yet implemented. "
            "See research contract: Gap 4 (standardized benchmark)."
        )

    elif env_name == "cartpole":
        # Demoware: standard CartPole with action injection at fixed timesteps.
        # Wraps the environment to simulate action-space expansion at step 1000.
        env = gym.make("CartPole-v1")
        env = ExpandingActionWrapper(env, expand_at=1000, new_action_name="do_nothing")
        return env

    elif env_name == "gridworld":
        # Demoware: simple gridworld where 'teleport' action unlocks at level 2.
        env = ExpandingActionWrapper(
            gym.make("GridWorld-v0") if hasattr(gym, "GridWorld") else gym.make("CartPole-v1"),
            expand_at=2000,
            new_action_name="teleport",
        )
        return env

    else:
        raise ValueError(f"Unknown environment: {env_name}")


class ExpandingActionWrapper(gym.Wrapper):
    """Minimal wrapper that simulates action-space expansion.

    At a fixed step threshold, expands the action space by one action.
    Used for development/testing before the full benchmark is built.
    """

    def __init__(self, env, expand_at: int, new_action_name: str = "new_action"):
        super().__init__(env)
        self.expand_at = expand_at
        self.new_action_name = new_action_name
        self.original_n = env.action_space.n
        self.step_count = 0
        self.has_expanded = False

    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.step_count += 1

        # Simulate expansion at threshold
        if not self.has_expanded and self.step_count >= self.expand_at:
            self._expand_action_space()

        return self.env.step(action)

    def _expand_action_space(self):
        self.has_expanded = True
        self.action_space = deepcopy(self.action_space)
        self.action_space.n = self.original_n + 1
        print(
            f"[env] Action space expanded from {self.original_n} → "
            f"{self.action_space.n} at step {self.step_count}"
        )


def compute_epsilon(cfg: ModelConfig, env_step: int, last_expansion_step: int) -> float:
    """Annealed epsilon for ε-greedy.

    After an action-space expansion, epsilon is temporarily boosted to
    encourage exploration of the new action.
    """
    # Base epsilon: linear decay
    fraction = min(1.0, env_step / cfg.epsilon_decay_steps)
    epsilon = cfg.epsilon_init + fraction * (cfg.epsilon_min - cfg.epsilon_init)

    # Expansion boost: if recently expanded, increase epsilon
    steps_since_expansion = env_step - last_expansion_step
    if steps_since_expansion < cfg.expansion_cooldown:
        epsilon = max(epsilon, 0.5)  # floor at 0.5 during cooldown

    return max(cfg.epsilon_min, epsilon)


def train_episode(
    agent: ActionIncrementalDQN,
    env,
    cfg: ModelConfig,
    epsilon: float,
) -> tuple[float, int, dict]:
    """Run a single training episode.

    Returns:
        (episode_reward, episode_length, metrics_dict)
    """
    s, _ = env.reset()
    episode_reward = 0.0
    episode_length = 0
    done = False
    q_vals_log = []
    expansion_occurred = False

    while not done:
        # ── Check for action-space expansion ──────────────────────────
        if env.action_space.n > agent.action_count:
            steps_since = agent.env_step - agent.expansion_step
            if steps_since >= cfg.expansion_cooldown:
                agent.handle_action_expansion(env)
                expansion_occurred = True

        # ── Select action ─────────────────────────────────────────────
        a = agent.select_action(s, epsilon)

        # Track visit counts for bonus decay
        if a in agent.new_action_indices:
            agent.action_visit_counts[a] += 1

        # ── Environment step ──────────────────────────────────────────
        s_next, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        episode_reward += r
        episode_length += 1

        # ── Store transition ──────────────────────────────────────────
        agent.replay_buffer.push(s, a, r, s_next, done)
        s = s_next
        agent.env_step += 1

        # ── Training step ─────────────────────────────────────────────
        if len(agent.replay_buffer) >= cfg.batch_size:
            for _ in range(cfg.gradient_steps):
                batch = agent.replay_buffer.sample(cfg.batch_size)
                metrics = agent.update(batch)

            # Target network update
            if agent.env_step % cfg.target_update_freq == 0:
                agent._sync_target()

            # Freeze schedule progression
            agent.maybe_update_freeze_schedule()

            # Decay exploration bonuses
            agent.decay_exploration_bonuses()

        # ── Log Q-values periodically (for degradation tracking) ──────
        if agent.env_step % 500 == 0:
            q_all = agent.get_q_for_actions(s)
            q_vals_log.append(q_all)

    return episode_reward, episode_length, {
        "epsilon": epsilon,
        "|A|": agent.action_count,
        "freeze_state": agent.freeze_state,
        "q_vals": np.array(q_vals_log) if q_vals_log else None,
        "expansion_occurred": expansion_occurred,
    }


def train(cfg: ModelConfig, env_name: str, n_episodes: int):
    """Main training loop with action-space expansion detection.

    Args:
        cfg: Model configuration.
        env_name: Environment name.
        n_episodes: Maximum episodes to run.
    """
    # Seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Environment
    env = build_env(env_name, level=1, seed=cfg.seed)
    initial_n_actions = env.action_space.n

    # Agent
    agent = ActionIncrementalDQN(cfg, n_actions=initial_n_actions)

    # Logging
    episode_rewards = []
    expansion_events = []
    q_degradation_log = []  # [(step, old_action_q_before, old_action_q_after)]

    print(f"Starting training | env={env_name} | |A_init|={initial_n_actions} | "
          f"cfg.n_actions_init={cfg.n_actions_init} | episodes={n_episodes}")
    print(f"Architecture: {cfg.q_form} | d_emb={cfg.d_embedding} | "
          f"k_nn={cfg.k_nn} | optimistic_bonus={cfg.optimistic_init_bonus}")
    print("-" * 80)

    start_time = time.time()
    last_expansion_step = 0

    for episode in range(1, n_episodes + 1):
        epsilon = compute_epsilon(cfg, agent.env_step, last_expansion_step)
        reward, length, info = train_episode(agent, env, cfg, epsilon)
        episode_rewards.append(reward)

        # Track expansion
        if info["expansion_occurred"]:
            expansion_events.append({
                "episode": episode,
                "env_step": agent.env_step,
                "|A|": info["|A|"],
            })
            last_expansion_step = agent.env_step

            # Log Q-values before expansion for degradation tracking
            q_pre = agent.get_q_for_actions(agent.replay_buffer.sample(1).state[0].numpy())
            q_degradation_log.append({
                "step": agent.env_step,
                "q_all": q_pre,
                "n_actions": len(q_pre),
            })

        # ── Logging ──────────────────────────────────────────────────
        if episode % cfg.log_freq == 0 or episode == 1:
            recent = episode_rewards[-cfg.log_freq:] if len(episode_rewards) >= cfg.log_freq else episode_rewards
            avg_reward = np.mean(recent)
            elapsed = time.time() - start_time
            print(
                f"Ep {episode:5d} | Step {agent.env_step:7d} | "
                f"Reward {avg_reward:+7.1f} | ε {epsilon:.3f} | "
                f"|A|={info['|A|']} | freeze={info['freeze_state']:>8s} | "
                f"buf={len(agent.replay_buffer):5d} | "
                f"time={elapsed:5.0f}s"
            )

        # ── Evaluation / checkpoint ──────────────────────────────────
        if episode % cfg.eval_freq == 0:
            _run_evaluation(agent, env, cfg)

        if episode % cfg.checkpoint_freq == 0:
            _save_checkpoint(agent, cfg, episode)

    # Final summary
    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"Training complete | {n_episodes} episodes | {elapsed:.0f}s")
    print(f"Total action expansions: {len(expansion_events)}")
    for ev in expansion_events:
        print(f"  Ep {ev['episode']} step {ev['env_step']}: |A| → {ev['|A|']}")
    print(f"Final |A| = {agent.action_count}")
    print("=" * 80)

    return {
        "episode_rewards": episode_rewards,
        "expansion_events": expansion_events,
        "q_degradation_log": q_degradation_log,
        "agent": agent,
    }


def _run_evaluation(agent: ActionIncrementalDQN, env, cfg: ModelConfig):
    """Evaluate agent (no exploration) and log performance."""
    eval_env = deepcopy(env)
    total_reward = 0.0
    n_eval_episodes = 5

    for _ in range(n_eval_episodes):
        s, _ = eval_env.reset()
        done = False
        while not done:
            with torch.no_grad():
                s_t = torch.tensor(s, dtype=torch.float32)
                if s_t.dim() == 1:
                    s_t = s_t.unsqueeze(0)
                phi_s = agent.encoder(s_t)

                q_vals = []
                for a in range(eval_env.action_space.n):
                    e_a = agent.action_embeddings(
                        torch.tensor([a], device=agent.device)
                    ).squeeze(0)
                    q = agent._compute_q(phi_s, e_a).item()
                    q_vals.append(q)

                a = int(np.argmax(q_vals))
            s, r, terminated, truncated, _ = eval_env.step(a)
            done = terminated or truncated
            total_reward += r

    avg_reward = total_reward / n_eval_episodes
    print(f"  [eval] |A|={eval_env.action_space.n} avg_reward={avg_reward:.2f}")


def _save_checkpoint(agent: ActionIncrementalDQN, cfg: ModelConfig, episode: int):
    """Save agent checkpoint."""
    path = f"checkpoint_ep{episode}.pt"
    torch.save(
        {
            "encoder": agent.encoder.state_dict(),
            "action_embeddings": agent.action_embeddings.state_dict(),
            "q_head": agent.q_head.state_dict(),
            "value_head": agent.value_head.state_dict() if agent.value_head else None,
            "action_count": agent.action_count,
            "freeze_state": agent.freeze_state,
            "new_action_indices": agent.new_action_indices,
            "cfg": cfg,
            "episode": episode,
        },
        path,
    )
    print(f"  [checkpoint] Saved to {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Action-Space Incremental DQN Training"
    )
    parser.add_argument("--env", default="dave", type=str,
                        help="Environment name (dave / cartpole / gridworld)")
    parser.add_argument("--n_episodes", default=5000, type=int,
                        help="Number of training episodes")
    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed")
    parser.add_argument("--q_form", default="factorized",
                        choices=["factorized", "dueling_factorized"],
                        help="Q-function architecture")
    parser.add_argument("--d_embedding", default=64, type=int,
                        help="Action embedding dimension")
    parser.add_argument("--k_nn", default=3, type=int,
                        help="k for k-NN embedding initialization")
    parser.add_argument("--optimistic_bonus", default=2.0, type=float,
                        help="Exploration bonus multiplier for new actions")
    parser.add_argument("--freeze_steps", default=10000, type=int,
                        help="Steps to keep encoder frozen after expansion")
    args = parser.parse_args()

    cfg = ModelConfig(
        n_actions_init=3,
        q_form=args.q_form,
        d_embedding=args.d_embedding,
        k_nn=args.k_nn,
        optimistic_init_bonus=args.optimistic_bonus,
        freeze_encoder_steps=args.freeze_steps,
        seed=args.seed,
    )

    results = train(cfg, args.env, args.n_episodes)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
