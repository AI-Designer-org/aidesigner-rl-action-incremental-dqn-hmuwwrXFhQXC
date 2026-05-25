"""Training loop for Action-Space Incremental DQN.

Detects action-space expansions from the environment and integrates
new actions into the agent without retraining from scratch.

Usage:
    python train.py --env cartpole --n_episodes 2000 --seed 42
    python train.py --env cartpole --q_form dueling_factorized --k_nn 5

Environment interface:
    The environment MUST expose action_space.n that can change mid-training.
    When env.action_space.n > agent.action_count, handle_action_expansion()
    is triggered automatically (with cooldown guard).

Evaluation protocol:
    - Every eval_freq episodes: evaluate with greedy policy for 5 episodes
    - Track Q-values before/after expansion for degradation measurement
    - Log: episode reward, epsilon, |A|, freeze state, buffer size
"""
import argparse
import time
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch

from model_config import ModelConfig
from agent import ActionIncrementalDQN
from env_wrapper import ExpandingActionWrapper
from networks import count_params


# ── Environment Builder ───────────────────────────────────────────────────────

def build_env(env_name: str, seed: int = 42):
    """Create environment instance with optional expansion wrapper.

    For standard benchmarks (CartPole, etc.), wraps with ExpandingActionWrapper
    to simulate action-space expansion at a fixed step threshold.

    For custom environments (DAVE-analogue), loads directly (TODO).

    Args:
        env_name: Environment identifier ("cartpole", "dave", "gridworld").
        seed: Random seed for reproducibility.
    Returns:
        Gymnasium environment instance.
    """
    if env_name == "dave" or env_name == "custom":
        # Placeholder: custom Gymnasium environment with expanding action sets.
        # Must expose:
        #   env.action_space.n  — changes when new action appears
        #   env.unwrapped.level — current game level
        #   env.action_set      — list of available action IDs
        raise NotImplementedError(
            "Custom DAVE-game environment not yet implemented. "
            "See research contract Gap 4 (standardized benchmark)."
        )

    elif env_name == "cartpole":
        # Demoware: CartPole with action injection at fixed timesteps.
        env = gym.make("CartPole-v1")
        env = ExpandingActionWrapper(
            env,
            expand_at_step=1000,
            new_action_name="do_nothing",
            max_expansions=1,
        )
        env.reset(seed=seed)
        return env

    elif env_name == "cartpole_multi":
        # Demoware: CartPole with MULTIPLE expansions.
        env = gym.make("CartPole-v1")
        env = ExpandingActionWrapper(
            env,
            expand_at_step=800,
            new_action_name="action_3",
            max_expansions=2,
        )
        env.reset(seed=seed)
        return env

    elif env_name == "gridworld":
        # Demoware: gridworld where 'teleport' action unlocks.
        env = ExpandingActionWrapper(
            gym.make("CartPole-v1"),  # placeholder for GridWorld
            expand_at_step=2000,
            new_action_name="teleport",
            max_expansions=1,
        )
        env.reset(seed=seed)
        return env

    else:
        raise ValueError(f"Unknown environment: {env_name}")


# ── Epsilon Schedule ──────────────────────────────────────────────────────────

def compute_epsilon(
    cfg: ModelConfig,
    env_step: int,
    last_expansion_step: int,
) -> float:
    """Annealed epsilon for epsilon-greedy exploration.

    After an action-space expansion, epsilon is temporarily boosted to
    encourage exploration of the new action. The boost decays as the
    expansion_cooldown window passes.

    Args:
        cfg: ModelConfig with epsilon schedule params.
        env_step: Current environment step count.
        last_expansion_step: Step of the most recent expansion (0 if none).

    Returns:
        Epsilon value in [cfg.epsilon_min, cfg.epsilon_init].
    """
    # Base epsilon: linear decay from init to min over epsilon_decay_steps
    fraction = min(1.0, env_step / max(1, cfg.epsilon_decay_steps))
    epsilon = cfg.epsilon_init + fraction * (cfg.epsilon_min - cfg.epsilon_init)

    # Expansion boost: floor at 0.5 during cooldown
    if last_expansion_step > 0:
        steps_since_expansion = env_step - last_expansion_step
        if steps_since_expansion < cfg.expansion_cooldown:
            epsilon = max(epsilon, 0.5)

    return max(cfg.epsilon_min, epsilon)


# ── Episode Runner ────────────────────────────────────────────────────────────

def train_episode(
    agent: ActionIncrementalDQN,
    env: gym.Env,
    cfg: ModelConfig,
    epsilon: float,
) -> tuple:
    """Run a single training episode.

    Args:
        agent: ActionIncrementalDQN agent.
        env: Gymnasium environment (may have expanding action space).
        cfg: ModelConfig.
        epsilon: Current epsilon for epsilon-greedy exploration.

    Returns:
        Tuple of (episode_reward, episode_length, info_dict).
    """
    s, _ = env.reset()
    episode_reward = 0.0
    episode_length = 0
    done = False
    truncated = False
    q_vals_log = []
    expansion_occurred = False

    while not done and not truncated:
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

        # Advance state and step counter
        s = s_next
        agent.env_step += 1

        # ── Training step (once buffer is warm) ───────────────────────
        if len(agent.replay_buffer) >= cfg.batch_size:
            for _ in range(cfg.gradient_steps):
                batch = agent.replay_buffer.sample(cfg.batch_size)
                metrics = agent.update(batch)

            # Target network update (hard sync at fixed interval)
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


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(agent: ActionIncrementalDQN, env: gym.Env, n_episodes: int = 5) -> float:
    """Evaluate agent with greedy (no exploration) policy.

    Args:
        agent: Agent to evaluate.
        env: Environment (evaluation instance, may differ from training env).
        n_episodes: Number of evaluation episodes.

    Returns:
        Average episode reward over n_episodes.
    """
    total_reward = 0.0

    for _ in range(n_episodes):
        s, _ = env.reset()
        done = False
        truncated = False

        while not done and not truncated:
            with torch.no_grad():
                s_t = torch.tensor(s, dtype=torch.float32, device=agent.device)
                if s_t.dim() == 1:
                    s_t = s_t.unsqueeze(0)                                 # (1, obs_dim)
                elif s_t.dim() == 3:
                    s_t = s_t.unsqueeze(0)                                 # (1, C, H, W)

                phi_s = agent.encoder(s_t)                                  # (1, d_enc)
                q_vals = agent._batch_q_values(phi_s)                      # (action_count,)
                a = int(q_vals.argmax().item())

            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total_reward += r

    return total_reward / n_episodes


# ── Checkpointing ────────────────────────────────────────────────────────────

def save_checkpoint(agent: ActionIncrementalDQN, cfg: ModelConfig, episode: int, path: str = None):
    """Save agent checkpoint to disk.

    Args:
        agent: Agent to save.
        cfg: ModelConfig (stored with checkpoint).
        episode: Current episode number.
        path: Output path. Defaults to checkpoint_ep{episode}.pt.
    """
    if path is None:
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
            "env_step": agent.env_step,
            "cfg": cfg,
            "episode": episode,
        },
        path,
    )
    print(f"  [checkpoint] Saved to {path}")


# ── Main Training Loop ───────────────────────────────────────────────────────

def train(cfg: ModelConfig, env_name: str, n_episodes: int):
    """Main training loop with action-space expansion detection.

    Args:
        cfg: Model configuration.
        env_name: Environment name (see build_env).
        n_episodes: Maximum episodes to run.

    Returns:
        Dict of training results including rewards, expansion events,
        Q-degradation logs, and the trained agent.
    """
    # ── Seeding ────────────────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── Environment ────────────────────────────────────────────────────
    env = build_env(env_name, seed=cfg.seed)
    initial_n_actions = int(env.action_space.n)

    # ── Agent ──────────────────────────────────────────────────────────
    agent = ActionIncrementalDQN(cfg, n_actions=initial_n_actions)

    # ── Logging ────────────────────────────────────────────────────────
    episode_rewards = []
    expansion_events = []
    q_degradation_log = []

    print(f"Starting training | env={env_name} | |A_init|={initial_n_actions} | "
          f"episodes={n_episodes}")
    print(f"Architecture: {cfg.q_form} | d_emb={cfg.d_embedding} | "
          f"k_nn={cfg.k_nn} | optimistic_bonus={cfg.optimistic_init_bonus}")
    print(f"Freeze schedule: encoder={cfg.freeze_encoder_steps} steps | "
          f"q_head={cfg.freeze_q_head_steps} steps")
    print("-" * 80)

    start_time = time.time()
    last_expansion_step = 0

    for episode in range(1, n_episodes + 1):
        # Compute epsilon with expansion boost
        epsilon = compute_epsilon(cfg, agent.env_step, last_expansion_step)

        # Run one episode
        reward, length, info = train_episode(agent, env, cfg, epsilon)
        episode_rewards.append(reward)

        # Track expansion events
        if info["expansion_occurred"]:
            expansion_events.append({
                "episode": episode,
                "env_step": agent.env_step,
                "|A|": info["|A|"],
            })
            last_expansion_step = agent.env_step

            # Sample a state and log Q-values for degradation tracking
            if len(agent.replay_buffer) > 0:
                sample_transition = agent.replay_buffer.sample(1)
                sample_state = sample_transition.state[0].cpu().numpy()
                q_pre = agent.get_q_for_actions(sample_state)
                q_degradation_log.append({
                    "step": agent.env_step,
                    "q_all": q_pre,
                    "n_actions": len(q_pre),
                })

        # ── Logging ──────────────────────────────────────────────────
        if episode % cfg.log_freq == 0 or episode == 1:
            recent = (episode_rewards[-cfg.log_freq:]
                      if len(episode_rewards) >= cfg.log_freq
                      else episode_rewards)
            avg_reward = np.mean(recent)
            elapsed = time.time() - start_time
            print(
                f"Ep {episode:5d} | Step {agent.env_step:7d} | "
                f"Reward {avg_reward:+7.1f} | eps {epsilon:.3f} | "
                f"|A|={info['|A|']} | freeze={info['freeze_state']:>8s} | "
                f"buf={len(agent.replay_buffer):5d} | "
                f"time={elapsed:5.0f}s"
            )

        # ── Evaluation ────────────────────────────────────────────────
        if episode % cfg.eval_freq == 0:
            eval_reward = evaluate(agent, env, n_episodes=5)
            print(f"  [eval] |A|={agent.action_count} "
                  f"avg_reward={eval_reward:.2f}")

        # ── Checkpoint ────────────────────────────────────────────────
        if episode % cfg.checkpoint_freq == 0:
            save_checkpoint(agent, cfg, episode)

    # ── Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"Training complete | {n_episodes} episodes | {elapsed:.0f}s")
    print(f"Total action expansions: {len(expansion_events)}")
    for ev in expansion_events:
        print(f"  Ep {ev['episode']} step {ev['env_step']}: |A| -> {ev['|A|']}")
    print(f"Final |A| = {agent.action_count}")
    print("=" * 80)

    return {
        "episode_rewards": episode_rewards,
        "expansion_events": expansion_events,
        "q_degradation_log": q_degradation_log,
        "agent": agent,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    """Entry point for command-line training."""
    parser = argparse.ArgumentParser(
        description="Action-Space Incremental DQN Training"
    )
    parser.add_argument(
        "--env", default="cartpole", type=str,
        choices=["cartpole", "cartpole_multi", "gridworld", "dave"],
        help="Environment name",
    )
    parser.add_argument(
        "--n_episodes", default=2000, type=int,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--seed", default=42, type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--q_form", default="factorized",
        choices=["factorized", "dueling_factorized"],
        help="Q-function architecture",
    )
    parser.add_argument(
        "--d_embedding", default=64, type=int,
        help="Action embedding dimension",
    )
    parser.add_argument(
        "--k_nn", default=3, type=int,
        help="k for k-NN embedding initialization",
    )
    parser.add_argument(
        "--optimistic_bonus", default=2.0, type=float,
        help="Exploration bonus multiplier for new actions",
    )
    parser.add_argument(
        "--freeze_steps", default=10000, type=int,
        help="Steps to keep encoder frozen after expansion",
    )
    parser.add_argument(
        "--no_aux_loss", action="store_true",
        help="Disable auxiliary dynamics loss",
    )
    args = parser.parse_args()

    cfg = ModelConfig(
        n_actions_init=3,
        q_form=args.q_form,
        d_embedding=args.d_embedding,
        k_nn=args.k_nn,
        optimistic_init_bonus=args.optimistic_bonus,
        freeze_encoder_steps=args.freeze_steps,
        use_auxiliary_dynamics_loss=not args.no_aux_loss,
        seed=args.seed,
    )

    results = train(cfg, args.env, args.n_episodes)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
