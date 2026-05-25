"""Environment wrappers for Action-Space Incremental RL.

Provides:
  - ExpandingActionWrapper: wraps a standard Gymnasium environment to
    simulate action-space expansion at a fixed timestep.
  - Helper functions for building test environments.

The environment wrapper is used for development/testing before the full
custom benchmark (DAVE-game analogue, continuous control, goal-conditioned)
is implemented.
"""
from copy import deepcopy
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np


class ExpandingActionWrapper(gym.Wrapper):
    """Minimal wrapper that simulates action-space expansion.

    At a fixed step threshold (or episode threshold), expands the
    action space by one action. Used for development/testing before
    the full benchmark is built.

    Usage:
        env = gym.make("CartPole-v1")
        env = ExpandingActionWrapper(env, expand_at_step=1000)
        # After 1000 env steps, env.action_space.n increases by 1.

    The new action maps to a no-op (or a configurable action mapping)
    so that the agent can explore it without crashing.
    """

    def __init__(
        self,
        env: gym.Env,
        expand_at_step: int = 1000,
        new_action_name: str = "new_action",
        expand_at_episode: Optional[int] = None,
        max_expansions: int = 1,
    ):
        """Initialise the wrapper.

        Args:
            env: Base Gymnasium environment.
            expand_at_step: Env step at which to expand (mutually exclusive
                with expand_at_episode).
            new_action_name: Human-readable name for the new action.
            expand_at_episode: Episode at which to expand (alternative to
                expand_at_step). If set, expands at the start of this episode.
            max_expansions: Maximum number of expansions to perform.
        """
        super().__init__(env)
        self.expand_at_step = expand_at_step
        self.expand_at_episode = expand_at_episode
        self.new_action_name = new_action_name
        self.max_expansions = max_expansions

        self.original_n = int(env.action_space.n)  # type: ignore[union-attr]
        self.current_n = self.original_n
        self.step_count = 0
        self.expansion_count = 0
        self.has_expanded = False

        # Enable new action mapping: random walk on expansion step
        self._override_action = False

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """Reset the environment. The cumulative step counter persists
        across episode boundaries."""
        # Step counter is NOT reset here — it tracks cumulative steps
        # across episodes so that expansion thresholds are global.
        return self.env.reset(**kwargs)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Step the environment, checking for action-space expansion.

        Args:
            action: Action index. If the action space has expanded, the
                new action (index = current_n - 1) maps to action 0
                (no-op in many environments) to avoid crashes.

        Returns:
            Standard Gymnasium step tuple.
        """
        self.step_count += 1

        # ── Check for step-based expansion ──────────────────────────────
        if (not self.has_expanded
                and self.expansion_count < self.max_expansions
                and self.step_count >= self.expand_at_step):
            self._expand_action_space()

        # ── Map potentially out-of-range actions ────────────────────────
        # If action >= original_n but we haven't learned the semantics yet,
        # map it to a safe default (action 0).
        mapped_action = action
        if action >= self.original_n and not self.has_expanded:
            mapped_action = 0
        elif action >= self.env.action_space.n:
            mapped_action = action % self.env.action_space.n

        return self.env.step(mapped_action)

    def _expand_action_space(self) -> None:
        """Expand the action space by one action."""
        self.has_expanded = True
        self.expansion_count += 1

        # Deep-copy and modify the action space
        new_space = deepcopy(self.env.action_space)
        # Discrete spaces support: new_space.n = new_value
        # Use setattr for gymnasium Discrete spaces
        import gymnasium.spaces as spaces
        if isinstance(new_space, spaces.Discrete):
            new_n = self.current_n + 1
            # Create a new Discrete space with the expanded size
            new_space = spaces.Discrete(new_n)
            self.env.action_space = new_space
            self.current_n = new_n
        else:
            raise TypeError(
                f"ExpandingActionWrapper only supports Discrete spaces, "
                f"got {type(new_space)}"
            )

        print(
            f"[env] Action space expanded from {self.current_n - 1} -> "
            f"{self.current_n} at step {self.step_count} "
            f"(expansion #{self.expansion_count})"
        )

    @property
    def action_space(self) -> gym.Space:
        """Return the (potentially expanded) action space."""
        return self.env.action_space

    @action_space.setter
    def action_space(self, space: gym.Space) -> None:
        """Allow external modification of the action space."""
        self.env.action_space = space
