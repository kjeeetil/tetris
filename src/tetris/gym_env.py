"""Gymnasium-compatible wrapper for placement-level Tetris.

Observation is a flat vector suitable for SB3 MlpPolicy by default.
It includes:
  - board mask (20x10=200)
  - active piece one-hot (7)
  - upcoming piece one-hot (7)
  - optional action mask (40)

Action space is Discrete(40). Only the first N entries are valid for the
current state; the rest are masked in the observation (and provided via
``info['action_mask']``). Invalid actions are penalised and treated as no-op.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

try:  # Gymnasium preferred
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # Fallback to classic gym
    import gym  # type: ignore
    from gym import spaces  # type: ignore

from .placement_env import PlacementEnv, MAX_PLACEMENT_ACTIONS
from .tetromino import TetrominoType


class TetrisPlacementGymEnv(gym.Env):
    metadata = {
        "render_modes": ["ansi"],
        "render_fps": 60,
    }

    def __init__(
        self,
        *,
        include_action_mask: bool = True,
        reward_per_line: int = 100,
        invalid_action_penalty: float = -0.1,
        top_out_penalty: float = -10.0,
        deterministic_bag: bool = True,
        step_penalty: float = 0.0,
        single_line_penalty: float = 0.0,
        max_steps: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._env = PlacementEnv(
            reward_per_line=reward_per_line,
            invalid_action_penalty=invalid_action_penalty,
            top_out_penalty=top_out_penalty,
            deterministic_bag=deterministic_bag,
            step_penalty=step_penalty,
            single_line_penalty=single_line_penalty,
        )
        self.include_action_mask = include_action_mask
        self.action_space = spaces.Discrete(MAX_PLACEMENT_ACTIONS)
        self._obs_size = 200 + 7 + 7 + (MAX_PLACEMENT_ACTIONS if include_action_mask else 0)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._obs_size,), dtype=np.float32
        )
        self._steps = 0
        self._max_steps = max_steps

    # ----------------------- Env API -----------------------
    def seed(self, seed: Optional[int] = None) -> None:
        self._env.seed(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.seed(seed)
        obs, info = self._env.reset(seed=seed)
        self._steps = 0
        return self._convert_obs(obs, info), self._convert_info(info)

    def step(self, action: int):
        obs, reward, done, info = self._env.step(action)
        self._steps += 1
        terminated = bool(done)
        truncated = False
        if self._max_steps is not None and self._steps >= self._max_steps:
            truncated = True
        return self._convert_obs(obs, info), float(reward), terminated, truncated, self._convert_info(info)

    def render(self):
        return self._env.render_ascii()

    def close(self):
        return None

    # -------------------- Helpers -------------------------
    def _convert_obs(self, obs: Dict, info: Dict) -> np.ndarray:
        board = np.array(obs["board"], dtype=np.float32)  # (20,10)
        board = board.reshape(-1)  # 200
        types = list(TetrominoType)
        active = int(obs["active_type"])  # 0..6
        upcoming = int(obs["upcoming_type"])  # 0..6 (always set in our engine)
        active_oh = np.zeros((7,), dtype=np.float32)
        upcoming_oh = np.zeros((7,), dtype=np.float32)
        if 0 <= active < 7:
            active_oh[active] = 1.0
        if 0 <= upcoming < 7:
            upcoming_oh[upcoming] = 1.0
        parts = [board, active_oh, upcoming_oh]
        if self.include_action_mask:
            mask = np.array(info.get("action_mask", [False] * MAX_PLACEMENT_ACTIONS), dtype=np.float32)
            parts.append(mask)
        out = np.concatenate(parts, dtype=np.float32)
        return out

    def _convert_info(self, info: Dict) -> Dict:
        return {
            "action_mask": np.array(info.get("action_mask", [False] * MAX_PLACEMENT_ACTIONS), dtype=bool)
        }

