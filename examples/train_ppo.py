"""Minimal PPO training script for the Gymnasium-compatible Tetris env.

Usage:
    pip install gymnasium stable-baselines3 torch numpy
    python -m examples.train_ppo

Notes:
    - The environment uses a placement-level action space (Discrete(40)).
    - Observations are a flat vector (board mask + one-hots + action mask).
    - Action mask is included in the observation vector to help the policy avoid invalid actions.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib.pyplot as plt

try:  # Gymnasium preferred
    import gymnasium as gym
except Exception:  # pragma: no cover
    import gym  # type: ignore

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# Allow running without installing the package (src-layout)
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from tetris.gym_env import TetrisPlacementGymEnv


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EnvConfig:
    include_action_mask: bool = True
    deterministic_bag: bool = True
    reward_per_line: int = 100
    invalid_action_penalty: float = -0.1
    top_out_penalty: float = -10.0
    step_penalty: float = -1.0
    single_line_penalty: float = -10.0


@dataclass
class PPOConfig:
    n_envs: int = 8
    total_timesteps: int = 200_000
    learning_rate: float = 2.5e-4
    n_steps: int = 128
    batch_size: int = 256
    gamma: float = 0.99
    ent_coef: float = 0.005
    clip_range: float = 0.2
    tensorboard_log: str | None = None


# ---------------------------------------------------------------------------
# Visualisation callback
# ---------------------------------------------------------------------------


class WeightVisualizationCallback(BaseCallback):
    """Display policy network weights with a colour map during training."""

    def __init__(self, update_freq: int = 1000):
        super().__init__()
        self.update_freq = update_freq
        self.layers: list[torch.nn.Linear] = []
        self.images: list[plt.AxesImage] = []
        self.fig: plt.Figure | None = None

    def _on_training_start(self) -> None:
        plt.ion()
        self.layers = [
            module
            for module in self.model.policy.mlp_extractor.policy_net
            if isinstance(module, torch.nn.Linear)
        ]
        self.fig, axes = plt.subplots(1, len(self.layers), figsize=(4 * len(self.layers), 4))
        if len(self.layers) == 1:
            axes = [axes]
        for ax, layer in zip(axes, self.layers):
            weights = layer.weight.detach().cpu().numpy()
            max_abs = np.abs(weights).max()
            im = ax.imshow(
                weights,
                aspect="auto",
                cmap="coolwarm",
                vmin=-max_abs,
                vmax=max_abs,
            )
            ax.set_title(f"{layer.in_features}→{layer.out_features}")
            ax.set_xlabel("Out")
            ax.set_ylabel("In")
            self.images.append(im)
        plt.tight_layout()
        assert self.fig is not None
        self.fig.canvas.draw()
        plt.show(block=False)
        plt.pause(0.001)

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            for im, layer in zip(self.images, self.layers):
                weights = layer.weight.detach().cpu().numpy()
                max_abs = np.abs(weights).max()
                im.set_data(weights)
                im.set_clim(-max_abs, max_abs)
            assert self.fig is not None
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_env(cfg: EnvConfig) -> TetrisPlacementGymEnv:
    """Construct a configured Tetris environment."""

    return TetrisPlacementGymEnv(
        include_action_mask=cfg.include_action_mask,
        deterministic_bag=cfg.deterministic_bag,
        reward_per_line=cfg.reward_per_line,
        invalid_action_penalty=cfg.invalid_action_penalty,
        top_out_penalty=cfg.top_out_penalty,
        step_penalty=cfg.step_penalty,
        single_line_penalty=cfg.single_line_penalty,
    )


def train(cfg: PPOConfig, env_cfg: EnvConfig) -> PPO:
    """Train a PPO agent and return the fitted model."""

    env_fn = lambda: make_env(env_cfg)
    vec_env = make_vec_env(env_fn, n_envs=cfg.n_envs)
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        ent_coef=cfg.ent_coef,
        clip_range=cfg.clip_range,
        tensorboard_log=cfg.tensorboard_log,
    )
    callback = WeightVisualizationCallback(update_freq=1000)
    model.learn(total_timesteps=cfg.total_timesteps, callback=callback)
    return model


def evaluate(model: PPO, env_cfg: EnvConfig) -> float:
    """Run a single evaluation episode and return the reward."""

    env = make_env(env_cfg)
    obs, info = env.reset(seed=0)
    done = False
    total = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total += reward
        done = terminated or truncated
    return float(total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_cfg = EnvConfig()
    cfg = PPOConfig(n_envs=args.n_envs, total_timesteps=args.timesteps,
                    tensorboard_log=os.environ.get("TB_LOGDIR"))
    model = train(cfg, env_cfg)
    total = evaluate(model, env_cfg)
    print(f"Eval total reward: {total}")


if __name__ == "__main__":
    main()

