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

import os

import numpy as np
import torch
import matplotlib.pyplot as plt

try:  # Gymnasium preferred
    import gymnasium as gym
except Exception:
    import gym  # type: ignore

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# Allow running without installing the package (src-layout)
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from tetris.gym_env import TetrisPlacementGymEnv


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
        # Collect linear layers from the policy network
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
        plt.show(block=False)

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

def make_env():
    return TetrisPlacementGymEnv(
        include_action_mask=True,
        deterministic_bag=True,
        reward_per_line=100,
        invalid_action_penalty=-0.1,
        top_out_penalty=-10.0,
    )


def main():
    n_envs = int(os.environ.get("N_ENVS", "8"))
    total_timesteps = int(os.environ.get("TIMESTEPS", "200000"))

    vec_env = make_vec_env(make_env, n_envs=n_envs)
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=256,
        gamma=0.99,
        ent_coef=0.005,
        clip_range=0.2,
        tensorboard_log=os.environ.get("TB_LOGDIR"),
    )

    callback = WeightVisualizationCallback(update_freq=1000)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Quick evaluation run
    env = make_env()
    obs, info = env.reset(seed=0)
    done = False
    total = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total += reward
        done = terminated or truncated
    print(f"Eval total reward: {total}")


if __name__ == "__main__":
    main()
