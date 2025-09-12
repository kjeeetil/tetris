"""Placement-level environment for Tetris.

This module exposes a light-weight, gym-like environment where each action is a
complete placement decision for the current tetromino: choose a rotation and a
target column, hard-drop to the resting position, lock, clear any full rows,
receive a reward based on cleared lines, then immediately spawn the next piece.

The environment does not enforce path feasibility (i.e., how a human would
translate/rotate the falling piece to reach the final position). This is
intentional for placement-level control, which dramatically shortens the
decision horizon and tends to learn faster.

Example usage
-------------

>>> from tetris.placement_env import PlacementEnv
>>> env = PlacementEnv()
>>> obs, info = env.reset(seed=0)
>>> done = False
>>> total = 0
>>> while not done:
...     # Pick a random valid placement
...     actions = info["action_list"]
...     if not actions:
...         break
...     import random
...     a = random.randrange(len(actions))
...     obs, reward, done, info = env.step(a)
...     total += reward
>>> total >= 0
True

Notes
-----
- Observation is a simple dict with the board occupancy and piece type indices.
- `info` contains a stable-size `action_mask` of length 40 (max 4 rotations *
  10 columns) and the concrete `action_list` mapping action indices to
  (rotation, column) placements valid in the current state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import random

from .board import Board
from .tetromino import Tetromino, TetrominoType, TETROMINO_SHAPES
from .utils import can_move


# Fixed upper bound on the number of placement actions: 4 rotations * 10 cols
MAX_PLACEMENT_ACTIONS = 40


@dataclass(frozen=True)
class Placement:
    rotation: int
    column: int


def _unique_rotation_indices(shape: TetrominoType) -> List[int]:
    """Return indices of unique rotation states for ``shape``.

    Some shapes (e.g. ``O``, ``I``, ``S``, ``Z``) have duplicate states among
    the four rotations. This function returns a minimal set of indices whose
    states are unique by block layout.
    """

    seen = set()
    unique: List[int] = []
    for idx, state in enumerate(TETROMINO_SHAPES[shape]):
        key = tuple(sorted(state))
        if key not in seen:
            seen.add(key)
            unique.append(idx)
    return unique


def _state_width(state: Sequence[Tuple[int, int]]) -> int:
    """Return the width (in columns) of a rotation state (min-corner normalised)."""

    max_dc = max(dc for _, dc in state)
    return max_dc + 1


def _drop_row(board: Board, tetromino: Tetromino) -> Optional[int]:
    """Return the final row the tetromino would rest on, or ``None`` if invalid.

    The tetromino's current rotation and column are honoured; its row is
    assumed to be set to 0 on entry.
    """

    if not can_move(board, tetromino, 0, 0):
        return None
    while can_move(board, tetromino, 0, 1):
        tetromino.move(0, 1)
    return tetromino.position[0]


def _enumerate_placements(board: Board, shape: TetrominoType) -> List[Placement]:
    """Enumerate all valid (rotation, column) placements for ``shape``.

    A placement is considered valid if the piece, placed at row 0 with the
    chosen rotation and column and then hard-dropped, does not collide and
    remains within bounds at rest.
    """

    actions: List[Placement] = []
    for rot in _unique_rotation_indices(shape):
        state = TETROMINO_SHAPES[shape][rot]
        width = _state_width(state)
        for col in range(0, board.width - width + 1):
            temp = Tetromino(shape, rotation=rot, position=(0, col))
            final_row = _drop_row(board, temp)
            if final_row is not None:
                actions.append(Placement(rotation=rot, column=col))
    return actions


class PlacementEnv:
    """Placement-level Tetris environment with score-based rewards.

    Key properties
    - Action: index into the current list of valid (rotation, column) placements.
    - Reward: ``reward_per_line * lines_cleared`` (default 100 per line). When
      the game is over (no valid placements), an optional ``top_out_penalty``
      is applied once.
    - Episode termination: when there are no valid placements for the active
      piece.
    - Observation: dictionary containing the binary occupancy grid and piece
      identifiers for active/upcoming/held.
    """

    def __init__(
        self,
        *,
        reward_per_line: int = 100,
        invalid_action_penalty: float = -1.0,
        top_out_penalty: float = -500.0,
        deterministic_bag: bool = False,
    ) -> None:
        self.reward_per_line = reward_per_line
        self.invalid_action_penalty = invalid_action_penalty
        self.top_out_penalty = top_out_penalty
        self._state_random = random.Random()
        self._state: Optional["GameState"] = None
        self._cached_actions: List[Placement] = []
        self._done = False
        self._deterministic_bag = deterministic_bag

    # Lazy import to avoid circulars at module import time
    @property
    def state(self):
        if self._state is None:
            from .game_state import GameState  # local import

            self._state = GameState()
        return self._state

    def seed(self, seed: Optional[int]) -> None:
        """Seed the environment RNG (affects piece order when using Python random)."""

        if seed is not None:
            self._state_random.seed(seed)

    def reset(self, *, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """Reset environment and return ``(observation, info)``.

        If ``seed`` is provided, the environment's RNG is seeded. The board is
        cleared and the first piece is spawned.
        """

        from .game_state import GameState  # local import

        self.seed(seed)
        self._state = GameState()
        self._state.reset_game()
        # Replace random choice if deterministic 7-bag is requested
        if self._deterministic_bag:
            self._install_7bag(self._state_random)
        self._done = False
        obs, info = self._observe()
        return obs, info

    def _install_7bag(self, rng: random.Random) -> None:
        """Monkey-patch GameState piece generation to use a deterministic 7-bag.

        This keeps training more stable. It affects only this environment's
        state instance.
        """

        from .game_state import GameState  # local import

        bag: List[TetrominoType] = []

        def _bag_choice(self: GameState) -> TetrominoType:  # type: ignore[override]
            nonlocal bag
            if not bag:
                bag = list(TetrominoType)
                rng.shuffle(bag)
            return bag.pop()

        # Patch only this instance
        assert self._state is not None
        self._state._random_type = _bag_choice.__get__(self._state, GameState)  # type: ignore[attr-defined]

    def _observe(self) -> Tuple[Dict, Dict]:
        assert self.state.active is not None
        # Binary occupancy grid (0/1)
        board_mask = [[1 if v else 0 for v in row] for row in self.state.board.grid]
        # Map piece types to stable indices 0..6
        types = list(TetrominoType)
        idx_map = {t: i for i, t in enumerate(types)}
        active_idx = idx_map[self.state.active.shape]
        upc_idx = idx_map[self.state.upcoming] if self.state.upcoming else -1
        held_idx = idx_map[self.state.held] if self.state.held else -1

        actions = _enumerate_placements(self.state.board, self.state.active.shape)
        self._cached_actions = actions
        mask = [False] * MAX_PLACEMENT_ACTIONS
        for i in range(min(len(actions), MAX_PLACEMENT_ACTIONS)):
            mask[i] = True

        obs = {
            "board": board_mask,
            "active_type": active_idx,
            "upcoming_type": upc_idx,
            "held_type": held_idx,
            "score": self.state.score,
        }
        info = {
            "action_mask": mask,
            "action_list": actions,
        }
        return obs, info

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """Apply placement action and return ``(obs, reward, done, info)``.

        Invalid action indices incur ``invalid_action_penalty`` and result in a
        no-op observation (i.e., the same state) unless the episode is already
        done.
        """

        if self._done:
            # Episode already terminated; return zero reward and done
            obs, info = self._observe()
            return obs, 0.0, True, info

        # Ensure action cache corresponds to current state
        if not self._cached_actions:
            _ = self._observe()

        if action < 0 or action >= len(self._cached_actions):
            # Invalid index -> penalty and keep state
            obs, info = self._observe()
            return obs, float(self.invalid_action_penalty), False, info

        placement = self._cached_actions[action]
        reward = self._apply_placement(placement)

        # Check termination: if next piece has no valid placements
        next_actions = _enumerate_placements(self.state.board, self.state.active.shape) if self.state.active else []
        self._done = len(next_actions) == 0
        if self._done and self.top_out_penalty:
            reward += float(self.top_out_penalty)

        obs, info = self._observe()
        return obs, float(reward), self._done, info

    def _apply_placement(self, placement: Placement) -> int:
        """Apply a placement, clear rows, update score and spawn next piece.

        Returns the reward (score delta) from this placement.
        """

        assert self.state.active is not None
        piece = self.state.active
        piece.rotation = placement.rotation
        piece.position = (0, placement.column)

        final_row = _drop_row(self.state.board, piece)
        if final_row is None:
            # Should not happen if action came from enumerate; treat as invalid
            return int(self.invalid_action_penalty)
        # Move piece to final resting row (column already set)
        cur_row, cur_col = piece.position
        piece.position = (final_row, cur_col)

        # Lock and clear
        self.state.board.lock_piece(piece)
        cleared = self.state.board.clear_full_rows()
        score_delta = cleared * self.reward_per_line
        self.state.score += score_delta

        # Spawn next piece
        self.state.spawn_tetromino()

        # Refresh action cache for the new piece
        self._cached_actions = _enumerate_placements(self.state.board, self.state.active.shape) if self.state.active else []
        return score_delta

    # Convenience helpers -------------------------------------------------

    def action_space_n(self) -> int:
        """Return the fixed upper bound on action indices (40)."""

        return MAX_PLACEMENT_ACTIONS

    def legal_actions(self) -> List[Placement]:
        """Return the list of valid placements for the current piece."""

        if not self._cached_actions:
            _ = self._observe()
        return list(self._cached_actions)

    def render_ascii(self) -> str:
        """Return a simple ASCII rendering of the current board (no active overlay)."""

        chars = {0: ".", 1: "#"}
        return "\n".join(
            "".join(chars[1 if v else 0] for v in row) for row in self.state.board.grid
        )

