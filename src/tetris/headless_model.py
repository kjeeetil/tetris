"""Simplified placement-level engine optimised for headless training.

The :class:`SimplifiedHeadlessModel` mirrors the browser trainer's inputs but
collapses gameplay into placement-level decisions.  The engine keeps only the
board occupancy, the active/upcoming tetromino identifiers and a compact
feasibility test that derives landing rows from cached column heights.  Spawn
collisions and gravity timing are intentionally ignored – the focus is on fast
evaluation of candidate placements during headless learning runs.

Key differences from :mod:`tetris.placement_env`:

* Feasibility ignores spawn/path collisions.  Placements are deemed valid when
  the tetromino can occupy the final resting cells without overlapping locked
  blocks or leaving the board.
* Level timing is discarded; the session terminates as soon as the level would
  increase past nine (i.e. on the transition to level ten).
* Board updates run directly on bitmasks to minimise Python overhead, while the
  public API still exposes helpers to fetch grids and engineered feature
  vectors for compatibility with the existing TF.js MLP policy.

The module is intentionally self-contained so that unit tests can compare the
headless engine against the richer simulation without introducing additional
dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import random

from .board import Board
from .features import FEAT_DIM, features_from_grid
from .tetromino import TETROMINO_SHAPES, TetrominoType


FULL_ROW_MASK = (1 << Board.width) - 1


@dataclass(frozen=True)
class RotationProfile:
    """Static information describing one tetromino rotation."""

    rotation: int
    width: int
    height: int
    bottom_offsets: tuple[int, ...]
    row_masks: tuple[int, ...]
    column_mask: tuple[bool, ...]


@dataclass(frozen=True)
class HeadlessPlacement:
    """Placement decision for the simplified headless model."""

    rotation: int
    column: int
    row: int


def _build_rotation_profiles() -> tuple[
    Dict[TetrominoType, tuple[RotationProfile, ...]],
    Dict[TetrominoType, Dict[int, RotationProfile]],
]:
    """Pre-compute rotation metadata for every tetromino shape."""

    profiles: Dict[TetrominoType, List[RotationProfile]] = {}
    lookup: Dict[TetrominoType, Dict[int, RotationProfile]] = {}
    for shape, states in TETROMINO_SHAPES.items():
        shape_profiles: List[RotationProfile] = []
        seen: set[tuple[tuple[int, int], ...]] = set()
        for rotation, cells in enumerate(states):
            key = tuple(sorted(cells))
            if key in seen:
                continue
            seen.add(key)

            width = max(dc for _, dc in cells) + 1
            height = max(dr for dr, _ in cells) + 1

            column_mask = [False] * width
            for _, dc in cells:
                column_mask[dc] = True

            bottom_offsets = [0] * width
            for local_col in range(width):
                offsets = [dr for dr, dc in cells if dc == local_col]
                bottom_offsets[local_col] = max(offsets) if offsets else 0

            row_masks: List[int] = []
            for local_row in range(height):
                mask = 0
                for dr, dc in cells:
                    if dr == local_row:
                        mask |= 1 << dc
                row_masks.append(mask)

            profile = RotationProfile(
                rotation=rotation,
                width=width,
                height=height,
                bottom_offsets=tuple(bottom_offsets),
                row_masks=tuple(row_masks),
                column_mask=tuple(column_mask),
            )
            shape_profiles.append(profile)
        profiles[shape] = tuple(shape_profiles)
        lookup[shape] = {profile.rotation: profile for profile in shape_profiles}
    return profiles, lookup


ROTATION_PROFILES, ROTATION_LOOKUP = _build_rotation_profiles()


class SimplifiedHeadlessModel:
    """Light-weight placement-level engine tailored for headless training."""

    def __init__(
        self,
        *,
        reward_per_line: int = 100,
        deterministic_bag: bool = False,
    ) -> None:
        self.reward_per_line = reward_per_line
        self._deterministic_bag = deterministic_bag
        self._rng = random.Random()
        self._bag: List[TetrominoType] = []
        self._queue: List[TetrominoType] = []
        self._rows: List[int] = [0] * Board.height
        self._column_heights: List[int] = [0] * Board.width
        self._active: Optional[TetrominoType] = None
        self._cached_actions: Optional[List[HeadlessPlacement]] = None
        self._score = 0
        self._pieces = 0
        self._level = 0
        self._last_cleared = 0
        self._last_score_delta = 0
        self._game_over = False

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None) -> None:
        """Reset the engine to an empty board and spawn the first piece."""

        if seed is not None:
            self._rng.seed(seed)
        self._rows = [0] * Board.height
        self._column_heights = [0] * Board.width
        self._bag = []
        self._queue = []
        self._score = 0
        self._pieces = 0
        self._level = 0
        self._last_cleared = 0
        self._last_score_delta = 0
        self._game_over = False
        self._active = self._draw_piece()
        self._queue.append(self._draw_piece())
        self._cached_actions = None

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    @property
    def game_over(self) -> bool:
        return self._game_over

    @property
    def score(self) -> int:
        return self._score

    @property
    def level(self) -> int:
        return self._level

    @property
    def pieces(self) -> int:
        return self._pieces

    @property
    def active_piece(self) -> Optional[TetrominoType]:
        return self._active

    @property
    def upcoming_piece(self) -> Optional[TetrominoType]:
        return self._queue[0] if self._queue else None

    @property
    def last_cleared(self) -> int:
        return self._last_cleared

    @property
    def last_score_delta(self) -> int:
        return self._last_score_delta

    def board_grid(self) -> List[List[int]]:
        """Return a copy of the board as a 0/1 occupancy grid."""

        grid: List[List[int]] = []
        for mask in self._rows:
            row = [(mask >> col) & 1 for col in range(Board.width)]
            grid.append(row)
        return grid

    def feature_vector(self) -> List[float]:
        """Return the engineered feature vector for the current board."""

        feats = features_from_grid(self.board_grid(), self._last_cleared)
        if len(feats) != FEAT_DIM:
            raise AssertionError("Feature dimension mismatch")
        return feats

    # ------------------------------------------------------------------
    # Board helpers
    # ------------------------------------------------------------------
    def placements(self) -> List[HeadlessPlacement]:
        """Return all valid placements for the active piece."""

        if self._game_over or self._active is None:
            return []
        if self._cached_actions is None:
            self._cached_actions = self._enumerate_active()
        return list(self._cached_actions)

    def apply(self, placement: HeadlessPlacement) -> Dict[str, int | bool]:
        """Lock ``placement`` into the board and spawn the next piece."""

        if self._game_over or self._active is None:
            raise RuntimeError("Cannot apply placement when the game is over")

        profile = ROTATION_LOOKUP[self._active].get(placement.rotation)
        if profile is None:
            raise ValueError("Rotation index not valid for active piece")

        if not self._fits(profile, placement.row, placement.column):
            raise ValueError("Placement does not fit on the board")

        # Lock blocks
        for local_row, mask in enumerate(profile.row_masks):
            board_row = placement.row + local_row
            shifted = mask << placement.column
            self._rows[board_row] |= shifted

        # Update column heights only for the columns touched by the piece
        for local_col, has_block in enumerate(profile.column_mask):
            if not has_block:
                continue
            board_col = placement.column + local_col
            self._column_heights[board_col] = self._column_height(board_col)

        # Clear full rows
        cleared = self._clear_full_rows()
        multiplier = cleared if cleared > 1 else 1
        score_delta = cleared * self.reward_per_line * multiplier

        self._score += score_delta
        self._pieces += 1
        if self._pieces % 20 == 0:
            self._level += 1
            if self._level >= 10:
                self._game_over = True

        self._last_cleared = cleared
        self._last_score_delta = score_delta

        if self._game_over:
            self._active = None
            self._queue.clear()
            self._cached_actions = []
        else:
            next_active = self._queue.pop(0) if self._queue else self._draw_piece()
            self._active = next_active
            self._queue.append(self._draw_piece())
            self._cached_actions = None
            if not self.placements():
                self._game_over = True
                self._active = None
                self._queue.clear()
                self._cached_actions = []

        return {
            "lines_cleared": cleared,
            "score_delta": score_delta,
            "level": self._level,
            "pieces": self._pieces,
            "game_over": self._game_over,
        }

    # ------------------------------------------------------------------
    # Testing conveniences
    # ------------------------------------------------------------------
    def set_board(self, grid: Sequence[Sequence[int]]) -> None:
        """Replace the board occupancy using a 0/1 grid.

        This helper exists for unit tests that need to craft specific board
        states.  The method recomputes column heights and invalidates any cached
        placement list.  Piece counters and scores remain untouched.
        """

        if len(grid) != Board.height:
            raise ValueError("Grid height mismatch")
        rows: List[int] = []
        for row in grid:
            if len(row) != Board.width:
                raise ValueError("Grid width mismatch")
            mask = 0
            for col, value in enumerate(row):
                if value:
                    mask |= 1 << col
            rows.append(mask)
        self._rows = rows
        self._recompute_column_heights()
        self._cached_actions = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _draw_piece(self) -> TetrominoType:
        if not self._bag:
            self._bag = list(TetrominoType)
            if not self._deterministic_bag:
                self._rng.shuffle(self._bag)
        if self._deterministic_bag:
            return self._bag.pop(0)
        return self._bag.pop()

    def _enumerate_active(self) -> List[HeadlessPlacement]:
        assert self._active is not None
        placements: List[HeadlessPlacement] = []
        for profile in ROTATION_PROFILES[self._active]:
            max_col = Board.width - profile.width
            for column in range(max_col + 1):
                row = self._landing_row(profile, column)
                if row is None:
                    continue
                placements.append(
                    HeadlessPlacement(rotation=profile.rotation, column=column, row=row)
                )
        return placements

    def _landing_row(self, profile: RotationProfile, column: int) -> Optional[int]:
        if column < 0 or column + profile.width > Board.width:
            return None

        candidate = Board.height - profile.height
        for local_col, has_block in enumerate(profile.column_mask):
            if not has_block:
                continue
            board_col = column + local_col
            col_height = self._column_heights[board_col]
            slot = Board.height - col_height - 1 - profile.bottom_offsets[local_col]
            if slot < candidate:
                candidate = slot

        if candidate < 0:
            return None

        row = candidate
        while row >= 0:
            if self._fits(profile, row, column):
                return row
            row -= 1
        return None

    def _fits(self, profile: RotationProfile, row: int, column: int) -> bool:
        if row < 0 or row + profile.height > Board.height:
            return False
        for local_row, mask in enumerate(profile.row_masks):
            board_row = row + local_row
            shifted = mask << column
            if self._rows[board_row] & shifted:
                return False
        return True

    def _column_height(self, column: int) -> int:
        for row, mask in enumerate(self._rows):
            if mask & (1 << column):
                return Board.height - row
        return 0

    def _recompute_column_heights(self) -> None:
        for col in range(Board.width):
            self._column_heights[col] = self._column_height(col)

    def _clear_full_rows(self) -> int:
        remaining = [row for row in self._rows if row != FULL_ROW_MASK]
        cleared = Board.height - len(remaining)
        if cleared:
            self._rows = [0] * cleared + remaining
            self._recompute_column_heights()
        return cleared


__all__ = ["SimplifiedHeadlessModel", "HeadlessPlacement"]

