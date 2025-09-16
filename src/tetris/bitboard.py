"""Bitboard acceleration helpers for placement enumeration.

This module precomputes rotation metadata for every tetromino and exposes
vectorised helpers that operate on a compact bitboard representation of the
Tetris board.  ``_enumerate_placements`` in :mod:`placement_env` uses these
helpers to avoid repeatedly constructing :class:`Tetromino` instances and
executing Python-heavy collision loops during every call.

The public entry points are intentionally small:

``board_to_bitmask``
    Convert a :class:`~tetris.board.Board` instance into an array of row
    bitmasks (``uint16``).  Each bit represents a column of the board.

``board_key``
    Return an immutable tuple suitable for use as a cache key.

``drop_row``
    Compute the resting row for a rotated piece at a given column using
    vectorised bitwise checks.

``path_clear``
    Evaluate whether the gravity-aware path from spawn to a target placement is
    feasible.  The function mirrors ``placement_env._path_clear`` but performs
    all collision checks via the bitboard representation instead of calling
    ``can_move`` repeatedly.

The heavy lifting is performed in NumPy which keeps the critical inner loops in
compiled code.  This drastically reduces the call count observed when profiling
``PlacementEnv`` and allows upstream code to cache placement lists keyed by the
bitboard representation of the playfield.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .board import Board
from .tetromino import TETROMINO_SHAPES, TetrominoType
from .utils import gravity_interval_ms


BOARD_WIDTH = Board.width
BOARD_HEIGHT = Board.height
SPAWN_COLUMN = BOARD_WIDTH // 2 - 2


@dataclass(frozen=True)
class RotationInfo:
    """Cached information for a concrete tetromino rotation."""

    shape: TetrominoType
    rotation: int
    width: int
    height: int
    row_masks: np.ndarray
    shifted_masks: np.ndarray

    def __post_init__(self) -> None:  # pragma: no cover - defensive programming
        # Mark the arrays as read-only so accidental mutation raises immediately
        self.row_masks.setflags(write=False)
        self.shifted_masks.setflags(write=False)


def _state_row_masks(state: Sequence[Tuple[int, int]]) -> Tuple[int, np.ndarray]:
    """Return ``(width, row_masks)`` for ``state``."""

    max_r = max(dr for dr, _ in state)
    max_c = max(dc for _, dc in state)
    height = max_r + 1
    width = max_c + 1
    masks = np.zeros(height, dtype=np.uint16)
    for dr, dc in state:
        masks[dr] |= np.uint16(1 << dc)
    return width, masks


def _build_rotation_infos(shape: TetrominoType) -> List[RotationInfo]:
    infos: List[RotationInfo] = []
    states = TETROMINO_SHAPES[shape]
    for rot, state in enumerate(states):
        width, row_masks = _state_row_masks(state)
        height = row_masks.size
        max_shift = BOARD_WIDTH - width
        shifted = np.zeros((max_shift + 1, height), dtype=np.uint16)
        for shift in range(max_shift + 1):
            shifted[shift] = np.left_shift(row_masks, shift).astype(np.uint16, copy=False)
        infos.append(
            RotationInfo(
                shape=shape,
                rotation=rot,
                width=width,
                height=height,
                row_masks=row_masks,
                shifted_masks=shifted,
            )
        )
    return infos


# Rotation metadata for every shape/state.
ROTATION_INFOS: Dict[TetrominoType, List[RotationInfo]] = {
    shape: _build_rotation_infos(shape) for shape in TetrominoType
}


# Pre-compute the unique rotation indices per shape (i.e. deduplicate S/Z/I/O).
UNIQUE_ROTATIONS: Dict[TetrominoType, Tuple[int, ...]] = {}
for shape, infos in ROTATION_INFOS.items():
    seen: Dict[bytes, int] = {}
    unique: List[int] = []
    for info in infos:
        key = (info.row_masks.tobytes(), info.width, info.height)
        if key not in seen:
            seen[key] = info.rotation
            unique.append(info.rotation)
    UNIQUE_ROTATIONS[shape] = tuple(unique)


def board_to_bitmask(board: Board) -> np.ndarray:
    """Return a NumPy array of ``uint16`` row bitmasks for ``board``."""

    masks = np.zeros(BOARD_HEIGHT, dtype=np.uint16)
    for r, row in enumerate(board.grid):
        mask = 0
        for c, value in enumerate(row):
            if value:
                mask |= 1 << c
        masks[r] = mask
    return masks


def board_key(board_masks: np.ndarray) -> Tuple[int, ...]:
    """Return an immutable cache key for ``board_masks``."""

    return tuple(int(v) for v in np.asarray(board_masks, dtype=np.uint16))


def _collision_mask(board_masks: np.ndarray, info: RotationInfo, column: int) -> np.ndarray:
    """Return a view of the masked board rows for ``info`` at ``column``."""

    shifted = info.shifted_masks[column]
    windows = sliding_window_view(board_masks, info.height)
    return np.bitwise_and(windows, shifted)


def can_place(board_masks: np.ndarray, info: RotationInfo, row: int, column: int) -> bool:
    """Return ``True`` if the piece can occupy ``(row, column)``."""

    if column < 0 or column + info.width > BOARD_WIDTH:
        return False
    if row < 0 or row + info.height > BOARD_HEIGHT:
        return False

    window = board_masks[row : row + info.height]
    mask = info.shifted_masks[column]
    return bool(np.all(np.bitwise_and(window, mask) == 0))


def drop_row(board_masks: np.ndarray, info: RotationInfo, column: int) -> int | None:
    """Return the resting row for ``info`` placed at ``column`` or ``None``."""

    if not can_place(board_masks, info, 0, column):
        return None
    max_row = BOARD_HEIGHT - info.height
    collisions = _collision_mask(board_masks, info, column)
    blocked = np.where(np.any(collisions != 0, axis=1))[0]
    if blocked.size == 0:
        return max_row
    first_blocked = int(blocked[0])
    return max(0, first_blocked - 1)


def _rotate_steps(num_rotations: int, start: int, target: int) -> Iterable[int]:
    current = start
    while current != target:
        current = (current + 1) % num_rotations
        yield current


def path_clear(
    board_masks: np.ndarray,
    shape: TetrominoType,
    rotation: int,
    column: int,
    level: int,
    *,
    final_row: int | None = None,
) -> bool:
    """Return ``True`` if the spawn-to-placement path is feasible."""

    infos = ROTATION_INFOS[shape]
    num_rotations = len(infos)

    current_rot = 0
    piece_info = infos[current_rot]
    row = 0
    col = SPAWN_COLUMN

    if not can_place(board_masks, piece_info, row, col):
        return False

    target_rot = rotation % num_rotations
    for step in _rotate_steps(num_rotations, current_rot, target_rot):
        piece_info = infos[step]
        if not can_place(board_masks, piece_info, row, col):
            return False

    piece_info = infos[target_rot]
    if final_row is None:
        final_row = drop_row(board_masks, piece_info, column)
        if final_row is None:
            return False

    if column == col:
        while row < final_row:
            if not can_place(board_masks, piece_info, row + 1, col):
                return False
            row += 1
        return True

    gravity = gravity_interval_ms(level)
    if gravity <= 0:
        return False

    moves_per_row = max(1, int(gravity // HORIZONTAL_MOVE_INTERVAL_MS))

    target_col = column
    while col != target_col:
        if not can_place(board_masks, piece_info, row + 1, col):
            return False
        row += 1

        direction = 1 if target_col > col else -1
        steps = min(moves_per_row, abs(target_col - col))
        for _ in range(steps):
            new_col = col + direction
            if not can_place(board_masks, piece_info, row, new_col):
                return False
            col = new_col
            if col == target_col:
                break

    while row < final_row:
        if not can_place(board_masks, piece_info, row + 1, target_col):
            return False
        row += 1

    return True


# Horizontal move cadence matches :mod:`placement_env`.
HORIZONTAL_MOVE_INTERVAL_MS = 100


@lru_cache(maxsize=2048)
def cached_placements(
    board_state: Tuple[int, ...], shape: TetrominoType, level: int
) -> Tuple[Tuple[int, int, int], ...]:
    """Return cached placements for ``board_state``/``shape``/``level``.

    The cache stores primitive tuples instead of :class:`Placement` instances to
    remain independent from ``placement_env`` and to keep the cache payload
    small.  ``placement_env`` is responsible for wrapping these tuples when
    constructing the public action list.
    """

    board_masks = np.array(board_state, dtype=np.uint16)
    actions: List[Tuple[int, int, int]] = []
    for rot in UNIQUE_ROTATIONS[shape]:
        info = ROTATION_INFOS[shape][rot]
        max_col = BOARD_WIDTH - info.width
        for col in range(max_col + 1):
            final_row = drop_row(board_masks, info, col)
            if final_row is None:
                continue
            if path_clear(
                board_masks,
                shape,
                rot,
                col,
                level,
                final_row=final_row,
            ):
                actions.append((rot, col, final_row))
    return tuple(actions)

