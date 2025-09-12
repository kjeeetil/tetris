"""Tetromino definitions and basic behaviour.

This module provides a very small subset of a Tetris implementation.  Only the
data structure representing a falling piece and a couple of operations on that
piece are implemented.  The rest of the game engine is intentionally left as
placeholders elsewhere in the code base.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

RotationState = List[Tuple[int, int]]


class TetrominoType(str, Enum):
    """Enumeration of the seven standard tetromino shapes."""

    I = "I"
    O = "O"
    T = "T"
    S = "S"
    Z = "Z"
    J = "J"
    L = "L"


def _rotate(state: RotationState) -> RotationState:
    """Return ``state`` rotated 90 degrees clockwise.

    The rotation is performed around the origin.  The resulting coordinates are
    normalised so that the minimum row and column are zero.  This makes the
    returned state suitable for use as a set of relative offsets from a piece's
    position.
    """

    rotated = [(c, -r) for r, c in state]
    min_r = min(r for r, _ in rotated)
    min_c = min(c for _, c in rotated)
    return [(r - min_r, c - min_c) for r, c in rotated]


def _generate_rotations(state: RotationState) -> List[RotationState]:
    """Generate the four rotation states for a piece starting from ``state``."""

    rotations = [state]
    for _ in range(3):
        state = _rotate(state)
        rotations.append(state)
    return rotations


# Basic shapes for each tetromino in their spawn orientation.  The remaining
# rotation states are derived automatically via ``_generate_rotations``.
_BASE_SHAPES: Dict[TetrominoType, RotationState] = {
    TetrominoType.I: [(0, 0), (0, 1), (0, 2), (0, 3)],
    TetrominoType.O: [(0, 0), (0, 1), (1, 0), (1, 1)],
    TetrominoType.T: [(0, 0), (0, 1), (0, 2), (1, 1)],
    TetrominoType.S: [(0, 1), (0, 2), (1, 0), (1, 1)],
    TetrominoType.Z: [(0, 0), (0, 1), (1, 1), (1, 2)],
    TetrominoType.J: [(0, 0), (1, 0), (1, 1), (1, 2)],
    TetrominoType.L: [(0, 2), (1, 0), (1, 1), (1, 2)],
}


TETROMINO_SHAPES: Dict[TetrominoType, List[RotationState]] = {
    t_type: _generate_rotations(shape) for t_type, shape in _BASE_SHAPES.items()
}


def shape_blocks(shape: TetrominoType, rotation: int) -> RotationState:
    """Return the block offsets for ``shape`` at ``rotation``.

    Parameters
    ----------
    shape:
        The :class:`TetrominoType` to query.
    rotation:
        Index of the desired rotation state.  Values are wrapped so any integer
        is accepted.
    """

    states = TETROMINO_SHAPES[shape]
    return states[rotation % len(states)]


@dataclass
class Tetromino:
    """Active falling piece in the game."""

    shape: TetrominoType
    rotation: int = 0
    position: Tuple[int, int] = (0, 0)  # (row, col)

    def rotate(self, direction: int = 1) -> None:
        """Rotate the piece.

        Parameters
        ----------
        direction:
            Positive values rotate clockwise whilst negative values rotate
            counter-clockwise.  Only the sign of ``direction`` matters; the
            rotation wraps around the number of available states.
        """

        states = TETROMINO_SHAPES[self.shape]
        self.rotation = (self.rotation + direction) % len(states)

    def move(self, dx: int, dy: int) -> None:
        """Move the piece by the given offsets.

        ``dx`` moves horizontally (columns) and ``dy`` moves vertically
        (rows).  The piece's position is stored as ``(row, col)``.
        """

        row, col = self.position
        self.position = (row + dy, col + dx)

    def blocks(self) -> List[Tuple[int, int]]:
        """Return the global block coordinates for this piece."""

        row, col = self.position
        state = shape_blocks(self.shape, self.rotation)
        return [(row + dr, col + dc) for dr, dc in state]

