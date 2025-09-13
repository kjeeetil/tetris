"""Utility helpers for the Tetris engine."""

from __future__ import annotations

from typing import Optional, List

from .board import Board, PIECE_VALUES
from .tetromino import Tetromino


BASE_GRAVITY_MS = 500


def gravity_interval_ms(level: int) -> float:
    """Return the fall interval in milliseconds for ``level``.

    The interval decreases as the level rises, speeding up the falling
    pieces.  For level ``29`` and above the function returns ``0`` to
    indicate that the active piece should drop to the bottom immediately.
    """

    if level >= 29:
        return 0.0
    # Exponentially decrease the delay but keep a practical lower bound
    return max(20.0, BASE_GRAVITY_MS * (0.85 ** level))


def can_move(board: Board, tetromino: Tetromino, dx: int, dy: int) -> bool:
    """Return ``True`` if ``tetromino`` can move by ``dx`` and ``dy`` on ``board``.

    The function checks that translating the piece by the provided offsets would
    keep all of its blocks within the board's boundaries and that none of the
    destination cells are already occupied.  It is intended for use within the
    game loop to validate both movement and rotation attempts before they are
    applied.
    """

    for row, col in tetromino.blocks():
        new_row = row + dy
        new_col = col + dx
        if not (0 <= new_row < board.height and 0 <= new_col < board.width):
            return False
        if not board.is_empty(new_row, new_col):
            return False
    return True


def render_grid(board: Board, active: Optional[Tetromino] = None) -> List[List[int]]:
    """Return a copy of the board grid with the active piece overlaid.

    This is a convenience for renderers that want a single 2D array to draw
    without mutating the underlying board state (i.e. without locking the
    piece). Cells occupied by the active piece receive the mapped integer
    value for the piece's shape.
    """

    grid = [row[:] for row in board.grid]
    if active is not None:
        for r, c in active.blocks():
            if 0 <= r < board.height and 0 <= c < board.width:
                grid[r][c] = PIECE_VALUES[active.shape]
    return grid
