"""Utility helpers for the Tetris engine."""

from __future__ import annotations

from .board import Board
from .tetromino import Tetromino


def can_move(board: Board, tetromino: Tetromino, dx: int, dy: int) -> bool:
    """Return ``True`` if ``tetromino`` can move by ``dx`` and ``dy`` on ``board``.

    The function checks that all blocks of the piece remain within the board's
    boundaries and do not collide with existing filled cells after the move.
    """

    for row, col in tetromino.blocks():
        new_row = row + dy
        new_col = col + dx
        if not board.is_empty(new_row, new_col):
            return False
    return True
