"""Utility helpers for the Tetris engine."""

from __future__ import annotations

from .board import Board
from .tetromino import Tetromino


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
