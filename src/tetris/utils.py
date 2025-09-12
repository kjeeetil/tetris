"""Utility helpers for the Tetris engine."""

from __future__ import annotations

from .board import Board
from .tetromino import Tetromino


def can_move(board: Board, tetromino: Tetromino, dx: int, dy: int) -> bool:  # pragma: no cover - placeholder
    """Return True if the tetromino can move by ``dx`` and ``dy`` on ``board``."""
    raise NotImplementedError
