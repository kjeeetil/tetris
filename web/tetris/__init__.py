"""Placeholder interfaces for a simple Tetris implementation."""

from .board import Board
from .tetromino import Tetromino, TetrominoType, shape_blocks
from .game_state import GameState
from .utils import can_move

__all__ = [
    "Board",
    "Tetromino",
    "TetrominoType",
    "GameState",
    "can_move",
    "shape_blocks",
]
