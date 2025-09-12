"""High level game state container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import random

from .board import Board, create_empty_grid
from .tetromino import Tetromino, TetrominoType


@dataclass
class GameState:
    """Mutable state for a Tetris game session."""

    board: Board = field(default_factory=Board)
    active: Optional[Tetromino] = None
    upcoming: Optional[TetrominoType] = None
    score: int = 0

    def spawn_tetromino(self) -> Tetromino:
        """Spawn and return a new active tetromino.

        A random piece is chosen for the active slot while another random piece
        is stored as ``upcoming`` to emulate the standard preview behaviour.
        The new piece spawns near the top centre of the board.
        """

        shape = self.upcoming or random.choice(list(TetrominoType))
        self.active = Tetromino(shape, position=(0, self.board.width // 2 - 2))
        self.upcoming = random.choice(list(TetrominoType))
        return self.active

    def swap_hold(self) -> None:  # pragma: no cover - placeholder
        """Swap the active piece with the held one."""
        raise NotImplementedError

    def reset_game(self) -> None:
        """Reset the entire game state for a new game."""

        self.board.grid = create_empty_grid()
        self.score = 0
        self.active = None
        self.upcoming = None
        self.spawn_tetromino()
