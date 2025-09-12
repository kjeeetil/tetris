"""High level game state container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .board import Board
from .tetromino import Tetromino, TetrominoType


@dataclass
class GameState:
    """Mutable state for a Tetris game session."""

    board: Board = field(default_factory=Board)
    active: Optional[Tetromino] = None
    upcoming: Optional[TetrominoType] = None
    score: int = 0

    def spawn_tetromino(self) -> Tetromino:  # pragma: no cover - placeholder
        """Spawn and return a new active tetromino."""
        raise NotImplementedError

    def swap_hold(self) -> None:  # pragma: no cover - placeholder
        """Swap the active piece with the held one."""
        raise NotImplementedError

    def reset_game(self) -> None:  # pragma: no cover - placeholder
        """Reset the entire game state for a new game."""
        raise NotImplementedError
