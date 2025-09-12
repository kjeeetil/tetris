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
    held: Optional[TetrominoType] = None
    hold_used: bool = False
    score: int = 0
    level: int = 0
    pieces: int = 0

    def _random_type(self) -> TetrominoType:
        """Return a random tetromino type."""

        return random.choice(list(TetrominoType))

    def spawn_tetromino(self) -> Tetromino:
        """Spawn and return a new active tetromino.

        The piece in ``upcoming`` becomes active and a new upcoming piece is
        randomly selected.  The hold flag is reset so the player may hold again
        for the new piece.  The new piece spawns near the top centre of the
        board.
        """

        shape = self.upcoming or self._random_type()
        self.active = Tetromino(shape)
        self.active.position = (0, self.board.width // 2 - 2)
        self.upcoming = self._random_type()
        self.hold_used = False
        return self.active

    def swap_hold(self) -> None:
        """Swap the active piece with the held one.

        Implements the standard Tetris hold mechanic.  The swap may only happen
        once per spawned piece; additional calls are ignored until another piece
        is spawned.
        """

        if self.active is None or self.hold_used:
            return

        if self.held is None:
            self.held = self.active.shape
            self.spawn_tetromino()
        else:
            current = self.active.shape
            self.active = Tetromino(self.held)
            self.active.position = (0, self.board.width // 2 - 2)
            self.held = current

        self.hold_used = True

    def reset_game(self) -> None:
        """Reset the entire game state for a new game."""

        self.board = Board()
        self.board.grid = create_empty_grid()
        self.score = 0
        self.level = 0
        self.pieces = 0
        self.active = None
        self.upcoming = None
        self.held = None
        self.hold_used = False
        self.spawn_tetromino()

    def game_over(self, log_fn=None) -> None:
        """Log a game-over message and reset the game state.

        ``log_fn`` is an optional callable used for diagnostics.  Any
        exceptions raised during logging are ignored so that the game loop can
        continue unhindered.
        """

        if log_fn is not None:
            try:
                log_fn("Game over. Resetting.")
            except Exception:
                pass
        self.reset_game()

