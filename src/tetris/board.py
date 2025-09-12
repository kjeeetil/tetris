"""Board representation for the Tetris playfield."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .tetromino import Tetromino

# Dimensions of the standard Tetris board.
WIDTH = 10
HEIGHT = 20

Grid = List[List[int]]


def create_empty_grid() -> Grid:
    """Return a new empty board grid filled with zeros."""
    return [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]


class Board:
    """Tetris board holding the occupied cells."""

    width: int = WIDTH
    height: int = HEIGHT

    def __init__(self) -> None:
        self.grid: Grid = create_empty_grid()

    def get_cell(self, row: int, col: int) -> int:
        """Safely return the value at ``(row, col)``.

        Raises:
            IndexError: If the coordinates are outside the board.
        """
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.grid[row][col]
        raise IndexError("Cell out of bounds")

    def set_cell(self, row: int, col: int, value: int) -> None:
        """Safely set the value at ``(row, col)``.

        Raises:
            IndexError: If the coordinates are outside the board.
        """
        if 0 <= row < self.height and 0 <= col < self.width:
            self.grid[row][col] = value
        else:
            raise IndexError("Cell out of bounds")

    def is_empty(self, row: int, col: int) -> bool:  # pragma: no cover - placeholder
        """Check if the given cell is empty."""
        raise NotImplementedError

    def lock_piece(
        self, tetromino: Tetromino
    ) -> None:  # pragma: no cover - placeholder
        """Lock the tetromino's blocks into the board."""
        raise NotImplementedError

    def clear_full_rows(self) -> int:  # pragma: no cover - placeholder
        """Remove completed rows and return the number cleared."""
        raise NotImplementedError
