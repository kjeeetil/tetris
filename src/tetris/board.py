"""Board representation for the Tetris playfield."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .tetromino import Tetromino, TetrominoType

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

    def is_empty(self, row: int, col: int) -> bool:
        """Return ``True`` if the cell at ``(row, col)`` is empty."""

        if 0 <= row < self.height and 0 <= col < self.width:
            return self.grid[row][col] == 0
        return False

    def lock_piece(self, tetromino: "Tetromino") -> None:
        """Lock all blocks of ``tetromino`` into the board grid."""

        from .tetromino import TetrominoType

        shape_id = list(TetrominoType).index(tetromino.shape) + 1
        for r, c in tetromino.blocks():
            if 0 <= r < self.height and 0 <= c < self.width:
                self.grid[r][c] = shape_id
            else:
                raise IndexError("Block out of bounds")

    def clear_full_rows(self) -> int:
        """Clear completed rows and return how many were removed."""

        new_grid: Grid = [row for row in self.grid if any(cell == 0 for cell in row)]
        cleared = self.height - len(new_grid)
        for _ in range(cleared):
            new_grid.insert(0, [0] * self.width)
        self.grid = new_grid
        return cleared
