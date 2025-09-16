"""Board representation for the Tetris playfield."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .tetromino import Tetromino, TetrominoType


# Dimensions of the standard Tetris board.
WIDTH = 10
HEIGHT = 20

Grid = NDArray[np.uint8]

# Mapping from ``TetrominoType`` to the integer stored in the grid.  The
# specific numeric values are not important as long as ``0`` represents an empty
# cell.
PIECE_VALUES = {t: i + 1 for i, t in enumerate(TetrominoType)}


def create_empty_grid() -> Grid:
    """Return a new empty board grid filled with zeros."""

    return np.zeros((HEIGHT, WIDTH), dtype=np.uint8)


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
            return int(self.grid[row, col])
        raise IndexError("Cell out of bounds")

    def set_cell(self, row: int, col: int, value: int) -> None:
        """Safely set the value at ``(row, col)``.

        Raises:
            IndexError: If the coordinates are outside the board.
        """
        if 0 <= row < self.height and 0 <= col < self.width:
            self.grid[row, col] = np.uint8(value)
        else:
            raise IndexError("Cell out of bounds")

    def is_empty(self, row: int, col: int) -> bool:
        """Return ``True`` if the cell at ``(row, col)`` is empty.

        Any coordinates outside the board are treated as occupied.  This makes
        collision detection simpler as off-board positions are automatically
        rejected.
        """

        if 0 <= row < self.height and 0 <= col < self.width:
            return bool(self.grid[row, col] == 0)
        return False

    def lock_piece(self, tetromino: Tetromino) -> None:
        """Lock the tetromino's blocks into the board grid."""

        coordinates = np.asarray(tetromino.blocks(), dtype=np.int16)
        if coordinates.size == 0:
            return

        rows, cols = coordinates.T
        if (
            np.any(rows < 0)
            or np.any(rows >= self.height)
            or np.any(cols < 0)
            or np.any(cols >= self.width)
        ):
            raise IndexError("Block out of bounds")

        value = np.uint8(PIECE_VALUES[tetromino.shape])
        self.grid[rows, cols] = value

    def clear_full_rows(self) -> int:
        """Clear completed rows and return how many were removed."""

        full_rows = np.all(self.grid != 0, axis=1)
        cleared = int(np.count_nonzero(full_rows))
        if cleared:
            remaining = self.grid[~full_rows]
            new_rows = np.zeros((cleared, self.width), dtype=self.grid.dtype)
            self.grid = np.vstack((new_rows, remaining))
        return cleared

