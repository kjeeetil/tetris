"""Board representation for the Tetris playfield."""

from __future__ import annotations

from typing import List

from .tetromino import Tetromino, TetrominoType


# Dimensions of the standard Tetris board.
WIDTH = 10
HEIGHT = 20

Grid = List[List[int]]

# Mapping from ``TetrominoType`` to the integer stored in the grid.  The
# specific numeric values are not important as long as ``0`` represents an empty
# cell.
PIECE_VALUES = {t: i + 1 for i, t in enumerate(TetrominoType)}


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
        """Return ``True`` if the cell at ``(row, col)`` is empty.

        Any coordinates outside the board are treated as occupied.  This makes
        collision detection simpler as off-board positions are automatically
        rejected.
        """

        if 0 <= row < self.height and 0 <= col < self.width:
            return self.grid[row][col] == 0
        return False

    def lock_piece(self, tetromino: Tetromino) -> None:
        """Lock the tetromino's blocks into the board grid.

        Each block of the piece is written to the grid using a small integer
        representing the piece's shape.  Values greater than zero represent
        filled cells whilst ``0`` means empty.
        """

        for row, col in tetromino.blocks():
            if 0 <= row < self.height and 0 <= col < self.width:
                self.grid[row][col] = PIECE_VALUES[tetromino.shape]

    def clear_full_rows(self) -> int:
        """Remove any completely filled rows and return the number cleared."""

        new_grid = [row for row in self.grid if any(cell == 0 for cell in row)]
        cleared = self.height - len(new_grid)
        while len(new_grid) < self.height:
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
