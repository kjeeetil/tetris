import pytest

from tetris.board import Board
from tetris.tetromino import TetrominoType
from tetris.placement_env import _enumerate_placements


def test_unreachable_columns_excluded():
    board = Board()
    # block between spawn column (3) and column 0
    board.grid[0][2] = 1
    actions = _enumerate_placements(board, TetrominoType.I)
    cols = [a.column for a in actions]
    assert all(c >= 3 for c in cols), cols
