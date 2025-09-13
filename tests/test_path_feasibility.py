import pytest

from tetris.board import Board
from tetris.tetromino import TetrominoType
from tetris.placement_env import _enumerate_placements


def test_unreachable_columns_excluded():
    board = Board()
    # block between spawn column (3) and column 0
    board.grid[1][2] = 1
    actions = _enumerate_placements(board, TetrominoType.I, level=0)
    cols = [a.column for a in actions]
    assert all(c >= 3 for c in cols), cols


def test_deep_obstacle_blocks_path():
    board = Board()
    # At high level only one horizontal move is allowed per row. Place an
    # obstacle a few rows below the spawn to ensure the path check accounts for
    # bricks already on the board.
    board.grid[2][2] = 1
    actions = _enumerate_placements(board, TetrominoType.I, level=20)
    cols = [a.column for a in actions]
    assert all(c >= 2 for c in cols), cols


def test_level_29_no_horizontal_movement():
    board = Board()
    actions = _enumerate_placements(board, TetrominoType.I, level=29)
    cols = {a.column for a in actions}
    assert cols == {board.width // 2 - 2}
