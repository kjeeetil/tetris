from __future__ import annotations

import pytest

from tetris.board import Board
from tetris.features import FEATURE_NAMES, FEAT_DIM, features_from_grid


def test_tetris_well_positive_for_single_deep_well() -> None:
    board = Board()
    well_col = 4
    depth = 6
    for row in range(Board.height - depth, Board.height):
        for col in range(Board.width):
            if col == well_col:
                continue
            board.set_cell(row, col, 1)

    features = features_from_grid(board.grid, lines=0)
    assert len(features) == FEAT_DIM
    idx = FEATURE_NAMES.index("Tetris Well")
    expected = depth / Board.height
    assert features[idx] == pytest.approx(expected)
    assert features[idx] > 0
