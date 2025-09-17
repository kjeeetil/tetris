from __future__ import annotations

import pytest

from tetris.features import FEAT_DIM, features_from_grid
from tetris import HeadlessPlacement, SimplifiedHeadlessModel
from tetris.tetromino import TetrominoType


def test_headless_line_clear_scores_and_resets_bottom_row() -> None:
    model = SimplifiedHeadlessModel(deterministic_bag=True)
    model.reset(seed=0)

    # Pre-fill the bottom row except for the leftmost four cells so a horizontal
    # I piece completes the line.
    grid = [[0] * 10 for _ in range(20)]
    grid[-1] = [1] * 10
    for col in range(4):
        grid[-1][col] = 0
    model.set_board(grid)

    assert model.active_piece == TetrominoType.I

    placements = model.placements()
    target = next(
        p for p in placements if p.rotation == 0 and p.column == 0
    )  # rotation 0 is horizontal
    assert isinstance(target, HeadlessPlacement)

    info = model.apply(target)

    assert info["lines_cleared"] == 1
    assert info["score_delta"] == 100
    # Bottom row should now be empty because it was cleared.
    assert model.board_grid()[-1] == [0] * 10


def test_headless_reaches_level_ten_and_ends_session() -> None:
    model = SimplifiedHeadlessModel(deterministic_bag=True)
    model.reset(seed=123)

    model._pieces = 199  # prime counters so the next placement enters level 10
    model._level = 9

    placement = model.placements()[0]
    assert isinstance(placement, HeadlessPlacement)
    model.apply(placement)

    assert model.game_over is True
    assert model.level == 10
    assert model.active_piece is None


def test_headless_feature_vector_matches_reference() -> None:
    model = SimplifiedHeadlessModel(deterministic_bag=True)
    model.reset(seed=1)

    placement = model.placements()[0]
    assert isinstance(placement, HeadlessPlacement)
    model.apply(placement)

    features = model.feature_vector()
    assert len(features) == FEAT_DIM

    manual = features_from_grid(model.board_grid(), model.last_cleared)
    assert features == pytest.approx(manual)
