import sys
sys.path.append('src')

from tetris.game_state import GameState


def test_level_advances_every_20_pieces():
    state = GameState()
    for _ in range(40):
        state.piece_locked()
    assert state.pieces == 40
    assert state.level == 2


def test_reset_resets_counters():
    state = GameState()
    for _ in range(5):
        state.piece_locked()
    state.reset_game()
    assert state.pieces == 0
    assert state.level == 0
