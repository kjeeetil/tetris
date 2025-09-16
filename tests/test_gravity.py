import sys
import types

from tetris.utils import gravity_interval_ms


def test_gravity_speed_increases_with_level():
    assert gravity_interval_ms(1) < gravity_interval_ms(0)
    assert gravity_interval_ms(10) == 0


def test_level_10_drops_piece_immediately(monkeypatch):
    class DummyWindow:
        def requestAnimationFrame(self, _):
            return 0
        def cancelAnimationFrame(self, _):
            pass

    class DummyDocument:
        def getElementById(self, _):
            return None
        def addEventListener(self, *args, **kwargs):
            pass
        def createElement(self, *args, **kwargs):
            return types.SimpleNamespace(textContent=None, prepend=lambda x: None)

    sys.modules['js'] = types.SimpleNamespace(window=DummyWindow(), document=DummyDocument())
    sys.modules['pyodide'] = types.SimpleNamespace(ffi=types.SimpleNamespace(create_proxy=lambda f: f))

    from tetris import run_canvas
    from tetris.game_state import GameState

    runner = run_canvas.Runner()
    runner._draw = lambda: None
    runner._update_level = lambda: None
    runner.state = GameState()
    runner.state.reset_game()
    runner.running = True
    runner.state.level = 10
    active_before = runner.state.active
    pieces_before = runner.state.pieces
    runner._tick(500)
    assert runner.state.pieces == pieces_before + 1
    assert runner.state.active is not active_before
