import types
import sys

sys.path.append('src')

# Stub out browser-specific modules used by run_canvas
js = types.SimpleNamespace(
    document=types.SimpleNamespace(
        getElementById=lambda *_args, **_kwargs: None,
        createElement=lambda *_args, **_kwargs: types.SimpleNamespace(prepend=lambda _x: None),
    ),
    window=types.SimpleNamespace(
        requestAnimationFrame=lambda _f: None,
        cancelAnimationFrame=lambda *_args, **_kwargs: None,
    ),
)
sys.modules['js'] = js
ffi = types.SimpleNamespace(create_proxy=lambda f: f)
pyodide = types.SimpleNamespace(ffi=ffi)
sys.modules['pyodide'] = pyodide
sys.modules['pyodide.ffi'] = ffi

from tetris.run_canvas import Runner
from tetris.game_state import GameState


def test_tick_resets_after_exception():
    runner = Runner()
    runner.state = GameState()
    runner.state.reset_game()
    runner.running = True
    runner.last_ts = 10.0
    runner.drop_accum = 5.0
    runner.state.score = 42

    def boom():
        raise RuntimeError('boom')

    runner._draw = boom
    runner._tick(0.0)

    assert runner.last_ts == 0
    assert runner.drop_accum == 0
    assert runner.state.score == 0
