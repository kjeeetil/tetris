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


def test_game_over_resets_timers():
    runner = Runner()
    runner.state = GameState()
    # Pretend some time has passed in the current game
    runner.last_ts = 123.4
    runner.drop_accum = 250.0
    runner._game_over()
    assert runner.last_ts == 0
    assert runner.drop_accum == 0
