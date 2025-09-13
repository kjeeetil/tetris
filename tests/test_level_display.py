import types
import sys
import importlib

sys.path.append('src')

# Setup minimal js/pyodide stubs similar to test_game_over
level_el = types.SimpleNamespace(textContent=None)
js = types.SimpleNamespace(
    document=types.SimpleNamespace(
        getElementById=lambda id: level_el if id == "level" else None,
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

import tetris.run_canvas as rc
importlib.reload(rc)
Runner = rc.Runner
from tetris.game_state import GameState


def test_update_level_writes_dom():
    runner = Runner()
    runner.state = GameState()
    runner.state.level = 3
    runner._update_level()
    assert level_el.textContent == "Level: 3"
