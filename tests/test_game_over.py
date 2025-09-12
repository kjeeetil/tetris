import types
import sys

# Ensure source directory on path
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
from tetris.tetromino import Tetromino, TetrominoType


def test_clearing_top_rows_not_game_over():
    runner = Runner()
    runner.state = GameState()
    board = runner.state.board
    for row in range(4):
        board.grid[row] = [1] * board.width
        board.grid[row][0] = 0
    runner.state.upcoming = TetrominoType.I
    runner.state.active = Tetromino(TetrominoType.I, rotation=1, position=(0, 0))
    board_before = runner.state.board
    runner._lock_or_game_over()
    assert board_before is runner.state.board
    assert all(cell == 0 for cell in runner.state.board.grid[0])


def test_spawn_collision_triggers_game_over():
    runner = Runner()
    runner.state = GameState()
    board = runner.state.board
    board.grid[1][3] = 1
    runner.state.upcoming = TetrominoType.O
    runner.state.active = Tetromino(TetrominoType.I, position=(board.height - 1, 0))
    board_before = runner.state.board
    runner._lock_or_game_over()
    assert board_before is not runner.state.board
