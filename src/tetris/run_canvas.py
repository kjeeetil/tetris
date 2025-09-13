"""Canvas-based web front-end for Tetris.

This renderer draws directly to the HTML5 canvas via PyScript/pyodide's JS
bridge. It supports start/pause/resume/stop and basic keyboard controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from js import document, window  # type: ignore
from pyodide.ffi import create_proxy  # type: ignore

from .board import Board
from .game_state import GameState
from .tetromino import TetrominoType
from .utils import can_move, gravity_interval_ms


CELL_SIZE = 30

SHAPE_COLORS = {
    TetrominoType.I: "#00ffff",
    TetrominoType.O: "#ffff00",
    TetrominoType.T: "#800080",
    TetrominoType.S: "#00ff00",
    TetrominoType.Z: "#ff0000",
    TetrominoType.J: "#0000ff",
    TetrominoType.L: "#ffa500",
}


@dataclass
class Runner:
    state: Optional[GameState] = None
    running: bool = False
    paused: bool = False
    last_ts: float = 0.0
    drop_accum: float = 0.0
    raf_handle: Optional[int] = None

    def _ctx(self):
        canvas = document.getElementById("canvas")
        return canvas.getContext("2d")

    def _log(self, msg: str) -> None:
        el = document.getElementById("diagnostics")
        if el:
            div = document.createElement("div")
            div.textContent = msg
            el.prepend(div)

    def _update_level(self) -> None:
        """Update the level readout in the DOM if present."""
        if not self.state:
            return
        el = document.getElementById("level")
        if el:
            el.textContent = f"Level: {self.state.level}"

    def _game_over(self) -> None:
        """Handle end of game by logging and resetting state."""
        self._log("Game over. Resetting.")
        if self.state:
            self.state.reset_game()
            self._update_level()
            # Reset timing accumulators so a new game starts cleanly.
            # Without clearing these values the animation loop would
            # continue using the old timestamps, causing the freshly
            # spawned piece to drop immediately or skip frames.
            self.last_ts = 0
            self.drop_accum = 0

    def _lock_or_game_over(self) -> None:
        """Lock the active piece or reset if the game has ended."""
        if not self.state or not self.state.active:
            self._game_over()
            return
        if not can_move(self.state.board, self.state.active, 0, 0):
            self._game_over()
            return
        self.state.board.lock_piece(self.state.active)
        self.state.board.clear_full_rows()
        self.state.piece_locked()
        self._update_level()
        if any(self.state.board.grid[0]):
            self._game_over()
            return
        self.state.spawn_tetromino()
        if not can_move(self.state.board, self.state.active, 0, 0):
            self._game_over()

    def _draw(self) -> None:
        ctx = self._ctx()
        w = Board.width * CELL_SIZE
        h = Board.height * CELL_SIZE
        # background
        ctx.fillStyle = "#000000"
        ctx.fillRect(0, 0, w, h)
        # grid and locked cells
        if not self.state:
            return
        for r in range(self.state.board.height):
            for c in range(self.state.board.width):
                val = self.state.board.grid[r][c]
                if val:
                    # Map integer back to a color roughly by shape order
                    # For simplicity draw locked blocks in grey
                    ctx.fillStyle = "#444444"
                    ctx.fillRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                ctx.strokeStyle = "#333333"
                ctx.strokeRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        # active piece
        if self.state.active:
            color = SHAPE_COLORS[self.state.active.shape]
            ctx.fillStyle = color
            for r, c in self.state.active.blocks():
                ctx.fillRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                ctx.strokeStyle = "#333333"
                ctx.strokeRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)

    def _tick(self, ts: float) -> None:
        if not self.running:
            return
        try:
            if self.last_ts == 0:
                self.last_ts = ts
            dt = ts - self.last_ts
            self.last_ts = ts
            if not self.paused and self.state:
                self.drop_accum += dt
                delay = gravity_interval_ms(self.state.level)
                if self.drop_accum >= delay:
                    self.drop_accum = 0
                    if delay <= 0:
                        if self.state.active:
                            while can_move(self.state.board, self.state.active, 0, 1):
                                self.state.active.move(0, 1)
                            # Lock the piece and spawn the next one, resetting if
                            # the game has reached a terminal state.
                            self._lock_or_game_over()
                    elif self.state.active and can_move(self.state.board, self.state.active, 0, 1):
                        self.state.active.move(0, 1)
                    else:
                        # Lock the piece and spawn the next one, resetting if the
                        # game has reached a terminal state.
                        self._lock_or_game_over()
            self._draw()
        except Exception as exc:  # pragma: no cover - defensive guard
            # If anything goes wrong during the animation tick, log the error
            # and reset the game so a fresh session can continue.
            self._log(f"Crash detected: {exc}")
            self._game_over()
        self.raf_handle = window.requestAnimationFrame(create_proxy(self._tick))

    def _on_key(self, evt) -> None:
        if not self.running or self.paused or not self.state or not self.state.active:
            return
        key = evt.key
        if key == "ArrowLeft" and can_move(self.state.board, self.state.active, -1, 0):
            self.state.active.move(-1, 0)
        elif key == "ArrowRight" and can_move(self.state.board, self.state.active, 1, 0):
            self.state.active.move(1, 0)
        elif key == "ArrowUp":
            self.state.active.rotate()
            if not can_move(self.state.board, self.state.active, 0, 0):
                self.state.active.rotate(-1)
        elif key == "ArrowDown" and can_move(self.state.board, self.state.active, 0, 1):
            self.state.active.move(0, 1)
        elif key == " ":  # Space
            while can_move(self.state.board, self.state.active, 0, 1):
                self.state.active.move(0, 1)
            self._lock_or_game_over()
        self._draw()

    def start(self) -> None:
        if self.running:
            self._log("Already running")
            return
        # Focus canvas for keyboard input
        canvas = document.getElementById("canvas")
        if canvas:
            canvas.focus()
        self.state = GameState()
        self.state.reset_game()
        self.running = True
        self.paused = False
        self.last_ts = 0
        self.drop_accum = 0
        self._update_level()
        document.addEventListener("keydown", create_proxy(self._on_key))
        self._draw()
        self.raf_handle = window.requestAnimationFrame(create_proxy(self._tick))
        self._log("Game started")

    def pause(self) -> None:
        if not self.running:
            self._log("Pause ignored: not running")
            return
        self.paused = True
        self._log("Paused")

    def resume(self) -> None:
        if not self.running:
            self._log("Resume ignored: not running")
            return
        self.paused = False
        self._log("Resumed")

    def stop(self) -> None:
        if not self.running:
            self._log("Stop ignored: not running")
            return
        self.running = False
        self.paused = False
        if self.raf_handle is not None:
            try:
                window.cancelAnimationFrame(self.raf_handle)
            except Exception:
                pass
            self.raf_handle = None
        self._log("Game stopped")


runner = Runner()


def start() -> None:
    runner.start()


def pause() -> None:
    runner.pause()


def resume() -> None:
    runner.resume()


def stop() -> None:
    runner.stop()

