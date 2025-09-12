"""Simple pygame front-end for the Tetris engine.

This module provides a minimal playable version of Tetris using the small
engine implemented in the surrounding modules.  It is intentionally lightweight
and is meant purely as a demonstration of how the core objects can be glued
together with ``pygame`` for rendering and input.
"""

from __future__ import annotations

import os
import asyncio
import pygame

from .board import Board, PIECE_VALUES
from .game_state import GameState
from .tetromino import TetrominoType
from .utils import can_move

# Size of a single board cell in pixels
CELL_SIZE = 30
# Milliseconds between automatic downward moves
GRAVITY_MS = 500
# Frames per second to run the game loop at
FPS = 60

# Colours for each tetromino type
SHAPE_COLORS = {
    TetrominoType.I: (0, 255, 255),
    TetrominoType.O: (255, 255, 0),
    TetrominoType.T: (128, 0, 128),
    TetrominoType.S: (0, 255, 0),
    TetrominoType.Z: (255, 0, 0),
    TetrominoType.J: (0, 0, 255),
    TetrominoType.L: (255, 165, 0),
}

# Mapping from the integer stored in the board grid to a colour
CELL_COLORS = {0: (0, 0, 0)}
for shape, value in PIECE_VALUES.items():
    CELL_COLORS[value] = SHAPE_COLORS[shape]


_LOG_BUFFER: list[str] = []


def log(msg: str) -> None:
    """Append a message to diagnostics on the web page if available.

    In non-web environments this becomes a no-op but messages are stored in an
    internal buffer for potential debugging.
    """

    _LOG_BUFFER.append(msg)
    try:  # Only available when running under PyScript (browser)
        from js import document, Date  # type: ignore

        ts = Date().toLocaleTimeString()
        el = document.getElementById("diagnostics")
        if el:
            entry = document.createElement("div")
            entry.textContent = f"[{ts}] {msg}"
            el.prepend(entry)
    except Exception:
        # Silently ignore if `js` is unavailable
        pass


def draw_board(screen: pygame.Surface, board: Board) -> None:
    """Render the existing board grid."""

    for r in range(board.height):
        for c in range(board.width):
            value = board.grid[r][c]
            color = CELL_COLORS[value]
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (50, 50, 50), rect, 1)


def draw_tetromino(screen: pygame.Surface, state: GameState) -> None:
    """Render the currently active tetromino."""

    if not state.active:
        return
    color = SHAPE_COLORS[state.active.shape]
    for r, c in state.active.blocks():
        rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, (50, 50, 50), rect, 1)


def lock_and_continue(state: GameState) -> None:
    """Lock the active piece, clear rows and spawn the next piece."""

    if not state.active:
        return
    state.board.lock_piece(state.active)
    # If any blocks reach the top row after locking, it's game over.
    if any(cell != 0 for cell in state.board.grid[0]):
        try:
            log("Game over. Resetting.")
        except Exception:
            pass
        state.reset_game()
        return

    cleared = state.board.clear_full_rows()
    if cleared:
        per_line = 100 * (cleared if cleared > 1 else 1)
        state.score += cleared * per_line
        try:
            log(f"Cleared {cleared} row(s). Score: {state.score}")
        except Exception:
            pass

    state.spawn_tetromino()
    if not can_move(state.board, state.active, 0, 0):
        # Game over -> reset
        try:
            log("Game over. Resetting.")
        except Exception:
            pass
        state.reset_game()


def handle_key(event: pygame.event.Event, state: GameState) -> None:
    """Process keyboard events for piece movement."""

    if not state.active:
        return
    if event.key == pygame.K_LEFT and can_move(state.board, state.active, -1, 0):
        state.active.move(-1, 0)
    elif event.key == pygame.K_RIGHT and can_move(state.board, state.active, 1, 0):
        state.active.move(1, 0)
    elif event.key == pygame.K_UP:
        state.active.rotate()
        if not can_move(state.board, state.active, 0, 0):
            state.active.rotate(-1)
    elif event.key == pygame.K_DOWN and can_move(state.board, state.active, 0, 1):
        state.active.move(0, 1)
    elif event.key == pygame.K_SPACE:
        while can_move(state.board, state.active, 0, 1):
            state.active.move(0, 1)
        lock_and_continue(state)


class GameRunner:
    """Manage the game loop with start/pause/resume/stop controls."""

    def __init__(self) -> None:
        self._running = False
        self._paused = False
        self._task: asyncio.Task | None = None
        self._screen: pygame.Surface | None = None
        self._state: GameState | None = None
        self._clock: pygame.time.Clock | None = None
        self._drop_timer = 0

    @property
    def running(self) -> bool:
        return self._running

    @property
    def paused(self) -> bool:
        return self._paused

    async def _run_loop(self) -> None:
        # Ensure SDL/pygame binds to the visible canvas in the page when running on Web.
        os.environ.setdefault("SDL_HINT_EMSCRIPTEN_CANVAS_ELEMENT_ID", "#canvas")
        os.environ.setdefault("SDL_HINT_EMSCRIPTEN_KEYBOARD_ELEMENT", "#canvas")
        pygame.init()
        board_px = Board.width * CELL_SIZE
        board_py = Board.height * CELL_SIZE
        self._screen = pygame.display.set_mode((board_px, board_py))
        pygame.display.set_caption("Tetris")
        self._clock = pygame.time.Clock()

        self._state = GameState()
        self._state.reset_game()
        log("Game started")

        self._drop_timer = 0
        self._running = True
        while self._running:
            dt = self._clock.tick(FPS) if self._clock else 0
            # Even when paused, process events so the window remains responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                elif event.type == pygame.KEYDOWN and self._state and not self._paused:
                    handle_key(event, self._state)

            if not self._paused and self._state:
                self._drop_timer += dt
                if self._drop_timer >= GRAVITY_MS:
                    self._drop_timer = 0
                    if self._state.active and can_move(self._state.board, self._state.active, 0, 1):
                        self._state.active.move(0, 1)
                    else:
                        lock_and_continue(self._state)

            if self._screen and self._state:
                self._screen.fill((0, 0, 0))
                draw_board(self._screen, self._state.board)
                if not self._paused:
                    draw_tetromino(self._screen, self._state)
                else:
                    # Draw the piece but dim? For now, still draw it
                    draw_tetromino(self._screen, self._state)
                pygame.display.set_caption(
                    f"Tetris - {'Paused - ' if self._paused else ''}Score: {self._state.score}"
                )
                pygame.display.flip()

            # Yield to the browser/host event loop to keep UI responsive
            await asyncio.sleep(0)

        pygame.quit()
        log("Game stopped")

    def start(self) -> None:
        if self._task and not self._task.done():
            log("Game already running")
            return
        self._paused = False
        try:
            loop = asyncio.get_event_loop()
            self._task = loop.create_task(self._run_loop())
        except RuntimeError:
            # No running loop (e.g., plain Python); run synchronously
            asyncio.run(self._run_loop())

    def pause(self) -> None:
        if not self._running:
            log("Pause ignored: game not running")
            return
        self._paused = True
        log("Paused")

    def resume(self) -> None:
        if not self._running:
            log("Resume ignored: game not running")
            return
        self._paused = False
        log("Resumed")

    async def stop_async(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._task:
            try:
                await self._task
            except Exception:
                pass

    def stop(self) -> None:
        if not self._running:
            log("Stop ignored: game not running")
            return
        # Schedule graceful shutdown
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self.stop_async())
        except RuntimeError:
            # If no loop, just set flag; loop exits promptly
            self._running = False


# Module-level runner instance for convenience from PyScript
runner = GameRunner()


def start() -> None:
    runner.start()


def pause() -> None:
    runner.pause()


def resume() -> None:
    runner.resume()


def stop() -> None:
    runner.stop()


def main() -> None:
    """Backward-compatible entry to run immediately (desktop/local)."""
    runner.start()
    # If running in a normal Python environment, block until finished
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(runner.stop_async())
    except RuntimeError:
        # Already ran to completion
        pass


if __name__ == "__main__":  # pragma: no cover - manual execution only
    main()
