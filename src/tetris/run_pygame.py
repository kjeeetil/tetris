"""Simple pygame front-end for the Tetris engine.

This module provides a minimal playable version of Tetris using the small
engine implemented in the surrounding modules.  It is intentionally lightweight
and is meant purely as a demonstration of how the core objects can be glued
together with ``pygame`` for rendering and input.
"""

from __future__ import annotations

import os
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
    cleared = state.board.clear_full_rows()
    if cleared:
        state.score += cleared * 100
    state.spawn_tetromino()
    if not can_move(state.board, state.active, 0, 0):
        # Game over -> reset
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


def main() -> None:
    # Ensure SDL/pygame binds to the visible canvas in the page when running on Web.
    os.environ.setdefault("SDL_HINT_EMSCRIPTEN_CANVAS_ELEMENT_ID", "#canvas")
    os.environ.setdefault("SDL_HINT_EMSCRIPTEN_KEYBOARD_ELEMENT", "#canvas")
    pygame.init()
    board_px = Board.width * CELL_SIZE
    board_py = Board.height * CELL_SIZE
    screen = pygame.display.set_mode((board_px, board_py))
    pygame.display.set_caption("Tetris")
    clock = pygame.time.Clock()

    state = GameState()
    state.reset_game()

    drop_timer = 0
    running = True
    while running:
        dt = clock.tick(FPS)
        drop_timer += dt
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                handle_key(event, state)

        if drop_timer >= GRAVITY_MS:
            drop_timer = 0
            if state.active and can_move(state.board, state.active, 0, 1):
                state.active.move(0, 1)
            else:
                lock_and_continue(state)

        screen.fill((0, 0, 0))
        draw_board(screen, state.board)
        draw_tetromino(screen, state)
        pygame.display.set_caption(f"Tetris - Score: {state.score}")
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":  # pragma: no cover - manual execution only
    main()
