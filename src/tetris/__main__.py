"""Simple ASCII demo for the Tetris engine.

Run with: `python -m tetris`

This module prints a single frame composed of the board plus the active
tetromino, useful as a minimal smoke test to ensure renderers see more than a
blank grid.
"""

from __future__ import annotations

from time import sleep

from . import GameState, render_grid


def _print_grid(grid: list[list[int]]) -> None:
    for row in grid:
        print("".join("#" if cell else "." for cell in row))


def main() -> None:
    gs = GameState()
    gs.reset_game()
    grid = render_grid(gs.board, gs.active)
    _print_grid(grid)


if __name__ == "__main__":
    main()

