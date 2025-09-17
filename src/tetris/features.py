"""Feature extraction helpers mirroring the web client implementation.

The functions in this module compute the same normalized feature vector used by
``web/index.html`` so that Python-side experiments and unit tests can reason
about the identical inputs as the browser trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .board import Board

WIDTH = Board.width
HEIGHT = Board.height

GridLike = Sequence[Sequence[int]]


@dataclass(frozen=True)
class WellMetrics:
    """Summary statistics describing wells on the board."""

    well_sum: int
    edge_well: int
    max_well_depth: int
    well_count: int
    tetris_well: int


FEATURE_NAMES = [
    "Lines",
    "Lines²",
    "Single Clear",
    "Double Clear",
    "Triple Clear",
    "Tetris",
    "Holes",
    "Bumpiness",
    "Max Height",
    "Well Sum",
    "Edge Wells",
    "Tetris Well",
    "Contact",
    "Row Transitions",
    "Col Transitions",
    "Aggregate Height",
]
FEAT_DIM = len(FEATURE_NAMES)


def _cell_filled(grid: GridLike, row: int, col: int) -> bool:
    return bool(grid[row][col])


def column_heights(grid: GridLike) -> list[int]:
    heights = [0] * WIDTH
    for col in range(WIDTH):
        row = 0
        while row < HEIGHT and not _cell_filled(grid, row, col):
            row += 1
        heights[col] = HEIGHT - row
    return heights


def count_holes(grid: GridLike) -> int:
    holes = 0
    for col in range(WIDTH):
        seen_block = False
        for row in range(HEIGHT):
            if _cell_filled(grid, row, col):
                seen_block = True
            elif seen_block:
                holes += 1
    return holes


def bumpiness(heights: Sequence[int]) -> int:
    total = 0
    for col in range(WIDTH - 1):
        total += abs(heights[col] - heights[col + 1])
    return total


def well_metrics(heights: Sequence[int]) -> WellMetrics:
    well_sum = 0
    edge_well = 0
    max_depth = 0
    well_count = 0
    for col in range(WIDTH):
        left = heights[col - 1] if col > 0 else float("inf")
        right = heights[col + 1] if col < WIDTH - 1 else float("inf")
        min_neighbour = left if left < right else right
        depth = min_neighbour - heights[col]
        if depth > 0:
            well_sum += depth
            well_count += 1
            if depth > max_depth:
                max_depth = depth
        if col == 0:
            edge_well = max(edge_well, right - heights[0])
        if col == WIDTH - 1:
            edge_well = max(edge_well, left - heights[WIDTH - 1])
    tetris_well = max_depth if well_count == 1 else 0
    return WellMetrics(
        well_sum=well_sum,
        edge_well=max(0, edge_well),
        max_well_depth=max_depth,
        well_count=well_count,
        tetris_well=tetris_well,
    )


def contact_area(grid: GridLike) -> int:
    contact = 0
    for row in range(HEIGHT):
        for col in range(WIDTH):
            if not _cell_filled(grid, row, col):
                continue
            if row == HEIGHT - 1 or _cell_filled(grid, row + 1, col):
                contact += 1
            if col > 0 and _cell_filled(grid, row, col - 1):
                contact += 1
            if col < WIDTH - 1 and _cell_filled(grid, row, col + 1):
                contact += 1
    return contact


def row_transitions(grid: GridLike) -> int:
    transitions = 0
    for row in range(HEIGHT):
        prev = 0
        for col in range(WIDTH):
            cur = 1 if _cell_filled(grid, row, col) else 0
            if cur != prev:
                transitions += 1
                prev = cur
        if prev != 0:
            transitions += 1
    return transitions


def col_transitions(grid: GridLike) -> int:
    transitions = 0
    for col in range(WIDTH):
        prev = 0
        for row in range(HEIGHT):
            cur = 1 if _cell_filled(grid, row, col) else 0
            if cur != prev:
                transitions += 1
                prev = cur
        if prev != 0:
            transitions += 1
    return transitions


def features_from_grid(grid: GridLike, lines: int) -> list[float]:
    heights = column_heights(grid)
    holes = count_holes(grid)
    bump = bumpiness(heights)
    max_height = max(heights) if heights else 0
    wells = well_metrics(heights)
    tetris_well_depth = wells.tetris_well if holes == 0 else 0
    contact = contact_area(grid)
    row_trans = row_transitions(grid)
    col_trans = col_transitions(grid)
    aggregate_height = sum(heights)

    cleared = int(lines)
    s_lines = cleared / 4.0
    s_lines_sq = (cleared * cleared) / 16.0
    denom_area = WIDTH * HEIGHT if WIDTH * HEIGHT else 1
    s_holes = holes / denom_area
    s_bump = bump / (((WIDTH - 1) * HEIGHT) if WIDTH > 1 else 1)
    s_max_h = max_height / HEIGHT if HEIGHT else 0.0
    s_well = wells.well_sum / denom_area
    s_edge = wells.edge_well / HEIGHT if HEIGHT else 0.0
    s_tetris = tetris_well_depth / HEIGHT if HEIGHT else 0.0
    s_contact = contact / (WIDTH * HEIGHT * 2) if WIDTH and HEIGHT else 0.0
    s_row = row_trans / denom_area
    s_col = col_trans / denom_area
    s_agg = aggregate_height / denom_area

    return [
        s_lines,
        s_lines_sq,
        1.0 if cleared == 1 else 0.0,
        1.0 if cleared == 2 else 0.0,
        1.0 if cleared == 3 else 0.0,
        1.0 if cleared == 4 else 0.0,
        s_holes,
        s_bump,
        s_max_h,
        s_well,
        s_edge,
        s_tetris,
        s_contact,
        s_row,
        s_col,
        s_agg,
    ]


__all__ = [
    "FEATURE_NAMES",
    "FEAT_DIM",
    "WellMetrics",
    "column_heights",
    "count_holes",
    "bumpiness",
    "well_metrics",
    "contact_area",
    "row_transitions",
    "col_transitions",
    "features_from_grid",
]
