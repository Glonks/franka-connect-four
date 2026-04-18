"""
Shared Lab 3 table / grid / supply-stack geometry (matches build_lab3_xml output).

Grid indices: row 0..2, col 0..2 (row grows with +Y in world; col grows with +X).
Red stack is at smaller world X; blue stack at larger world X.
"""

from __future__ import annotations

EndofTable = 0.55 + 0.135 + 0.05

TABLE_PLANE_POS = [EndofTable - 0.275, 0.0, -0.005]
TABLE_PLANE_SIZE = [0.275, 0.504, 0.0051]

GRID_CENTER_X = EndofTable - 0.145
GRID_CENTER_Y = 0.0
CELL_SPACING = 0.052
Z_BLOCK_CENTER = 0.05
BLOCK_HALF_SIZE = 0.02

MARGIN_GRID_TO_COLUMN = 0.045
_EDGE_CLEAR = 0.015


def table_x_range() -> tuple[float, float]:
    cx, hx = TABLE_PLANE_POS[0], TABLE_PLANE_SIZE[0]
    return cx - hx, cx + hx


def lateral_offset_from_grid_center() -> float:
    preferred = 1.5 * CELL_SPACING + 2.0 * BLOCK_HALF_SIZE + MARGIN_GRID_TO_COLUMN
    x_min, x_max = table_x_range()
    dx_cap = min(
        GRID_CENTER_X - x_min - BLOCK_HALF_SIZE - _EDGE_CLEAR,
        x_max - GRID_CENTER_X - BLOCK_HALF_SIZE - _EDGE_CLEAR,
    )
    return max(0.04, min(preferred, dx_cap))


def cell_center(row: int, col: int) -> tuple[float, float]:
    x = GRID_CENTER_X + (col - 1) * CELL_SPACING
    y = GRID_CENTER_Y + (row - 1) * CELL_SPACING
    return (x, y)


def stack_row_y() -> list[float]:
    return [
        -1.5 * CELL_SPACING,
        -0.5 * CELL_SPACING,
        0.5 * CELL_SPACING,
        1.5 * CELL_SPACING,
    ]


def stack_x(color: str) -> float:
    dx = lateral_offset_from_grid_center()
    if color == "red":
        return GRID_CENTER_X - dx
    if color == "blue":
        return GRID_CENTER_X + dx
    raise ValueError(color)


def stack_world_pose(color: str, slot: int) -> tuple[float, float, float]:
    """slot 0..3 matches MJCF row ordering in build_lab3_xml."""
    ys = stack_row_y()
    if not 0 <= slot < 4:
        raise ValueError(slot)
    return (stack_x(color), ys[slot], Z_BLOCK_CENTER)


def grid_world_pose(row: int, col: int) -> tuple[float, float, float]:
    x, y = cell_center(row, col)
    return (x, y, Z_BLOCK_CENTER)
