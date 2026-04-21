"""
Shared Lab 3 table / grid / supply-stack geometry (matches build_lab3_xml output).

Grid: ``GRID_SIZE``×``GRID_SIZE`` (default 4×4). Indices row, col in ``0 .. GRID_SIZE-1``
(+Y row, +X col).

Sixteen blocks: ids 0–7 red, 8–15 blue. Supply sits on the **two Y-edges** of the grid
(**bottom** and **top** in world **Y**): red along the **smaller-Y** edge, blue along the
**larger-Y** edge, each with **eight slots stepped along X** (not left/right columns along Y).
"""

from __future__ import annotations

EndofTable = 0.55 + 0.135 + 0.05

TABLE_PLANE_POS = [EndofTable - 0.275, 0.0, -0.005]
TABLE_PLANE_SIZE = [0.275, 0.504, 0.0051]

GRID_CENTER_X = EndofTable - 0.26
GRID_CENTER_Y = 0.0

CELL_SPACING = 0.07
GRID_SIZE = 4

Z_BLOCK_CENTER = 0.05
BLOCK_HALF_SIZE = 0.02

MARGIN_GRID_TO_COLUMN = 0.05
_EDGE_CLEAR = 0.015

# Eight slots per edge, stepped along X (parallel to grid columns).
STACK_X_SPACING = 0.055


def table_x_range() -> tuple[float, float]:
    cx, hx = TABLE_PLANE_POS[0], TABLE_PLANE_SIZE[0]
    return cx - hx, cx + hx


def lateral_offset_from_grid_center() -> float:
    half_grid = ((GRID_SIZE - 1) / 2) * CELL_SPACING
    preferred = half_grid + BLOCK_HALF_SIZE + MARGIN_GRID_TO_COLUMN
    x_min, x_max = table_x_range()
    dx_cap = min(
        GRID_CENTER_X - x_min - BLOCK_HALF_SIZE - _EDGE_CLEAR,
        x_max - GRID_CENTER_X - BLOCK_HALF_SIZE - _EDGE_CLEAR,
    )
    return max(0.04, min(preferred, dx_cap))


def cell_center(row: int, col: int) -> tuple[float, float]:
    if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
        raise ValueError(f"cell ({row},{col}) out of range for GRID_SIZE={GRID_SIZE}")
    mid = (GRID_SIZE - 1) / 2
    x = GRID_CENTER_X + (col - mid) * CELL_SPACING
    y = GRID_CENTER_Y + (row - mid) * CELL_SPACING
    return (x, y)


def _y_supply_bottom() -> float:
    """World Y for supply along the grid edge with smaller Y (below row 0)."""
    mid = (GRID_SIZE - 1) / 2
    y_row0 = GRID_CENTER_Y + (0 - mid) * CELL_SPACING
    return y_row0 - CELL_SPACING * 0.5 - BLOCK_HALF_SIZE - MARGIN_GRID_TO_COLUMN


def _y_supply_top() -> float:
    """World Y for supply along the grid edge with larger Y (above last row)."""
    mid = (GRID_SIZE - 1) / 2
    last = GRID_SIZE - 1
    y_last = GRID_CENTER_Y + (last - mid) * CELL_SPACING
    return y_last + CELL_SPACING * 0.5 + BLOCK_HALF_SIZE + MARGIN_GRID_TO_COLUMN


def stack_y_edge(color: str) -> float:
    """Fixed world Y for one color's supply row (red = bottom edge, blue = top edge)."""
    if color == "red":
        return _y_supply_bottom()
    if color == "blue":
        return _y_supply_top()
    raise ValueError(color)


def stack_row_x() -> list[float]:
    """Eight X centers along each supply edge (slot 0 = min X, 7 = max X)."""
    return [GRID_CENTER_X + (-3.5 + k) * STACK_X_SPACING for k in range(8)]


def stack_world_pose(color: str, slot: int) -> tuple[float, float, float]:
    """slot 0..7 — same Y per color; X steps along the top/bottom edge."""
    xs = stack_row_x()
    if not 0 <= slot < 8:
        raise ValueError(slot)
    return (xs[slot], stack_y_edge(color), Z_BLOCK_CENTER)


def grid_world_pose(row: int, col: int) -> tuple[float, float, float]:
    x, y = cell_center(row, col)
    return (x, y, Z_BLOCK_CENTER)
