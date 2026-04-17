#!/usr/bin/env python3
"""
Build MJCF for Robot Autonomy Lab 3 (RobotAutonomy_Lab3.pdf).

Figure 1: central 3×3 grid of *target locations* (gray squares in the figure are
slots to place blocks, not extra cubes), four red blocks on one side, four blue
on the other.

Scene: panda_torque_table.xml + TablePlane (Lab 2 run.py) + nine thin floor
markers (visual only, no collision) showing the 3×3 + eight free red/blue blocks.

Grid cell centers match _cell_center(row, col) — use these for task planning.

Writes: franka_emika_panda/panda_torque_table_lab3.xml

Run from repo root:
    python lab3/build_lab3_xml.py
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ROOT_MODEL_XML = os.path.join(_REPO, "franka_emika_panda", "panda_torque_table.xml")
OUTPUT_XML = os.path.join(_REPO, "franka_emika_panda", "panda_torque_table_lab3.xml")

EndofTable = 0.55 + 0.135 + 0.05

TABLE_PLANE_POS = [EndofTable - 0.275, 0.0, -0.005]
TABLE_PLANE_SIZE = [0.275, 0.504, 0.0051]
TABLE_PLANE_RGBA = [0.2, 0.2, 0.9, 1.0]

GRID_CENTER_X = EndofTable - 0.145
GRID_CENTER_Y = 0.0
CELL_SPACING = 0.052
Z_BLOCK_CENTER = 0.05
BLOCK_HALF_SIZE = 0.02

MARGIN_GRID_TO_COLUMN = 0.045

RED_RGBA = [0.9, 0.12, 0.1, 1.0]
BLUE_RGBA = [0.1, 0.14, 0.9, 1.0]


def _add_free_block(
    worldbody: ET.Element,
    name: str,
    pos: list[float],
    density: float,
    size: list[float],
    rgba: list[float],
    free: bool,
) -> None:
    body = ET.SubElement(
        worldbody,
        "body",
        {"name": str(name), "pos": f"{pos[0]} {pos[1]} {pos[2]}"},
    )
    ET.SubElement(
        body,
        "geom",
        {
            "type": "box",
            "density": f"{density}",
            "size": f"{size[0]} {size[1]} {size[2]}",
            "rgba": f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}",
        },
    )
    if free:
        ET.SubElement(body, "freejoint")


def _cell_center(row: int, col: int) -> tuple[float, float]:
    x = GRID_CENTER_X + (col - 1) * CELL_SPACING
    y = GRID_CENTER_Y + (row - 1) * CELL_SPACING
    return (x, y)


def _add_grid_slot_markers(worldbody: ET.Element) -> None:
    """Nine flat decals marking where blocks may be placed — not physical blocks."""
    z = 0.002
    half = CELL_SPACING * 0.48
    hz = 0.0008
    for row in range(3):
        for col in range(3):
            cx, cy = _cell_center(row, col)
            body = ET.SubElement(
                worldbody,
                "body",
                {"name": f"GridSlot_{row}_{col}", "pos": f"{cx} {cy} {z}"},
            )
            ET.SubElement(
                body,
                "geom",
                {
                    "type": "box",
                    "size": f"{half} {half} {hz}",
                    "rgba": "0.72 0.72 0.74 0.45",
                    "contype": "0",
                    "conaffinity": "0",
                },
            )


def _lateral_offset_from_grid_center() -> float:
    return 1.5 * CELL_SPACING + 2.0 * BLOCK_HALF_SIZE + MARGIN_GRID_TO_COLUMN


def _build_figure1_block_specs() -> list[tuple[int, list[float], list[float]]]:
    dx = _lateral_offset_from_grid_center()
    x_red = GRID_CENTER_X - dx
    x_blue = GRID_CENTER_X + dx
    row_y = [
        -1.5 * CELL_SPACING,
        -0.5 * CELL_SPACING,
        0.5 * CELL_SPACING,
        1.5 * CELL_SPACING,
    ]
    specs: list[tuple[int, list[float], list[float]]] = []
    for i in range(4):
        z = Z_BLOCK_CENTER
        specs.append((i, [x_red, row_y[i], z], RED_RGBA))
        specs.append((4 + i, [x_blue, row_y[i], z], BLUE_RGBA))
    return specs


def build_lab3_mjcf() -> ET.ElementTree:
    tree = ET.parse(ROOT_MODEL_XML)
    worldbody = tree.getroot().find("worldbody")
    if worldbody is None:
        raise RuntimeError("No worldbody in base MJCF")

    _add_free_block(
        worldbody,
        "TablePlane",
        TABLE_PLANE_POS,
        20,
        TABLE_PLANE_SIZE,
        TABLE_PLANE_RGBA,
        free=False,
    )

    _add_grid_slot_markers(worldbody)

    for bid, pos, rgba in _build_figure1_block_specs():
        _add_free_block(
            worldbody,
            f"Block{bid}",
            pos,
            20,
            [BLOCK_HALF_SIZE, BLOCK_HALF_SIZE, BLOCK_HALF_SIZE],
            rgba,
            free=True,
        )

    return tree


def main() -> None:
    tree = build_lab3_mjcf()
    tree.write(OUTPUT_XML, encoding="utf-8", xml_declaration=True)
    dx = _lateral_offset_from_grid_center()
    print(f"Wrote {OUTPUT_XML}")
    print(f"  3×3 = placement targets (GridSlot_* decals), 4 red / 4 blue on sides at x≈{GRID_CENTER_X - dx:.3f} / {GRID_CENTER_X + dx:.3f}")
    print(f"  cell spacing={CELL_SPACING}")


if __name__ == "__main__":
    main()
