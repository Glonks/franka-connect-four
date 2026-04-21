#!/usr/bin/env python3
"""
Build MJCF for Lab 3 table scene: 4×4 placement grid + sixteen blocks (8 red / 8 blue).

Visual grid markers are thin floor decals (no collision). Block bodies match
``lab3.geometry`` for planning.

Writes: franka_emika_panda/panda_torque_table_lab3.xml

Run from repo root:
    python lab3/build_lab3_xml.py
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lab3 import geometry as geo

ROOT_MODEL_XML = os.path.join(_REPO, "franka_emika_panda", "panda_torque_table.xml")
OUTPUT_XML = os.path.join(_REPO, "franka_emika_panda", "panda_torque_table_lab3.xml")

TABLE_PLANE_RGBA = [0.2, 0.2, 0.9, 1.0]

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


def _add_grid_slot_markers(worldbody: ET.Element) -> None:
    z = 0.002
    half = geo.CELL_SPACING * 0.48
    hz = 0.0008
    n = geo.GRID_SIZE
    for row in range(n):
        for col in range(n):
            cx, cy = geo.cell_center(row, col)
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


def _build_block_specs() -> list[tuple[int, list[float], list[float]]]:
    specs: list[tuple[int, list[float], list[float]]] = []
    z = geo.Z_BLOCK_CENTER
    for i in range(8):
        x, y, _ = geo.stack_world_pose("red", i)
        specs.append((i, [x, y, z], RED_RGBA))
    for i in range(8):
        x, y, _ = geo.stack_world_pose("blue", i)
        specs.append((8 + i, [x, y, z], BLUE_RGBA))
    return specs


def build_lab3_mjcf() -> ET.ElementTree:
    tree = ET.parse(ROOT_MODEL_XML)
    worldbody = tree.getroot().find("worldbody")
    if worldbody is None:
        raise RuntimeError("No worldbody in base MJCF")

    _add_free_block(
        worldbody,
        "TablePlane",
        geo.TABLE_PLANE_POS,
        20,
        geo.TABLE_PLANE_SIZE,
        TABLE_PLANE_RGBA,
        free=False,
    )

    _add_grid_slot_markers(worldbody)

    for bid, pos, rgba in _build_block_specs():
        _add_free_block(
            worldbody,
            f"Block{bid}",
            pos,
            20,
            [geo.BLOCK_HALF_SIZE, geo.BLOCK_HALF_SIZE, geo.BLOCK_HALF_SIZE],
            rgba,
            free=True,
        )

    return tree


def main() -> None:
    tree = build_lab3_mjcf()
    tree.write(OUTPUT_XML, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {OUTPUT_XML}")
    yb = geo.stack_y_edge("red")
    yt = geo.stack_y_edge("blue")
    print(
        f"  {geo.GRID_SIZE}×{geo.GRID_SIZE} grid (GridSlot_*), "
        f"8 red along bottom edge (y≈{yb:.3f}), 8 blue along top edge (y≈{yt:.3f}), "
        f"slots stepped along X"
    )
    print(f"  cell spacing={geo.CELL_SPACING}, stack_x spacing={geo.STACK_X_SPACING}")


if __name__ == "__main__":
    main()
