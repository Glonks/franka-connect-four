"""
Analytic box obstacles passed to RRTPlanner for Lab 3 (table-only scene).

Must match the TablePlane body in `panda_torque_table_lab3.xml` / `lab3/build_lab3_xml.py`.
Shelf geoms are NOT included — Lab 3 uses the table + blocks only.

Format per entry: [name, center_xyz, half_extents_xyz] (same as `run.py` BLOCKS).
"""

from __future__ import annotations

from lab3.geometry import TABLE_PLANE_POS, TABLE_PLANE_SIZE

LAB3_RRT_BLOCKS = [
    ["TablePlane", list(TABLE_PLANE_POS), list(TABLE_PLANE_SIZE)],
]
