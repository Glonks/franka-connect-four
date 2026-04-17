#!/usr/bin/env python3
"""Open the Lab 3 MJCF in MuJoCo with the camera aimed at the table + blocks.

`run.py` loads `panda_torque_table_shelves.xml`, so use this to preview Lab 3:

    python lab3/view_lab3.py
"""

from __future__ import annotations

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_XML = os.path.join(_REPO, "franka_emika_panda", "panda_torque_table_lab3.xml")

if not os.path.isfile(_XML):
    print(f"Missing {_XML} — run: python lab3/build_lab3_xml.py", file=sys.stderr)
    sys.exit(1)

import mujoco as mj
from mujoco import viewer


def main() -> None:
    model = mj.MjModel.from_xml_path(_XML)
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    with viewer.launch_passive(model, data) as v:
        v.cam.lookat[:] = [0.55, 0.0, 0.12]
        v.cam.distance = 1.65
        v.cam.azimuth = 140.0
        v.cam.elevation = -28.0

        while v.is_running():
            mj.mj_step(model, data)
            v.sync()


if __name__ == "__main__":
    main()
