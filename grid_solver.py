#!/usr/bin/env python3
"""
Lab 3: symbolic plan → motion (GoTo + GripperAction from actions.py).

Pipeline
--------
1. **Plan** — list of `PlanStep` from `lab3.task_planner.compute_symbolic_plan()` (or your
   hardcoded list): (place|remove, red|blue, (row,col), v|h).
2. **Motion** — `build_motion_sequence` expands each step into pick/place macros:
   - **Cartesian** `GoTo(position, quaternion)` → IK + **RRT** (`lab3.rrt_obstacles.LAB3_RRT_BLOCKS`:
     **table slab only**; no shelf boxes).
   - IK **null-space bias** toward `CommonPoses.PreInitialGrasp` (not shelf poses).
   - **Joint** `GoTo(CommonPoses.Home)` at the end only.
   - `GripperAction` for open/close.

Symbolic grid coordinates → world XYZ via `lab3.geometry` (must match MJCF).

Pattern-based symbolic plans live in `lab3/pattern_grid_solver.py`.

Run (after `python lab3/build_lab3_xml.py`):
    python grid_solver.py
"""

from __future__ import annotations

import functools
import time
from typing import List, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

import mujoco as mj
from mujoco import viewer

from actions import CommonPoses, GoTo, GripperAction, GripperState
from inverse_kinematics import IKSolver
from kinematics import PandaKinematics
from motion_planning import RRTPlanner
from runtime import MujocoRuntime

from lab3 import geometry as geo
from lab3.rrt_obstacles import LAB3_RRT_BLOCKS
from lab3.task_planner import PlanStep, compute_symbolic_plan

ARM_INDEX = [0, 1, 2, 3, 4, 5, 6]
ROOT_MODEL_XML = "franka_emika_panda/panda_torque_table.xml"
MODEL_XML = "franka_emika_panda/panda_torque_table_lab3.xml"

# Hand targets at block height often IK into configurations the table box marks as
# collision; bias IK + keep the palm a bit higher than block centers for planning.
_MIN_HAND_Z = 0.14
_IK_BIAS = np.asarray(CommonPoses.PreInitialGrasp, dtype=np.float64)


# Vertical / horizontal top grasp: small rotation difference about world Z.
def grip_quaternion(grip: str) -> np.ndarray:
    if grip == "v":
        rot = R_scipy.from_euler("xyz", [np.pi - 0.15, 0.0, 0.0])
    else:
        rot = R_scipy.from_euler("xyz", [np.pi - 0.15, 0.0, np.pi / 2])
    x, y, z, w = rot.as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def _pose(pos: np.ndarray, quat: np.ndarray):
    return (np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64))


def _above(p: np.ndarray, dz: float) -> np.ndarray:
    out = np.array(p, dtype=np.float64, copy=True)
    out[2] += dz
    return out


def _hand_target(p: np.ndarray) -> np.ndarray:
    """Lift low table poses so elbow/wrist clears the analytic table slab in RRT."""
    out = np.array(p, dtype=np.float64, copy=True)
    out[2] = max(float(out[2]), _MIN_HAND_Z)
    return out


class BlockState:
    """Track which block id sits on which stack slot / grid cell (matches MJCF Block0..7)."""

    def __init__(self) -> None:
        self.grid: list[list[int | None]] = [[None] * 3 for _ in range(3)]
        self.stack_slots: dict[str, dict[int, int]] = {
            "red": {i: i for i in range(4)},
            "blue": {i: 4 + i for i in range(4)},
        }

    @staticmethod
    def color_of_block(bid: int) -> str:
        return "red" if bid < 4 else "blue"

    def take_block_for_place(self, color: str) -> int:
        for slot in (3, 2, 1, 0):
            if slot in self.stack_slots[color]:
                return self.stack_slots[color].pop(slot)
        raise RuntimeError(f"No {color} block left on stack")

    def lowest_free_slot(self, color: str) -> int:
        for slot in range(4):
            if slot not in self.stack_slots[color]:
                return slot
        raise RuntimeError(f"No free {color} stack slot")

    def place_on_grid(self, color: str, row: int, col: int) -> None:
        bid = self.take_block_for_place(color)
        if self.grid[row][col] is not None:
            raise RuntimeError("Grid cell occupied")
        self.grid[row][col] = bid

    def remove_to_stack(self, color: str, row: int, col: int) -> int:
        """Returns target stack slot for the remove (lowest free)."""
        bid = self.grid[row][col]
        if bid is None:
            raise RuntimeError("No block in cell")
        if self.color_of_block(bid) != color:
            raise RuntimeError("Color mismatch vs grid")
        slot = self.lowest_free_slot(color)
        self.grid[row][col] = None
        self.stack_slots[color][slot] = bid
        return slot

    def top_stack_slot(self, color: str) -> int:
        for slot in (3, 2, 1, 0):
            if slot in self.stack_slots[color]:
                return slot
        raise RuntimeError(f"No {color} block on stack")


def _fallback_plan() -> List[PlanStep]:
    """Last resort if `compute_symbolic_plan()` returns an empty list."""
    return [("place", "red", (1, 1), "v"), ("remove", "red", (1, 1), "v")]


def load_symbolic_plan() -> List[PlanStep]:
    plan = compute_symbolic_plan()
    return plan if plan else _fallback_plan()


def make_lab3_rrt(robot_model: PandaKinematics) -> RRTPlanner:
    """RRT with table-only analytic obstacles (see `lab3/rrt_obstacles.py`)."""
    return RRTPlanner(
        robot_model,
        LAB3_RRT_BLOCKS,
        step_size=0.08,
        max_iterations=8000,
        goal_threshold=0.35,
    )


def build_lab3_actions(
    robot_model: PandaKinematics,
    ik_solver: IKSolver,
    planner: RRTPlanner,
) -> list:
    """Task planner (`lab3.task_planner`) → motion primitives (`GoTo` / `GripperAction`)."""
    return build_motion_sequence(load_symbolic_plan(), robot_model, ik_solver, planner)


def build_motion_sequence(
    plan: Sequence[PlanStep],
    robot_model: PandaKinematics,
    ik_solver: IKSolver,
    planner: RRTPlanner,
    state: BlockState | None = None,
) -> list:
    """Expand each plan tuple into pick/place via Cartesian `GoTo` (IK+RRT) and gripper macros.

    Cartesian moves use `bias=PreInitialGrasp`; final return uses joint `GoTo(Home)` without bias.
    RRT collision model is `LAB3_RRT_BLOCKS` (table only), independent of the symbolic grid.
    """
    st = state or BlockState()
    _GoTo = functools.partial(
        GoTo, robot_model=robot_model, ik_solver=ik_solver, planner=planner
    )
    _GoToCart = functools.partial(
        GoTo,
        robot_model=robot_model,
        ik_solver=ik_solver,
        planner=planner,
        bias=_IK_BIAS,
    )
    _Open = functools.partial(GripperAction, gripper_target=GripperState.Open)
    _Close = functools.partial(GripperAction, gripper_target=GripperState.Closed)

    dz_approach = 0.14
    actions: list = []

    def append_pick_at_world(p: np.ndarray, quat: np.ndarray) -> None:
        p_h = _hand_target(p)
        actions.append(_Open())
        actions.append(_GoToCart(_pose(_above(p_h, dz_approach), quat)))
        actions.append(_GoToCart(_pose(p_h, quat)))
        actions.append(_Close())
        actions.append(_GoToCart(_pose(_above(p_h, 0.12), quat)))

    def append_place_at_world(p: np.ndarray, quat: np.ndarray) -> None:
        p_h = _hand_target(p)
        actions.append(_GoToCart(_pose(_above(p_h, dz_approach), quat)))
        actions.append(_GoToCart(_pose(p_h, quat)))
        actions.append(_Open())
        actions.append(_GoToCart(_pose(_above(p_h, 0.12), quat)))

    for op, color, (row, col), grip in plan:
        quat = grip_quaternion(grip)
        if op == "place":
            slot = st.top_stack_slot(color)
            p_stack = np.array(geo.stack_world_pose(color, slot), dtype=np.float64)
            p_grid = np.array(geo.grid_world_pose(row, col), dtype=np.float64)
            append_pick_at_world(p_stack, quat)
            append_place_at_world(p_grid, quat)
            st.place_on_grid(color, row, col)
        elif op == "remove":
            p_grid = np.array(geo.grid_world_pose(row, col), dtype=np.float64)
            slot = st.lowest_free_slot(color)
            append_pick_at_world(p_grid, quat)
            p_stack = np.array(geo.stack_world_pose(color, slot), dtype=np.float64)
            append_place_at_world(p_stack, quat)
            st.remove_to_stack(color, row, col)
        else:
            raise ValueError(op)

    actions.append(_GoTo(CommonPoses.Home))
    return actions


def main() -> None:
    robot_model = PandaKinematics(ROOT_MODEL_XML)
    ik_solver = IKSolver(robot_model)
    planner = make_lab3_rrt(robot_model)

    plan = load_symbolic_plan()
    print("Symbolic plan (task_planner → motion):", plan)
    actions = build_motion_sequence(plan, robot_model, ik_solver, planner)
    print(f"Built {len(actions)} motion primitives")

    model = mj.MjModel.from_xml_path(MODEL_XML)
    data = mj.MjData(model)
    runtime = MujocoRuntime(model, data, ARM_INDEX)
    runtime.set_configuration(CommonPoses.Home)

    with viewer.launch_passive(model, data) as v:
        v.cam.lookat[:] = [0.55, 0.0, 0.12]
        v.cam.distance = 1.65
        v.cam.azimuth = 140.0
        v.cam.elevation = -28.0

        for action in actions:
            print(action)
            t, done = 0.0, False
            while not done:
                if not v.is_running():
                    break
                command, done = action.control(runtime.get_state(), t)
                runtime.step(command)
                v.sync()
                t += runtime.dt
            time.sleep(0.05)

    time.sleep(0.1)


if __name__ == "__main__":
    main()
