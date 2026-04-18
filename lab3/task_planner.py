"""
Lab 3 symbolic task planner → consumed by `grid_solver.build_motion_sequence`.

Primary API
-----------
`compute_symbolic_plan() -> List[PlanStep]`

Each `PlanStep` is:
    ("place" | "remove", "red" | "blue", (row, col), "v" | "h")

Resolution order
----------------
1. Optional file `lab3/symbolic_plan.json` (repo root relative to CWD) if present.
2. Otherwise `default_symbolic_plan()` below — replace with your search / pattern code.

Repo: https://github.com/Glonks/franka-connect-four
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Tuple

PlanStep = Tuple[str, str, Tuple[int, int], str]

# Path can be overridden for tests
_PLAN_JSON_ENV = "LAB3_SYMBOLIC_PLAN_JSON"


def _parse_steps(raw: Any) -> List[PlanStep]:
    out: List[PlanStep] = []
    for row in raw:
        op, color, cell, grip = row[0], row[1], tuple(row[2]), row[3]
        if op not in ("place", "remove"):
            raise ValueError(op)
        if color not in ("red", "blue"):
            raise ValueError(color)
        if grip not in ("v", "h"):
            raise ValueError(grip)
        r, c = int(cell[0]), int(cell[1])
        out.append((op, color, (r, c), grip))
    return out


def _load_plan_json(path: str) -> List[PlanStep] | None:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "steps" in data:
        return _parse_steps(data["steps"])
    if isinstance(data, list):
        return _parse_steps(data)
    raise ValueError("symbolic_plan.json: expected list or {steps: [...]}")


def default_symbolic_plan() -> List[PlanStep]:
    """
    Short demo sequence (task planner + motion integration smoke test).
    Swap for your full pattern pipeline (pickaxe → axe → box → bow, etc.).
    """
    return [
        ("place", "red", (1, 1), "v"),
        ("place", "blue", (0, 1), "v"),
        ("remove", "blue", (0, 1), "v"),
        ("remove", "red", (1, 1), "v"),
    ]


def compute_symbolic_plan() -> List[PlanStep]:
    path = os.environ.get(_PLAN_JSON_ENV, "").strip()
    if path:
        loaded = _load_plan_json(path)
        if loaded is not None:
            return loaded
    here = os.path.dirname(__file__)
    default_path = os.path.join(here, "symbolic_plan.json")
    loaded = _load_plan_json(default_path)
    if loaded is not None:
        return loaded
    return default_symbolic_plan()
