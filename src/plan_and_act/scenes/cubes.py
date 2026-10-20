"""Three cubes on a table: reach, pick and stack."""
from __future__ import annotations

import numpy as np

from plan_and_act.scenes.base import Scene, Task
from plan_and_act.scenes.common import LIFT_Z, THREE_CUBES, above, free, grasp, put
from plan_and_act.types import Action

HALF = 0.02


def _reach(w) -> bool:
    return "red_cube" in w.gripper_contacts()


def _pick(w) -> bool:
    return bool(w.object_pos("red_cube")[2] >= HALF + 0.10)


def _stack(w) -> bool:
    red, blue = w.object_pos("red_cube"), w.object_pos("blue_cube")
    on_top = np.hypot(*(red[:2] - blue[:2])) < 0.02 and abs(red[2] - (blue[2] + 2 * HALF)) < 0.01
    return bool(on_top and free(w, "red_cube"))


def _reach_plan(w) -> list[Action]:
    red = w.object_pos("red_cube")
    return [Action("close"), above(red, LIFT_Z), above(red, 0.04)]


def _pick_plan(w) -> list[Action]:
    return [*grasp(w, "red_cube"), above(w.object_pos("red_cube"), 0.20)]


def _stack_plan(w) -> list[Action]:
    return put(w, "red_cube", w.object_pos("blue_cube"), 0.065)


CUBES = Scene(
    name="cubes", title="Cubes", objects=THREE_CUBES, fixtures_xml="", fixtures_text="",
    region=(0.40, 0.65, -0.25, 0.25), min_gap=0.10,
    tasks={t.name: t for t in (
        Task("reach", "Touch the red cube with the gripper.", _reach, _reach_plan),
        Task("pick", "Pick up the red cube and lift it at least 10 cm above the table.", _pick, _pick_plan),
        Task("stack", "Put the red cube on top of the blue cube and let go of it.", _stack, _stack_plan),
    )},
)
