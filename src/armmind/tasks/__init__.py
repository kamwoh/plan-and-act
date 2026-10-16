"""The three v1 tasks. Each has an instruction for the model and a success check in code."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from armmind.sim.world import CUBE_HALF, CUBES, Layout, World

MIN_GAP = 0.10


def random_layout(seed: int) -> Layout:
    rng = np.random.default_rng(seed)
    points: list[tuple[float, float]] = []
    while len(points) < len(CUBES):
        p = (round(float(rng.uniform(0.40, 0.65)), 3), round(float(rng.uniform(-0.25, 0.25)), 3))
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= MIN_GAP for q in points):
            points.append(p)
    return dict(zip(CUBES, points, strict=True))


@dataclass(frozen=True)
class Task:
    name: str
    instruction: str
    success: Callable[[World], bool]

    def layout(self, seed: int) -> Layout:
        return random_layout(seed)


def _reach(w: World) -> bool:
    return "red_cube" in w.gripper_contacts()


def _pick(w: World) -> bool:
    return bool(w.cube_pos("red_cube")[2] >= CUBE_HALF + 0.10)


def _stack(w: World) -> bool:
    red, blue = w.cube_pos("red_cube"), w.cube_pos("blue_cube")
    on_top = np.hypot(*(red[:2] - blue[:2])) < 0.02 and abs(red[2] - (blue[2] + 2 * CUBE_HALF)) < 0.01
    return bool(on_top and "red_cube" not in w.gripper_contacts())


TASKS: dict[str, Task] = {t.name: t for t in (
    Task("reach", "Touch the red cube with the gripper.", _reach),
    Task("pick", "Pick up the red cube and lift it at least 10 cm above the table.", _pick),
    Task("stack", "Put the red cube on top of the blue cube and let go of it.", _stack),
)}
