"""Parts that several scenes share: the three cubes, plan steps and checks."""
from __future__ import annotations

from typing import TYPE_CHECKING

from plan_and_act.scenes.base import ObjSpec
from plan_and_act.types import Action

if TYPE_CHECKING:
    from plan_and_act.sim.world import World

LIFT_Z = 0.15  # the height for moves above the table

THREE_CUBES = (
    ObjSpec("red_cube", "box", (0.02, 0.02, 0.02), "0.9 0.1 0.1 1", "red cube, 4 cm"),
    ObjSpec("blue_cube", "box", (0.02, 0.02, 0.02), "0.1 0.2 0.9 1", "blue cube, 4 cm"),
    ObjSpec("green_cube", "box", (0.02, 0.02, 0.02), "0.1 0.7 0.2 1", "green cube, 4 cm"),
)


def above(p, z: float) -> Action:
    return Action("move_to", (float(p[0]), float(p[1]), z))


def grasp(w: World, name: str) -> list[Action]:
    """Open, go above the object, down to its centre, close."""
    p = w.object_pos(name)
    return [Action("open"), above(p, LIFT_Z), above(p, float(p[2])), Action("close")]


def put(w: World, name: str, xy, release_z: float) -> list[Action]:
    """Grasp the object, carry it above xy, lower it to release_z, let go and move up."""
    p = w.object_pos(name)
    return [*grasp(w, name), above(p, LIFT_Z), above(xy, LIFT_Z), above(xy, release_z), Action("open"),
            above(xy, LIFT_Z)]


def free(w: World, name: str) -> bool:
    """True when the gripper does not touch the object."""
    return name not in w.gripper_contacts()
