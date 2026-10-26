"""The registry of scenes, in the order the page shows them."""
from __future__ import annotations

from dataclasses import replace
from functools import cache

from plan_and_act.scenes.base import Layout, ObjSpec, Scene, Task
from plan_and_act.scenes.bucket import BUCKET
from plan_and_act.scenes.buttons import BUTTONS_SCENE
from plan_and_act.scenes.cubes import CUBES
from plan_and_act.scenes.drawer import DRAWER
from plan_and_act.scenes.hanoi import HANOI
from plan_and_act.scenes.kitchen import KITCHEN
from plan_and_act.scenes.laptop import LAPTOP
from plan_and_act.scenes.microwave import MICROWAVE
from plan_and_act.scenes.mouse import MOUSE
from plan_and_act.scenes.peg import PEG
from plan_and_act.scenes.ranking import RANKING
from plan_and_act.scenes.shapes import SHAPES
from plan_and_act.scenes.shelf import SHELF
from plan_and_act.scenes.stapler import STAPLER
from plan_and_act.scenes.switch import SWITCH
from plan_and_act.scenes.toys import TOYS_SCENE
from plan_and_act.scenes.tray import TRAY
from plan_and_act.scenes.zones import ZONES_SCENE

SCENES: dict[str, Scene] = {s.name: s for s in (CUBES, TRAY, ZONES_SCENE, SHAPES, DRAWER, BUTTONS_SCENE, KITCHEN,
                                                  TOYS_SCENE, RANKING, SHELF, PEG, HANOI, BUCKET,
                                                  MOUSE, STAPLER, SWITCH, LAPTOP, MICROWAVE)}

ROBOT_SEP = "@"  # a scene key "cubes@ur5e" is the cubes scene with the UR5e arm. "cubes" has the Panda.


@cache
def find_scene(key: str) -> Scene | None:
    """The scene of a key: a scene name, then "@" and a robot name for an arm other than the Panda.
    None for an unknown scene or robot."""
    from plan_and_act.sim.robots import ROBOTS  # here, because the robots need MuJoCo
    name, _, robot = key.partition(ROBOT_SEP)
    base = SCENES.get(name)
    if base is None or (robot and robot not in ROBOTS):
        return None
    if not robot or robot == base.robot:
        return base
    return replace(base, name=f"{name}{ROBOT_SEP}{robot}", robot=robot)


__all__ = ["ROBOT_SEP", "SCENES", "Layout", "ObjSpec", "Scene", "Task", "find_scene"]
