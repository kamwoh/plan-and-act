"""The registry of scenes, in the order the page shows them."""
from __future__ import annotations

from plan_and_act.scenes.base import Layout, ObjSpec, Scene, Task
from plan_and_act.scenes.buttons import BUTTONS_SCENE
from plan_and_act.scenes.cubes import CUBES
from plan_and_act.scenes.drawer import DRAWER
from plan_and_act.scenes.kitchen import KITCHEN
from plan_and_act.scenes.shapes import SHAPES
from plan_and_act.scenes.toys import TOYS_SCENE
from plan_and_act.scenes.tray import TRAY
from plan_and_act.scenes.zones import ZONES_SCENE

SCENES: dict[str, Scene] = {s.name: s for s in (CUBES, TRAY, ZONES_SCENE, SHAPES, DRAWER, BUTTONS_SCENE, KITCHEN,
                                                  TOYS_SCENE)}

__all__ = ["SCENES", "Layout", "ObjSpec", "Scene", "Task"]
