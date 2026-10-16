"""The registry of scenes, in the order the page shows them."""
from __future__ import annotations

from armmind.scenes.base import Layout, ObjSpec, Scene, Task
from armmind.scenes.buttons import BUTTONS_SCENE
from armmind.scenes.cubes import CUBES
from armmind.scenes.drawer import DRAWER
from armmind.scenes.kitchen import KITCHEN
from armmind.scenes.shapes import SHAPES
from armmind.scenes.toys import TOYS_SCENE
from armmind.scenes.tray import TRAY
from armmind.scenes.zones import ZONES_SCENE

SCENES: dict[str, Scene] = {s.name: s for s in (CUBES, TRAY, ZONES_SCENE, SHAPES, DRAWER, BUTTONS_SCENE, KITCHEN,
                                                  TOYS_SCENE)}

__all__ = ["SCENES", "Layout", "ObjSpec", "Scene", "Task"]
