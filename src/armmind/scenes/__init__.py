"""The registry of scenes, in the order the page shows them."""
from __future__ import annotations

from armmind.scenes.base import Layout, ObjSpec, Scene, Task
from armmind.scenes.cubes import CUBES

SCENES: dict[str, Scene] = {s.name: s for s in (CUBES,)}

__all__ = ["SCENES", "Layout", "ObjSpec", "Scene", "Task"]
