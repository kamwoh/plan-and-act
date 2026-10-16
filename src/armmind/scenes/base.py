"""A scene: the objects on the table, the fixed items, where the objects start, and the tasks."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from armmind.types import Action

if TYPE_CHECKING:
    from armmind.sim.world import World

Layout = dict[str, tuple[float, float]]
MAX_TRIES = 10000  # random points to try before layout() gives up


@dataclass(frozen=True)
class ObjSpec:
    name: str
    shape: str  # "box" or "cylinder"
    size: tuple[float, ...]  # MuJoCo geom size: box (hx, hy, hz), cylinder (radius, half_height)
    rgba: str
    label: str  # for the prompt, e.g. "red cube, 4 cm"
    mass: float = 0.05

    @property
    def half_height(self) -> float:
        return self.size[2] if self.shape == "box" else self.size[1]


@dataclass(frozen=True)
class Task:
    name: str
    instruction: str
    success: Callable[[World], bool]
    solution: Callable[[World], list[Action]]  # a hard-coded plan that reads the true positions


@dataclass(frozen=True)
class Scene:
    name: str
    title: str
    objects: tuple[ObjSpec, ...]
    fixtures_xml: str  # extra static geoms for <worldbody>
    fixtures_text: str  # the fixed items, for the prompt
    region: tuple[float, float, float, float]  # x_min, x_max, y_min, y_max where the objects start
    min_gap: float  # the smallest distance between two object centres
    tasks: dict[str, Task]

    def layout(self, seed: int) -> Layout:
        """Random start positions. The same seed gives the same layout."""
        x0, x1, y0, y1 = self.region
        rng = np.random.default_rng(seed)
        points: list[tuple[float, float]] = []
        for _ in range(MAX_TRIES):
            p = (round(float(rng.uniform(x0, x1)), 3), round(float(rng.uniform(y0, y1)), 3))
            if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= self.min_gap for q in points):
                points.append(p)
                if len(points) == len(self.objects):
                    return dict(zip((o.name for o in self.objects), points, strict=True))
        raise ValueError(f"scene {self.name}: no layout with min_gap {self.min_gap} after {MAX_TRIES} tries")

    def describe(self) -> str:
        text = "Objects: " + "; ".join(o.label for o in self.objects) + "."
        return f"{text} {self.fixtures_text}" if self.fixtures_text else text
