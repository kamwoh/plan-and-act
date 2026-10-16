"""A scene: the objects on the table, the fixed items, where the objects start, and the tasks."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from armmind.types import Action

if TYPE_CHECKING:
    from armmind.sim.world import World

Layout = dict[str, tuple[float, float]]
Monitor = Callable[["World"], "str | None"]  # called after each physics step; returns an event or None
MAX_TRIES = 10000  # random points to try before layout() gives up


def threshold_monitor(joint: str, fire_at: float, rearm_at: float, event: str) -> Callable[[], Monitor]:
    """A factory for a monitor that gives `event` once when the joint goes above fire_at.
    It fires again only after the joint comes back below rearm_at."""
    def make() -> Monitor:
        armed = True

        def check(w: World) -> str | None:
            nonlocal armed
            q = w.joint_pos(joint)
            if armed and q > fire_at:
                armed = False
                return event
            if q < rearm_at:
                armed = True
            return None
        return check
    return make


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
    initial_qpos: dict[str, float] | None = None  # scene joint positions at the start, over the scene's own


@dataclass(frozen=True)
class Scene:
    name: str
    title: str
    objects: tuple[ObjSpec, ...]
    fixtures_xml: str  # extra bodies for <worldbody>: static geoms, or parts on joints
    fixtures_text: str  # the fixed items, for the prompt
    region: tuple[float, float, float, float]  # x_min, x_max, y_min, y_max where the objects start
    min_gap: float  # the smallest distance between two object centres
    tasks: dict[str, Task]
    joints: tuple[str, ...] = ()  # scene joints to report in the debug positions
    # Monitor factories: each world makes its own monitors, because a monitor keeps state.
    monitors: tuple[Callable[[], Monitor], ...] = ()
    initial_qpos: dict[str, float] = field(default_factory=dict)  # scene joint positions at the start
    camera_overrides: dict[str, tuple[str, str]] = field(default_factory=dict)  # camera: (pos, xyaxes)
    image_text: str | None = None  # replaces the last line of the step message, which names the images
    camera_labels: dict[str, str] = field(default_factory=lambda: {"front": "Front", "top": "Top · grid"})

    def layout(self, seed: int) -> Layout:
        """Random start positions. The same seed gives the same layout."""
        if not self.objects:
            return {}
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
        text = "Objects: " + ("; ".join(o.label for o in self.objects) or "none") + "."
        return f"{text} {self.fixtures_text}" if self.fixtures_text else text
