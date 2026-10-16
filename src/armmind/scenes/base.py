"""A scene: the objects on the table, the fixed items, where the objects start, and the tasks."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from armmind.types import Action

if TYPE_CHECKING:
    from armmind.sim.world import World

Layout = dict[str, tuple[float, float, float]]  # name -> (x, y, yaw in degrees)
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
    shape: str  # "box", "cylinder" or "mesh"
    size: tuple[float, ...] = ()  # MuJoCo geom size: box (hx, hy, hz), cylinder (radius, half_height); a mesh: ()
    rgba: str = ""
    label: str = ""  # for the prompt, e.g. "red cube, 4 cm". A mesh object gets its measured size after it.
    mass: float = 0.05
    mesh: str = ""  # a scanned object: its folder name in assets/scanned_objects/models
    random_yaw: bool = False  # the layout turns the object by a random yaw
    place: tuple[float, float] | None = None  # a fixed start point (x, y) in place of a random one

    def __post_init__(self):
        if (self.shape == "mesh") != bool(self.mesh):
            raise ValueError(f"{self.name}: give a mesh name exactly when the shape is mesh")

    @property
    def dims(self) -> tuple[float, float, float]:
        """Width (x), depth (y) and height in the object frame. A mesh object is measured."""
        if self.shape == "box":
            return (2 * self.size[0], 2 * self.size[1], 2 * self.size[2])
        if self.shape == "cylinder":
            return (2 * self.size[0], 2 * self.size[0], 2 * self.size[1])
        from armmind.sim.meshes import measure  # here, because sim imports this module
        return measure(self.mesh).dims

    @property
    def half_height(self) -> float:
        return self.dims[2] / 2

    @property
    def footprint(self) -> float:
        """The radius of the object's footprint on the table: half the horizontal diagonal."""
        if self.shape == "cylinder":
            return float(self.size[0])
        return float(np.hypot(self.dims[0], self.dims[1]) / 2)

    def text(self) -> str:
        """The label, and for a mesh object its measured size."""
        if self.shape != "mesh":
            return self.label
        from armmind.sim.assets import AssetsMissing
        try:
            w, d, h = self.dims
        except AssetsMissing:
            return self.label
        return f"{self.label}, {w * 100:.1f} cm wide, {d * 100:.1f} cm deep, {h * 100:.1f} cm tall"


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
    # Scene joints to report in the debug positions: name -> text with units, e.g. "drawer slide {:.3f} m".
    joints: dict[str, str] = field(default_factory=dict)
    # Monitor factories: each world makes its own monitors, because a monitor keeps state.
    monitors: tuple[Callable[[], Monitor], ...] = ()
    initial_qpos: dict[str, float] = field(default_factory=dict)  # scene joint positions at the start
    camera_overrides: dict[str, tuple[str, str]] = field(default_factory=dict)  # camera: (pos, xyaxes)
    image_text: str | None = None  # replaces the last line of the step message, which names the images
    camera_labels: dict[str, str] = field(default_factory=lambda: {"front": "Front", "top": "Top · grid"})

    @property
    def meshes(self) -> tuple[str, ...]:
        """The scanned objects that the scene needs."""
        return tuple(o.mesh for o in self.objects if o.shape == "mesh")

    def layout(self, seed: int) -> Layout:
        """Start poses: fixed places first, then random points in the region. The same seed gives the same
        layout. Two objects keep min_gap apart, or more when their footprints need more."""
        if not self.objects:
            return {}
        x0, x1, y0, y1 = self.region
        rng = np.random.default_rng(seed)
        out: Layout = {o.name: (*o.place, 0.0) for o in self.objects if o.place is not None}
        radius = {o.name: o.footprint for o in self.objects}
        tries = 0
        for o in self.objects:
            if o.place is not None:
                continue
            while True:
                tries += 1
                if tries > MAX_TRIES:
                    raise ValueError(f"scene {self.name}: no layout with min_gap {self.min_gap} "
                                     f"after {MAX_TRIES} tries")
                x, y = round(float(rng.uniform(x0, x1)), 3), round(float(rng.uniform(y0, y1)), 3)
                if all(np.hypot(x - q[0], y - q[1]) >= max(self.min_gap, radius[o.name] + radius[n])
                       for n, q in out.items()):
                    break
            # A yaw in (-180, 180], drawn only for a turned object, so that old layouts stay the same.
            yaw = round(180.0 - float(rng.uniform(0.0, 360.0)), 1) if o.random_yaw else 0.0
            yaw = yaw + 360.0 if yaw <= -180.0 else yaw
            out[o.name] = (x, y, yaw)
        return {o.name: out[o.name] for o in self.objects}

    def describe(self) -> str:
        text = "Objects: " + ("; ".join(o.text() for o in self.objects) or "none") + "."
        return f"{text} {self.fixtures_text}" if self.fixtures_text else text
