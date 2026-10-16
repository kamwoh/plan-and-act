"""A big cube, a small cube and a standing cylinder: different sizes and shapes."""
from __future__ import annotations

import numpy as np

from armmind.scenes.base import ObjSpec, Scene, Task
from armmind.scenes.common import above, free, grasp, put

BIG = ObjSpec("big_cube", "box", (0.03, 0.03, 0.03), "0.85 0.3 0.02 1", "big orange cube, 6 cm", mass=0.08)
SMALL = ObjSpec("small_cube", "box", (0.015, 0.015, 0.015), "0.5 0.2 0.75 1", "small purple cube, 3 cm", mass=0.03)
CYL = ObjSpec("cylinder", "cylinder", (0.02, 0.03), "0.95 0.85 0.1 1",
              "yellow cylinder, standing, 4 cm wide and 6 cm tall", mass=0.05)


def _pick(w) -> bool:
    return bool(w.object_pos("cylinder")[2] >= CYL.half_height + 0.10)


def _small_on_big(w) -> bool:
    small, big = w.object_pos("small_cube"), w.object_pos("big_cube")
    on_top = (np.hypot(*(small[:2] - big[:2])) < 0.02
              and abs(small[2] - (big[2] + BIG.half_height + SMALL.half_height)) < 0.01)
    return bool(on_top and free(w, "small_cube"))


def _pick_plan(w):
    return [*grasp(w, "cylinder"), above(w.object_pos("cylinder"), 0.20)]


def _small_on_big_plan(w):
    top = 2 * BIG.half_height + SMALL.half_height + 0.005  # release 5 mm above the resting height
    return put(w, "small_cube", w.object_pos("big_cube"), top)


SHAPES = Scene(
    name="shapes", title="Shapes", objects=(BIG, SMALL, CYL), fixtures_xml="", fixtures_text="",
    region=(0.40, 0.65, -0.25, 0.25), min_gap=0.12,
    tasks={t.name: t for t in (
        Task("pick_cylinder", "Pick up the yellow cylinder and lift it at least 10 cm above the table.",
             _pick, _pick_plan),
        Task("small_on_big", "Put the small purple cube on top of the big orange cube and let go of it.",
             _small_on_big, _small_on_big_plan),
    )},
)
