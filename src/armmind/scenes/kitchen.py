"""Scanned kitchen objects: a heavy bowl and a heavy plate at fixed places, and three small items to put on them."""
from __future__ import annotations

import numpy as np

from armmind.scenes.base import ObjSpec, Scene, Task
from armmind.scenes.common import free
from armmind.scenes.handling import body_top_z, items, place, surface_z

BOWL_XY, PLATE_XY = (0.62, 0.22), (0.62, -0.22)
BOWL_WALL = 0.01  # from the outer rim to the inner wall
DROP = 0.015  # release an item this far above the surface below it
PLATE_TOL = 0.01  # an item on the plate has its bottom this close to the plate floor
BOWL = ObjSpec("bowl", "mesh", label="white bowl", mass=0.8, mesh="Threshold_Bead_Cereal_Bowl_White", place=BOWL_XY)
PLATE = ObjSpec("plate", "mesh", label="turquoise plate", mass=0.8, mesh="Room_Essentials_Salad_Plate_Turquoise",
                place=PLATE_XY)
BOTTLE = ObjSpec("bottle", "mesh", label="white bottle", mass=0.15, mesh="Krill_Oil", random_yaw=True)
PURPLE = ObjSpec("purple_box", "mesh", label="purple box", mass=0.12, mesh="JarroSil_Activated_Silicon",
                 random_yaw=True)
YELLOW = ObjSpec("yellow_box", "mesh", label="yellow box", mass=0.12, mesh="Ultra_JarroDophilus",
                 random_yaw=True)
OBJECTS = (BOWL, PLATE, BOTTLE, PURPLE, YELLOW)
TEXT = (f"The bowl stands at x {BOWL_XY[0]:.2f}, y {BOWL_XY[1]:.2f} and the plate at x {PLATE_XY[0]:.2f}, "
        f"y {PLATE_XY[1]:.2f}. Both are heavy.")


def _xy_dist(w, a: str, b: str) -> float:
    return float(np.hypot(*(w.object_pos(a)[:2] - w.object_pos(b)[:2])))


def in_bowl(w, name: str) -> bool:
    """The item is inside the bowl rim, not above it, and the gripper does not touch it."""
    info, z = w.object_info("bowl"), w.object_pos("bowl")[2]
    inner = min(info.width, info.depth) / 2 - BOWL_WALL
    bottom, rim = z + info.bottom, z + info.bottom + info.height
    return bool(_xy_dist(w, name, "bowl") < inner - 0.01 and bottom < w.object_bottom(name) < rim + 0.03
                and free(w, name))


def on_plate(w, name: str) -> bool:
    """The item stands on the plate floor, inside the rim, and the gripper does not touch it.
    An item on the rim is 1.6 cm too high."""
    info, centre = w.object_info("plate"), w.object_pos("plate")
    radius, floor = min(info.width, info.depth) / 2, body_top_z(w, "plate", centre[:2])
    return bool(_xy_dist(w, name, "plate") < radius - 0.015 and abs(w.object_bottom(name) - floor) < PLATE_TOL
                and free(w, name))


def _to_bowl(w, name: str):
    xy = w.object_pos("bowl")[:2]
    return place(w, name, xy, surface_z(w, xy) + DROP)


def _to_plate(w, name: str, dy: float = 0.0):
    xy = w.object_pos("plate")[:2] + np.array([0.0, dy])
    return place(w, name, xy, surface_z(w, xy) + DROP)


def _clear_plan(w):
    return [*_to_bowl(w, "bottle"), *_to_plate(w, "purple_box", 0.05), *_to_plate(w, "yellow_box", -0.05)]


KITCHEN = Scene(
    name="kitchen", title="Kitchen", objects=OBJECTS, fixtures_xml="", fixtures_text=TEXT,
    region=(0.34, 0.58, -0.16, 0.16), min_gap=0.14,
    tasks={t.name: t for t in (
        Task("bottle_in_bowl", f"Put the {BOTTLE.label} in the bowl.", lambda w: in_bowl(w, "bottle"),
             lambda w: _to_bowl(w, "bottle")),
        Task("purple_box_on_plate", f"Put the {PURPLE.label} on the plate.", lambda w: on_plate(w, "purple_box"),
             lambda w: _to_plate(w, "purple_box")),
        Task("clear_table", "Put every item in the bowl or on the plate.",
             lambda w: all(in_bowl(w, o.name) or on_plate(w, o.name) for o in items(KITCHEN)), _clear_plan),
    )},
)
