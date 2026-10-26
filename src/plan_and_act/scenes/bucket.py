"""A small bucket with an upright handle, and a food box: put the food in the bucket and lift the bucket.

The bucket is a free body in fixtures_xml, not an ObjSpec, because it has several geoms. It always starts at
the same place. Its geoms are in SHAPES at that start pose."""
from __future__ import annotations

import numpy as np

from plan_and_act.scenes.base import ObjSpec, Scene, Task, box
from plan_and_act.scenes.common import LIFT_Z, above, free, grasp
from plan_and_act.types import Action

FOOD = ObjSpec("food_box", "box", (0.0175, 0.0175, 0.02), "0.95 0.6 0.15 1",
               "orange food box, 3.5 cm wide and 4 cm tall", mass=0.05)

CX, CY = 0.62, 0.20  # the bucket centre at the start
HALF, WALL, FLOOR_T, RIM = 0.06, 0.005, 0.005, 0.05  # outer half width, wall and floor thickness, wall top
INNER = HALF - WALL  # 0.055: the inner area is the centre +- this
BAR_Z, BAR_R = FLOOR_T + 0.08, 0.0125  # the handle bar along y, 8 cm above the floor and 2.5 cm thick
POST_HX = 0.004  # the two posts hold the bar ends on top of the left and right walls
MARGIN = 0.005
DROP_Z = 0.12  # the tcp height to let go of the food: the fingertips stay above the bar
LIFT_TO = 0.20  # the tcp height after the lift: the bucket bottom is then about 12 cm up
BUCKET_MIN = 0.05  # a lifted bucket has its bottom at least this high
PART_MASS = 0.012  # kg for each box part: the bucket is about 0.09 kg, light enough for the gripper
BLUE, DARK = "0.3 0.5 0.75 1", "0.25 0.25 0.28 1"


def _box(name: str, dx: float, dy: float, z: float, hx: float, hy: float, hz: float, rgba: str = BLUE) -> str:
    return (f'      <geom name="{name}" type="box" pos="{dx:g} {dy:g} {z:g}" '
            f'size="{hx:g} {hy:g} {hz:g}" rgba="{rgba}" mass="{PART_MASS:g}"/>')


_E = HALF - WALL / 2  # wall centre distance from the bucket centre
_POST_Z, _POST_HZ = (RIM + BAR_Z) / 2, (BAR_Z - RIM) / 2
FIXTURES = "\n".join([
    f'    <body name="bucket" pos="{CX:g} {CY:g} 0">',
    '      <freejoint name="bucket"/>',
    _box("bucket_floor", 0, 0, FLOOR_T / 2, HALF, HALF, FLOOR_T / 2),
    _box("bucket_wall_near", -_E, 0, RIM / 2, WALL / 2, HALF, RIM / 2),
    _box("bucket_wall_far", _E, 0, RIM / 2, WALL / 2, HALF, RIM / 2),
    _box("bucket_wall_left", 0, -_E, RIM / 2, HALF, WALL / 2, RIM / 2),
    _box("bucket_wall_right", 0, _E, RIM / 2, HALF, WALL / 2, RIM / 2),
    _box("bucket_post_left", 0, -_E, _POST_Z, POST_HX, WALL / 2, _POST_HZ, DARK),
    _box("bucket_post_right", 0, _E, _POST_Z, POST_HX, WALL / 2, _POST_HZ, DARK),
    (f'      <geom name="bucket_handle" type="capsule" fromto="0 {-_E:g} {BAR_Z:g} 0 {_E:g} {BAR_Z:g}" '
     f'size="{BAR_R:g}" rgba="{DARK}" mass="0.01" friction="1.5 0.01 0.001"/>'),
    "    </body>",
])
SHAPES = (
    box("bucket_floor", (CX, CY, FLOOR_T / 2), (2 * HALF, 2 * HALF, FLOOR_T), BLUE),
    box("bucket_wall_near", (CX - _E, CY, RIM / 2), (WALL, 2 * HALF, RIM), BLUE),
    box("bucket_wall_far", (CX + _E, CY, RIM / 2), (WALL, 2 * HALF, RIM), BLUE),
    box("bucket_wall_left", (CX, CY - _E, RIM / 2), (2 * HALF, WALL, RIM), BLUE),
    box("bucket_wall_right", (CX, CY + _E, RIM / 2), (2 * HALF, WALL, RIM), BLUE),
    box("bucket_post_left", (CX, CY - _E, _POST_Z), (2 * POST_HX, WALL, 2 * _POST_HZ), DARK),
    box("bucket_post_right", (CX, CY + _E, _POST_Z), (2 * POST_HX, WALL, 2 * _POST_HZ), DARK),
    # The bar is a capsule along y. A box around it is close enough.
    box("bucket_handle", (CX, CY, BAR_Z), (2 * BAR_R, 2 * (_E + BAR_R), 2 * BAR_R), DARK),
)
TEXT = (f"A blue bucket stands on the table. It is not fixed: the gripper can lift it by its handle. "
        f"At the start its inner area is x {CX - INNER:.3f} to {CX + INNER:.3f}, y {CY - INNER:.3f} to "
        f"{CY + INNER:.3f}, and its walls are {RIM * 100:.0f} cm high. The handle is a dark bar along y, "
        f"{2 * BAR_R * 100:.1f} cm thick, across the middle of the bucket at x {CX:.2f}, z {BAR_Z:.3f}. "
        f"Two posts on the left and right walls hold the bar ends.")


def bucket_pos(w) -> np.ndarray:
    return w.data.body("bucket").xpos.copy()


def in_bucket(w, name: str) -> bool:
    """The object's centre is over the inner area and below the rim, in the bucket frame."""
    b = w.data.body("bucket")
    x, y, z = b.xmat.reshape(3, 3).T @ (w.object_pos(name) - b.xpos)
    lim = INNER - MARGIN
    return bool(abs(x) <= lim and abs(y) <= lim and FLOOR_T < z < RIM)


def _food_in(w) -> bool:
    return in_bucket(w, "food_box") and free(w, "food_box")


def _lifted(w) -> bool:
    return bool(in_bucket(w, "food_box") and bucket_pos(w)[2] >= BUCKET_MIN)


def _food_plan(w) -> list[Action]:
    """Grasp the food, carry it over the bucket beside the bar and let it drop in."""
    p = w.object_pos("food_box")
    spot = bucket_pos(w)[:2] - np.array([0.035, 0.0])  # 3.5 cm in front of the bar, toward the robot
    return [*grasp(w, "food_box"), above(p, LIFT_Z), above(spot, LIFT_Z), above(spot, DROP_Z), Action("open"),
            above(spot, LIFT_Z)]


def _lift_plan(w) -> list[Action]:
    """Put the food in, then close the fingers across the bar and lift the bucket."""
    bar = bucket_pos(w)[:2]
    return [*_food_plan(w), Action("move_to", (float(bar[0]), float(bar[1]), LIFT_Z), yaw=0.0),
            above(bar, BAR_Z), Action("close"), above(bar, LIFT_TO)]


BRIEF = 'A blue bucket stands on the table. It is not fixed: the gripper can lift it by its handle. The handle is a dark bar along y, across the middle of the bucket, held by two posts.'  # the fixed items for a prompt with names only: no positions or sizes

BUCKET = Scene(
    name="bucket", fixtures_brief=BRIEF, title="Bucket", objects=(FOOD,), fixtures_xml=FIXTURES, fixtures_text=TEXT, fixtures=SHAPES,
    region=(0.38, 0.55, -0.25, 0.05), min_gap=0.10,
    tasks={t.name: t for t in (
        Task("food_in_bucket", "Put the food box in the bucket.", _food_in, _food_plan),
        Task("lift_bucket", "Put the food box in the bucket. Then lift the bucket by its handle at least 5 cm "
             "off the table, with the food box still inside.", _lifted, _lift_plan),
    )},
)
