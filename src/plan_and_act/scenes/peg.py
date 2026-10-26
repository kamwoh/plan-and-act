"""Peg in hole: a fixed grey block with two square holes, a red square peg and a blue round peg."""
from __future__ import annotations

import numpy as np

from plan_and_act.scenes.base import ObjSpec, Scene, Task, box
from plan_and_act.scenes.common import free
from plan_and_act.types import Action

PEG_HALF, PEG_HALF_H = 0.012, 0.05  # each peg is 2.4 cm wide and 10 cm tall
RED = ObjSpec("red_peg", "box", (PEG_HALF, PEG_HALF, PEG_HALF_H), "0.9 0.1 0.1 1",
              "red square peg, standing, 2.4 cm wide and 10 cm tall", random_yaw=True)
BLUE = ObjSpec("blue_peg", "cylinder", (PEG_HALF, PEG_HALF_H), "0.1 0.2 0.9 1",
               "blue round peg, standing, 2.4 cm wide and 10 cm tall")

# The block: four walls around two square holes, which go through to the table. The holes are 6 mm wider
# than a peg on each side, so that a peg goes in when the arm is up to 5 mm off.
BX, HOLE, WALL, BLOCK_H = 0.66, 0.018, 0.015, 0.04
HOLES = {"left": -0.04, "right": 0.04}  # hole name -> y of its centre
_MID = HOLES["right"] - HOLE  # the half width of the wall between the holes
_END = HOLES["right"] + HOLE + WALL / 2  # the y of an end wall centre
_SIDE = HOLE + WALL / 2  # the x offset of the near and far walls
_LEN = _END + WALL / 2  # the half length of the block along y
GREY = "0.55 0.57 0.6 1"
CARRY_Z = 0.22  # a carried peg clears the block and a peg in the other hole
GRIP_UP = 2 * PEG_HALF_H - 0.035  # grasp a peg 3.5 cm below its top, so that the hand stays clear of it
RELEASE_BOTTOM = 0.005  # let go when the peg bottom is this high: 3.5 cm deep in the hole
DEEP = BLOCK_H - 0.02  # a peg in a hole has its bottom below this
UPRIGHT = np.cos(np.radians(20))  # the peg axis is within 20 degrees of vertical


def _box(name: str, x: float, y: float, hx: float, hy: float) -> str:
    return (f'      <geom name="{name}" type="box" pos="{x:g} {y:g} {BLOCK_H / 2:g}" '
            f'size="{hx:g} {hy:g} {BLOCK_H / 2:g}" rgba="{GREY}"/>')


WALLS = (  # (name, x, y, half x, half y) in the block frame
    ("block_near", -_SIDE, 0.0, WALL / 2, _LEN),
    ("block_far", _SIDE, 0.0, WALL / 2, _LEN),
    ("block_left", 0.0, -_END, HOLE, WALL / 2),
    ("block_middle", 0.0, 0.0, HOLE, _MID),
    ("block_right", 0.0, _END, HOLE, WALL / 2),
)
FIXTURES = "\n".join([
    f'    <body name="block" pos="{BX:g} 0 0">',
    *(_box(*wall) for wall in WALLS),
    "    </body>",
])
SHAPES = tuple(box(name, (BX + x, y, BLOCK_H / 2), (2 * hx, 2 * hy, BLOCK_H), GREY) for name, x, y, hx, hy in WALLS)
TEXT = (f"A fixed grey block at x {BX - _SIDE - WALL / 2:.3f} to {BX + _SIDE + WALL / 2:.3f}, y {-_LEN:.3f} to "
        f"{_LEN:.3f}, {BLOCK_H * 100:.0f} cm tall. It has two square holes that go down to the table, each one "
        f"{2 * HOLE * 100:.1f} cm wide: "
        + " and ".join(f"the {h} hole at x {BX:.2f}, y {y:.2f}" for h, y in HOLES.items())
        + ". The holes have square sides along x and y.")


def in_hole(w, name: str, hole: str) -> bool:
    """The peg stands upright in the hole, its bottom is deep in it, and the gripper does not touch it."""
    x, y, _ = w.object_pos(name)
    inside = abs(x - BX) < HOLE and abs(y - HOLES[hole]) < HOLE
    upright = w.data.body(name).xmat.reshape(3, 3)[2, 2] > UPRIGHT
    return bool(inside and upright and w.object_bottom(name) < DEEP and free(w, name))


def _in_any(w, name: str) -> str | None:
    return next((h for h in HOLES if in_hole(w, name, h)), None)


def _red_in(w) -> bool:
    return _in_any(w, "red_peg") is not None


def _both_in(w) -> bool:
    red, blue = _in_any(w, "red_peg"), _in_any(w, "blue_peg")
    return red is not None and blue is not None and red != blue


def _at(p, z: float, yaw: float | None = None) -> Action:
    return Action("move_to", (float(p[0]), float(p[1]), float(z)), yaw=yaw)


def insert(w, name: str, hole: str) -> list[Action]:
    """Grasp the peg below its top, carry it above the hole and lower it in. A square peg is grasped across
    two sides with the smallest turn. The gripper then turns back to yaw 0, so the peg sides line up with
    the hole."""
    p = w.object_pos(name)
    yaw = (w.object_yaw(name) + 45.0) % 90.0 - 45.0 if name == "red_peg" else 0.0
    grip = w.object_bottom(name) + GRIP_UP
    to = (BX, HOLES[hole])
    return [Action("open"), _at(p, CARRY_Z, yaw), _at(p, grip), Action("close"), _at(p, CARRY_Z),
            _at(to, CARRY_Z, 0.0), _at(to, RELEASE_BOTTOM + GRIP_UP), Action("open"), _at(to, CARRY_Z)]


def _red_plan(w) -> list[Action]:
    return insert(w, "red_peg", "left")


def _both_plan(w) -> list[Action]:
    return [*insert(w, "red_peg", "left"), *insert(w, "blue_peg", "right")]


BRIEF = 'A fixed grey block with two square holes that go down to the table: a left hole and a right hole.'  # the fixed items for a prompt with names only: no positions or sizes

PEG = Scene(
    name="peg", fixtures_brief=BRIEF, title="Peg in hole", objects=(RED, BLUE), fixtures_xml=FIXTURES, fixtures_text=TEXT, fixtures=SHAPES,
    region=(0.38, 0.56, -0.28, 0.28), min_gap=0.10,
    tasks={t.name: t for t in (
        Task("insert_peg", "Put the red square peg upright into a hole of the grey block and let go of it.",
             _red_in, _red_plan),
        Task("fill_both_holes", "Put the red peg into one hole of the grey block and the blue peg into the other "
             "hole. Both pegs must stand upright, and the gripper must let go of them.", _both_in, _both_plan),
    )},
)
