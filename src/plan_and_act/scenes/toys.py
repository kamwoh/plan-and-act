"""Four small scanned toys and the grey tray of the tray scene: put toys in the tray."""
from __future__ import annotations

from plan_and_act.scenes.base import ObjSpec, Scene, Task
from plan_and_act.scenes.handling import items, place
from plan_and_act.scenes.tray import CX, CY, FIXTURES, FLOOR_HALF_H, SHAPES, TEXT, in_tray

DROP = 0.004  # release a toy this far above the tray floor
TOYS = (
    ObjSpec("orange_robot", "mesh", label="orange robot figure", mass=0.05, mesh="Android_Figure_Orange",
            random_yaw=True),
    ObjSpec("panda", "mesh", label="black and white panda figure", mass=0.05, mesh="Android_Figure_Panda",
            random_yaw=True),
    # The folder name says triceratops, but the scan is a green robot figure.
    ObjSpec("green_robot", "mesh", label="green robot figure", mass=0.05, mesh="Great_Dinos_Triceratops_Toy",
            random_yaw=True),
    ObjSpec("pink_toy", "mesh", label="pink rubber toy", mass=0.05, mesh="Kong_Puppy_Teething_Rubber_Small_Pink",
            random_yaw=True),
)
# One spot for each toy in the tray, in two rows 8 cm apart along y, in the order of placing. The gripper
# turns to yaw 90 before it lets go: the fingers open along y, toward the other row, and stay clear of it.
# The long side of a toy is along x. Each row has a long toy and a short toy. The short toys go first, on
# the far side of each row.
SPOTS = {"pink_toy": (CX + 0.037, CY - 0.04), "green_robot": (CX - 0.028, CY - 0.04),
         "panda": (CX + 0.034, CY + 0.04), "orange_robot": (CX - 0.03, CY + 0.04)}
RELEASE_YAW = 90.0

def _to_tray(w, name: str, xy=(CX, CY), yaw: float | None = None):
    return place(w, name, xy, 2 * FLOOR_HALF_H + DROP, yaw)


def _all_plan(w):
    return [a for name, xy in SPOTS.items() for a in _to_tray(w, name, xy, RELEASE_YAW)]


BRIEF = 'A grey tray with low walls.'  # the fixed items for a prompt with names only: no positions or sizes

TOYS_SCENE = Scene(
    name="toys", fixtures_brief=BRIEF, title="Toys", objects=TOYS, fixtures_xml=FIXTURES, fixtures=SHAPES,
    fixtures_text=TEXT,
    region=(0.38, 0.65, -0.25, 0.12), min_gap=0.10,
    tasks={t.name: t for t in (
        Task("panda_in_tray", f"Put the {TOYS[1].label} in the tray.", lambda w: in_tray(w, "panda"),
             lambda w: _to_tray(w, "panda")),
        Task("all_toys_in_tray", "Put all toys in the tray.",
             lambda w: all(in_tray(w, o.name) for o in items(TOYS_SCENE)), _all_plan),
    )},
)
