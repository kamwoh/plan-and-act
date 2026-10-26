"""Three cubes and a fixed grey tray: put cubes in the tray."""
from __future__ import annotations

from plan_and_act.scenes.base import Scene, Task, box
from plan_and_act.scenes.common import THREE_CUBES, free, put

CX, CY = 0.50, 0.28  # tray centre
HALF, WALL, WALL_HALF_H, FLOOR_HALF_H = 0.07, 0.005, 0.0125, 0.0025
INNER = HALF - WALL  # 0.065: the inner area is the centre +- this
MARGIN = 0.01
TOP_Z = 0.04  # an object in the tray has its lowest point below this: on the floor, not on top of another
RELEASE_Z = 0.06
GREY = "0.55 0.57 0.6 1"
# Three spots inside the tray, at least 6 cm apart, so that the cubes do not touch.
SPOTS = {"red_cube": (CX - 0.03, CY - 0.03), "blue_cube": (CX - 0.03, CY + 0.03), "green_cube": (CX + 0.03, CY)}


def _box(name: str, dx: float, dy: float, z: float, hx: float, hy: float, hz: float) -> str:
    return (f'      <geom name="{name}" type="box" pos="{dx:g} {dy:g} {z:g}" '
            f'size="{hx:g} {hy:g} {hz:g}" rgba="{GREY}"/>')


def _wall(name: str, dx: float, dy: float, hx: float, hy: float) -> str:
    return _box(name, dx, dy, WALL_HALF_H, hx, hy, WALL_HALF_H)


_E = HALF - WALL / 2  # wall centre distance from the tray centre
FIXTURES = "\n".join([
    f'    <body name="tray" pos="{CX:g} {CY:g} 0">',
    _box("tray_floor", 0, 0, FLOOR_HALF_H, HALF, HALF, FLOOR_HALF_H),
    _wall("tray_wall_near", -_E, 0, WALL / 2, HALF),
    _wall("tray_wall_far", _E, 0, WALL / 2, HALF),
    _wall("tray_wall_left", 0, -_E, HALF, WALL / 2),
    _wall("tray_wall_right", 0, _E, HALF, WALL / 2),
    "    </body>",
])
SHAPES = (
    box("tray_floor", (CX, CY, FLOOR_HALF_H), (2 * HALF, 2 * HALF, 2 * FLOOR_HALF_H), GREY),
    box("tray_wall_near", (CX - _E, CY, WALL_HALF_H), (WALL, 2 * HALF, 2 * WALL_HALF_H), GREY),
    box("tray_wall_far", (CX + _E, CY, WALL_HALF_H), (WALL, 2 * HALF, 2 * WALL_HALF_H), GREY),
    box("tray_wall_left", (CX, CY - _E, WALL_HALF_H), (2 * HALF, WALL, 2 * WALL_HALF_H), GREY),
    box("tray_wall_right", (CX, CY + _E, WALL_HALF_H), (2 * HALF, WALL, 2 * WALL_HALF_H), GREY),
)
TEXT = (f"A grey tray: inner area x {CX - INNER:.3f} to {CX + INNER:.3f}, y {CY - INNER:.3f} to {CY + INNER:.3f}, "
        f"walls {2 * WALL_HALF_H * 100:.1f} cm high.")


def in_tray(w, name: str) -> bool:
    """The object's centre is over the inner area, its lowest point is low, and the gripper does not touch it."""
    x, y, _ = w.object_pos(name)
    lim = INNER - MARGIN
    return bool(abs(x - CX) <= lim and abs(y - CY) <= lim and w.object_bottom(name) < TOP_Z and free(w, name))


def _red_plan(w):
    return put(w, "red_cube", (CX, CY), RELEASE_Z)


def _all_plan(w):
    return [a for name, xy in SPOTS.items() for a in put(w, name, xy, RELEASE_Z)]


BRIEF = 'A grey tray with low walls.'  # the fixed items for a prompt with names only: no positions or sizes

TRAY = Scene(
    name="tray", fixtures_brief=BRIEF, title="Tray", objects=THREE_CUBES, fixtures_xml=FIXTURES, fixtures_text=TEXT, fixtures=SHAPES,
    region=(0.40, 0.65, -0.25, 0.12), min_gap=0.10,
    tasks={t.name: t for t in (
        Task("red_in_tray", "Put the red cube in the tray.", lambda w: in_tray(w, "red_cube"), _red_plan),
        Task("all_in_tray", "Put all three cubes in the tray.",
             lambda w: all(in_tray(w, o.name) for o in THREE_CUBES), _all_plan),
    )},
)
