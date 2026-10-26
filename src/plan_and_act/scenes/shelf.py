"""Three cubes and a fixed two-level shelf: put cubes on the shelf levels.

The shelf is stepped: the top level is behind the bottom level and higher. So the hand reaches each level
from above, and the bottom level needs no room for the hand under the top level."""
from __future__ import annotations

from plan_and_act.scenes.base import Scene, Task, box
from plan_and_act.scenes.common import LIFT_Z, THREE_CUBES, free
from plan_and_act.types import Action

X0, X_MID, X1 = 0.55, 0.65, 0.75  # the bottom level is x X0 to X_MID, the top level is x X_MID to X1
HALF_Y, SIDE_T, BOARD_T, RISER_T = 0.20, 0.01, 0.01, 0.01  # the inner half width and the panel thicknesses
LOW_TOP, HIGH_TOP = 0.04, 0.10  # the board top heights
LEVELS = {"bottom": (X0, X_MID, LOW_TOP), "top": (X_MID + RISER_T, X1, HIGH_TOP)}  # name: (x_min, x_max, top z)
MARGIN = 0.01
CARRY_Z = 0.20  # high enough to carry a cube over the top level
YAW = 90.0  # fingers along y: the hand is thin along x, so it keeps clear of the riser
WOOD = "0.62 0.45 0.3 1"
# Places on the shelf for the scripted plans: (x, y) at the middle of a level.
LOW_X, HIGH_X = (X0 + X_MID) / 2, (X_MID + RISER_T + X1) / 2


def _box(name: str, x: float, y: float, z: float, hx: float, hy: float, hz: float) -> str:
    return (f'      <geom name="{name}" type="box" pos="{x:g} {y:g} {z:g}" '
            f'size="{hx:g} {hy:g} {hz:g}" rgba="{WOOD}"/>')


_CX, _HX = (X0 + X1) / 2, (X1 - X0) / 2
_SIDE_Y = HALF_Y + SIDE_T / 2
_LOW_X, _LOW_HX = (X0 + X_MID) / 2, (X_MID - X0) / 2
_HIGH_X, _HIGH_HX = (X_MID + X1) / 2, (X1 - X_MID) / 2
_RISER_Z, _RISER_HZ = (LOW_TOP + HIGH_TOP - BOARD_T) / 2, (HIGH_TOP - BOARD_T - LOW_TOP) / 2
_PARTS = {  # name: (centre, half size). The body is at the origin, so the geom centres are world points.
    "shelf_side_left": ((_CX, -_SIDE_Y, HIGH_TOP / 2), (_HX, SIDE_T / 2, HIGH_TOP / 2)),
    "shelf_side_right": ((_CX, _SIDE_Y, HIGH_TOP / 2), (_HX, SIDE_T / 2, HIGH_TOP / 2)),
    "shelf_bottom": ((_LOW_X, 0, LOW_TOP - BOARD_T / 2), (_LOW_HX, HALF_Y, BOARD_T / 2)),
    "shelf_riser": ((X_MID + RISER_T / 2, 0, _RISER_Z), (RISER_T / 2, HALF_Y, _RISER_HZ)),
    "shelf_top": ((_HIGH_X, 0, HIGH_TOP - BOARD_T / 2), (_HIGH_HX, HALF_Y, BOARD_T / 2)),
}
FIXTURES = "\n".join([
    '    <body name="shelf" pos="0 0 0">',
    *(_box(n, *c, *h) for n, (c, h) in _PARTS.items()),
    "    </body>",
])
SHAPES = tuple(box(n, c, tuple(2 * v for v in h), WOOD) for n, (c, h) in _PARTS.items())
TEXT = (f"A wooden shelf with two levels, open at the top. It is {2 * HALF_Y * 100:.0f} cm wide inside, "
        f"from y {-HALF_Y:.2f} to {HALF_Y:.2f}, between two side panels {HIGH_TOP * 100:.0f} cm high. "
        f"The bottom level is x {X0:.2f} to {X_MID:.2f}, with its surface at z {LOW_TOP:.2f}. "
        f"The top level is behind it, x {X_MID + RISER_T:.2f} to {X1:.2f}, with its surface at z {HIGH_TOP:.2f}.")


def on_level(w, name: str, level: str) -> bool:
    """The object's centre is over the level, its lowest point is on the level surface, and the gripper
    does not touch it."""
    x0, x1, top = LEVELS[level]
    x, y, _ = w.object_pos(name)
    bottom = w.object_bottom(name)
    return bool(x0 + MARGIN <= x <= x1 - MARGIN and abs(y) <= HALF_Y - MARGIN
                and top - 0.005 < bottom < top + 0.01 and free(w, name))


def on_shelf(w, name: str) -> bool:
    return any(on_level(w, name, level) for level in LEVELS)


def _at(p, z: float, yaw: float | None = None) -> Action:
    return Action("move_to", (float(p[0]), float(p[1]), float(z)), yaw=yaw)


def _put(w, name: str, xy, top: float) -> list[Action]:
    """Grasp the cube with the fingers along y, carry it above xy, lower it onto the level and let go."""
    p = w.object_pos(name)
    release = top + w.object_info(name).height / 2 + 0.01
    return [Action("open"), _at(p, LIFT_Z, YAW), _at(p, p[2]), Action("close"), _at(p, CARRY_Z),
            _at(xy, CARRY_Z), _at(xy, release), Action("open"), _at(xy, CARRY_Z)]


def _red_plan(w):
    return _put(w, "red_cube", (HIGH_X, 0.0), HIGH_TOP)


def _all_plan(w):
    return [*_put(w, "red_cube", (HIGH_X, 0.0), HIGH_TOP), *_put(w, "blue_cube", (LOW_X, -0.08), LOW_TOP),
            *_put(w, "green_cube", (LOW_X, 0.08), LOW_TOP)]


BRIEF = 'A wooden shelf with two levels, open at the top, between two side panels. The top level is behind the bottom level and higher.'  # the fixed items for a prompt with names only: no positions or sizes

SHELF = Scene(
    name="shelf", fixtures_brief=BRIEF, title="Shelf", objects=THREE_CUBES, fixtures_xml=FIXTURES, fixtures_text=TEXT, fixtures=SHAPES,
    region=(0.35, 0.49, -0.25, 0.25), min_gap=0.10,
    tasks={t.name: t for t in (
        Task("red_on_top_shelf", "Put the red cube on the top level of the shelf.",
             lambda w: on_level(w, "red_cube", "top"), _red_plan),
        Task("all_on_shelf", "Put all three cubes on the shelf. Any level is good.",
             lambda w: all(on_shelf(w, o.name) for o in THREE_CUBES), _all_plan),
    )},
)
