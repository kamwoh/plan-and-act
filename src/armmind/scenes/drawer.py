"""A cabinet with a drawer that slides out toward the robot, and a red cube on the table."""
from __future__ import annotations

import numpy as np

from armmind.scenes.base import ObjSpec, Scene, Task
from armmind.scenes.common import free
from armmind.types import Action

RED = ObjSpec("red_cube", "box", (0.02, 0.02, 0.02), "0.9 0.1 0.1 1", "red cube, 4 cm")

# The cabinet: a block below the drawer, two sides, a back and a top. The open face is at x FRONT, toward the robot.
CAB_X, CAB_HX, CAB_HY, CAB_H = 0.62, 0.09, 0.13, 0.16
FRONT = CAB_X - CAB_HX  # 0.53
# The drawer frame is the centre of its floor top, at slide 0 (closed). The slide is the distance it is out.
DRAWER_X, FLOOR_Z = 0.60, 0.09
IN_HX, IN_HY, DEPTH, WALL = 0.06, 0.085, 0.05, 0.005
PANEL_T = 0.01  # front panel thickness
HANDLE_DX, HANDLE_R, HANDLE_HY = -0.11, 0.008, 0.04  # the bar: 4 cm in front of the panel, at the drawer top
MAX_OPEN = 0.15
WOOD, DARK = "0.62 0.45 0.3 1", "0.35 0.25 0.17 1"
LIFT_Z = 0.25  # above the cabinet top
HOOK_YAW = 90.0  # fingers along y: the closed gripper fits between the bar and the panel


def _box(name: str, x: float, y: float, z: float, hx: float, hy: float, hz: float, rgba: str) -> str:
    return (f'      <geom name="{name}" type="box" pos="{x:g} {y:g} {z:g}" '
            f'size="{hx:g} {hy:g} {hz:g}" rgba="{rgba}"/>')


BASE_TOP, TOP_BOTTOM = 0.08, CAB_H - 0.01  # the drawer cavity is between them, 5 mm clear of the drawer
_SIDE_Z, _SIDE_HZ = (BASE_TOP + TOP_BOTTOM) / 2, (TOP_BOTTOM - BASE_TOP) / 2  # the sides and the back
_PX = -IN_HX - PANEL_T / 2  # panel centre in the drawer frame
FIXTURES = "\n".join([
    f'    <body name="cabinet" pos="{CAB_X:g} 0 0">',
    _box("cabinet_base", 0, 0, BASE_TOP / 2, CAB_HX, CAB_HY, BASE_TOP / 2, WOOD),
    _box("cabinet_top", 0, 0, CAB_H - 0.005, CAB_HX, CAB_HY, 0.005, WOOD),
    _box("cabinet_left", 0, -CAB_HY + 0.005, _SIDE_Z, CAB_HX, 0.005, _SIDE_HZ, WOOD),
    _box("cabinet_right", 0, CAB_HY - 0.005, _SIDE_Z, CAB_HX, 0.005, _SIDE_HZ, WOOD),
    _box("cabinet_back", CAB_HX - 0.005, 0, _SIDE_Z, 0.005, CAB_HY - 0.01, _SIDE_HZ, WOOD),
    "    </body>",
    # The drawer floats on its slide and keeps clear of the cabinet, so the two never touch.
    f'    <body name="drawer" pos="{DRAWER_X:g} 0 {FLOOR_Z:g}">',
    f'      <joint name="drawer" type="slide" axis="-1 0 0" range="0 {MAX_OPEN:g}" damping="8" frictionloss="1"/>',
    _box("drawer_floor", 0.0025, 0, -WALL / 2, IN_HX + 0.0025, IN_HY + WALL, WALL / 2, WOOD),
    _box("drawer_back", IN_HX + WALL / 2, 0, DEPTH / 2, WALL / 2, IN_HY + WALL, DEPTH / 2, WOOD),
    _box("drawer_left", 0, -IN_HY - WALL / 2, DEPTH / 2, IN_HX, WALL / 2, DEPTH / 2, WOOD),
    _box("drawer_right", 0, IN_HY + WALL / 2, DEPTH / 2, IN_HX, WALL / 2, DEPTH / 2, WOOD),
    _box("drawer_front", _PX, 0, (DEPTH - WALL) / 2, PANEL_T / 2, IN_HY + 0.01, (DEPTH + WALL) / 2, WOOD),
    (f'      <geom name="drawer_handle" type="capsule" fromto="{HANDLE_DX:g} {-HANDLE_HY:g} {DEPTH:g} '
    f'{HANDLE_DX:g} {HANDLE_HY:g} {DEPTH:g}" size="{HANDLE_R:g}" rgba="{DARK}"/>'),
    *(f'      <geom name="drawer_post_{i}" type="cylinder" fromto="{_PX - PANEL_T / 2:g} {y:g} {DEPTH:g} '
      f'{HANDLE_DX:g} {y:g} {DEPTH:g}" size="0.005" rgba="{DARK}"/>' for i, y in enumerate((-HANDLE_HY, HANDLE_HY))),
    '      <inertial pos="0 0 0.02" mass="0.4" diaginertia="0.002 0.002 0.002"/>',
    "    </body>",
])
TEXT = (f"A wooden cabinet at x {FRONT:.2f} to {CAB_X + CAB_HX:.2f}, y {-CAB_HY:.2f} to {CAB_HY:.2f}, "
        f"{CAB_H * 100:.0f} cm tall. Its drawer slides out toward the robot (-x), up to {MAX_OPEN * 100:.0f} cm. "
        f"The drawer handle is a dark bar along y, {2 * HANDLE_HY * 100:.0f} cm long, at z {FLOOR_Z + DEPTH:.2f}, "
        f"{-HANDLE_DX - IN_HX - PANEL_T:.2f} m in front of the drawer front. When the drawer is closed, "
        f"the bar is at x {DRAWER_X + HANDLE_DX:.2f}. The inside of the drawer is {2 * IN_HX * 100:.0f} by "
        f"{2 * IN_HY * 100:.0f} cm and {DEPTH * 100:.0f} cm deep, with the floor at z {FLOOR_Z:.2f}.")


def slide(w) -> float:
    return w.joint_pos("drawer")


def handle_pos(w) -> np.ndarray:
    return w.data.body("drawer").xpos + np.array([HANDLE_DX, 0.0, DEPTH])


def in_drawer(w, name: str) -> bool:
    """True when the object's centre is inside the drawer's inner area."""
    d = w.data.body("drawer")
    x, y, z = d.xmat.reshape(3, 3).T @ (w.object_pos(name) - d.xpos)
    return bool(abs(x) <= IN_HX and abs(y) <= IN_HY and 0 < z < DEPTH)


def _open(w) -> bool:
    return slide(w) >= 0.10


def _closed(w) -> bool:
    return slide(w) < 0.01 and "drawer" not in w.gripper_contacts()


def _cube_in(w) -> bool:
    return in_drawer(w, "red_cube") and free(w, "red_cube") and slide(w) >= 0.06


def _at(p, z: float, yaw: float | None = None) -> Action:
    return Action("move_to", (float(p[0]), float(p[1]), z), yaw=yaw)


GAP = (-HANDLE_DX - IN_HX - PANEL_T + HANDLE_R) / 2  # from the bar axis to the middle of the bar-panel gap


def _pull_plan(w, out: float = 0.14) -> list[Action]:
    """Hook the closed gripper between the bar and the panel, pull it out along -x, then lift."""
    h = handle_pos(w)
    hook = h + np.array([GAP, 0.0, 0.0])
    pulled = hook - np.array([out - slide(w), 0.0, 0.0])
    return [Action("close"), _at(hook, LIFT_Z, HOOK_YAW), _at(hook, h[2]), _at(pulled, h[2]), _at(pulled, LIFT_Z)]


def _push_plan(w) -> list[Action]:
    """Put the closed gripper in front of the bar and push it in along +x. Then back away."""
    h = handle_pos(w)
    start = h - np.array([0.05, 0.0, 0.0])
    end = np.array([DRAWER_X + HANDLE_DX - HANDLE_R - 0.008, 0.0, h[2]])  # a little past the closed position
    return [Action("close"), _at(start, LIFT_Z, HOOK_YAW), _at(start, h[2]), _at(end, h[2]),
            _at(end - np.array([0.06, 0, 0]), h[2]), _at(end - np.array([0.06, 0, 0]), LIFT_Z)]


def _cube_plan(w) -> list[Action]:
    """Open the drawer, then carry the cube in. The gripper turns to yaw 90 so the open fingers fit inside."""
    p = w.object_pos("red_cube")
    centre = np.array([DRAWER_X - 0.135, 0.0])  # the drawer centre after the pull: the hook has 5 mm of slack
    return [*_pull_plan(w), Action("open"), _at(p, 0.15, 0.0), _at(p, float(p[2])), Action("close"),
            _at(p, LIFT_Z), _at(centre, LIFT_Z, HOOK_YAW), _at(centre, FLOOR_Z + 0.025), Action("open"),
            _at(centre, LIFT_Z)]


DRAWER = Scene(
    name="drawer", title="Drawer", objects=(RED,), fixtures_xml=FIXTURES, fixtures_text=TEXT,
    region=(0.35, 0.45, -0.25, -0.12), min_gap=0.10, joints=("drawer",),
    tasks={t.name: t for t in (
        Task("open_drawer", "Open the drawer at least 10 cm.", _open, _pull_plan),
        Task("close_drawer", "Close the drawer.", _closed, _push_plan, initial_qpos={"drawer": 0.12}),
        Task("cube_in_drawer", "Open the drawer and put the red cube inside it.", _cube_in, _cube_plan),
    )},
)
