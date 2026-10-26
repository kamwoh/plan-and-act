"""A laptop on the table. Its lid turns on a hinge along y at the back of the base."""
from __future__ import annotations

import numpy as np

from plan_and_act.scenes.base import Scene, Task, box
from plan_and_act.types import Action

# The base. The hinge axis is along y, at the back top edge of the base.
BASE_X0, BASE_X1, BASE_HY, BASE_H = 0.40, 0.62, 0.15, 0.02
HINGE_X, HINGE_Z = BASE_X1, BASE_H
LID_L, LID_HY, LID_T = 0.21, 0.145, 0.01  # lid length from the hinge, half width, thickness
TAB_T, TAB_HY, TAB_H = 0.01, 0.05, 0.025  # a small tab up at the front edge of the lid
MAX_OPEN = 115.0  # degrees: the MJCF range is in degrees, qpos is in radians
OPEN_AT, CLOSED_AT = np.radians(70.0), np.radians(5.0)
SILVER, DARK = "0.75 0.76 0.78 1", "0.2 0.2 0.22 1"
LIFT_Z = 0.25
CLEAR_Z = 0.30  # well above the top edge of the open lid
YAW = 90.0  # fingers along y: the closed gripper is thin along x

# Contact bits: the gripper and the table use 1. The base (2/1) and the lid (4/1) touch the gripper,
# but not each other, so that the lid can rest on its joint limit.
FIXTURES = "\n".join([
    f'    <body name="laptop" pos="{(BASE_X0 + BASE_X1) / 2:g} 0 {BASE_H / 2:g}">',
    (f'      <geom name="laptop_base" type="box" size="{(BASE_X1 - BASE_X0) / 2:g} {BASE_HY:g} {BASE_H / 2:g}" '
     f'rgba="{SILVER}" contype="2" conaffinity="1"/>'),
    "    </body>",
    f'    <body name="laptop_lid" pos="{HINGE_X:g} 0 {HINGE_Z:g}">',
    # The armature stops a slow creep of the lid under gravity.
    (f'      <joint name="laptop_lid" type="hinge" axis="0 1 0" range="0 {MAX_OPEN:g}" damping="0.05" '
     'armature="0.05" frictionloss="0.2"/>'),
    (f'      <geom name="laptop_lid" type="box" pos="{-LID_L / 2:g} 0 {LID_T / 2:g}" '
     f'size="{LID_L / 2:g} {LID_HY:g} {LID_T / 2:g}" rgba="{SILVER}" contype="4" conaffinity="1"/>'),
    (f'      <geom name="laptop_tab" type="box" pos="{-LID_L + TAB_T / 2:g} 0 {LID_T + TAB_H / 2:g}" '
     f'size="{TAB_T / 2:g} {TAB_HY:g} {TAB_H / 2:g}" rgba="{DARK}" contype="4" conaffinity="1"/>'),
    f'      <inertial pos="{-LID_L / 2:g} 0 {LID_T / 2:g}" mass="0.08" diaginertia="0.0006 0.0003 0.0009"/>',
    "    </body>",
])
SHAPES = (  # the lid closed
    box("laptop_base", ((BASE_X0 + BASE_X1) / 2, 0, BASE_H / 2), (BASE_X1 - BASE_X0, 2 * BASE_HY, BASE_H), SILVER),
    box("laptop_lid", (HINGE_X - LID_L / 2, 0, HINGE_Z + LID_T / 2), (LID_L, 2 * LID_HY, LID_T), SILVER),
    box("laptop_tab", (HINGE_X - LID_L + TAB_T / 2, 0, HINGE_Z + LID_T + TAB_H / 2), (TAB_T, 2 * TAB_HY, TAB_H),
        DARK),
)
TEXT = (f"A silver laptop. Its base is at x {BASE_X0:.2f} to {BASE_X1:.2f}, y {-BASE_HY:.2f} to {BASE_HY:.2f}, "
        f"{BASE_H * 100:.0f} cm tall. The lid is {LID_L * 100:.0f} cm long and {LID_T * 100:.0f} cm thick. "
        f"It turns on a hinge along y at x {HINGE_X:.2f}, z {HINGE_Z:.2f}. The lid opens up to about "
        f"{MAX_OPEN:.0f} degrees. When the lid is closed, it lies flat on the base and its front edge "
        f"is at x {HINGE_X - LID_L:.2f}. A dark tab, {2 * TAB_HY * 100:.0f} cm wide and {TAB_H * 100:.1f} cm tall, "
        "stands up at that edge, at the lid centre (y 0). The lid stays where you leave it. To open it, press the "
        "closed fingers on the front of the tab and move them up and back along the arc of the lid edge.")


def lid_angle(w) -> float:
    return w.joint_pos("laptop_lid")


def _open(w) -> bool:
    return bool(lid_angle(w) >= OPEN_AT)


def _closed(w) -> bool:
    return bool(lid_angle(w) < CLOSED_AT) and "laptop_lid" not in w.gripper_contacts()


def _at(x: float, z: float, yaw: float | None = None) -> Action:
    return Action("move_to", (float(x), 0.0, float(z)), yaw=yaw)


FINGER_HX = 0.011  # half the thickness of the closed fingers along x, at yaw 90
PRESS = 0.01  # the fingers go this far into the lid, so that they keep a hold on it


def _under_edge(angle_deg: float, press: float = PRESS) -> Action:
    """The tcp for a lid angle: the front face of the fingers at the lower front edge of the lid, and the
    fingertips 5 mm below that edge. At a low angle the fingertips stay 1 mm above the lid top."""
    a = np.radians(angle_deg)
    x, z = HINGE_X - LID_L * np.cos(a), HINGE_Z + LID_L * np.sin(a)
    tip = max(z - 0.005, HINGE_Z + LID_T + 0.001)
    return _at(x - FINGER_HX + press, tip + 0.01)


OPEN_ANGLES = (0, 3, 6, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95)  # degrees: the lid angles of the push
# Between these lid angles the lid holds the fingers back about 2 cm. There the fingers aim less far into it,
# so that each move ends within the reach tolerance.
HEAVY, HEAVY_EASE = (35, 65), 0.012


def _open_plan(w) -> list[Action]:
    """Press the closed fingers on the tab and move them up along the arc of the lower front edge of the lid.
    Friction lifts the lid first. Then the fingers push its lower front edge. Stop at about 95 degrees,
    where the lid leans back, and lift the fingers over its top edge."""
    start, top = HINGE_X - LID_L - 0.04, _under_edge(95).xyz[0]
    return [Action("close", expect="nothing"), _at(start, LIFT_Z, YAW), _at(start, HINGE_Z + LID_T + 0.011),
            *(_under_edge(a, PRESS - HEAVY_EASE if HEAVY[0] <= a <= HEAVY[1] else PRESS) for a in OPEN_ANGLES),
            _at(top, CLEAR_Z), _at(BASE_X0 - 0.05, CLEAR_Z)]


TIP_STEPS = (-0.04,)  # m from the hinge along x: the pushes that tip the open lid forward
TIP_Z = 0.19  # the height of those pushes: high on the lid, where a push tips it with less force
S_PUSH = 0.15  # the distance from the hinge along the lid where the fingers push it down


def _on_lid(angle_deg: float, gap: float) -> Action:
    """The tcp above the outer face of the lid at the given angle, S_PUSH from the hinge. The fingertips
    are gap above the face."""
    a = np.radians(angle_deg)
    along, out = np.array([-np.cos(a), np.sin(a)]), np.array([np.sin(a), np.cos(a)])
    x, z = np.array([HINGE_X, HINGE_Z]) + S_PUSH * along + (LID_T + gap) * out + np.array([0.0, 0.01])
    return _at(x, z)


def _close_plan(w) -> list[Action]:
    """Push the top of the lid toward -x from behind until it leans forward. Then press it down along
    its arc, with the fingertips a little inside the outer face."""
    behind = HINGE_X + 0.09
    return [Action("close", expect="nothing"), _at(behind, LIFT_Z, YAW), _at(behind, TIP_Z),
            *(_at(HINGE_X + dx, TIP_Z) for dx in TIP_STEPS),
            *(_on_lid(a, -0.005) for a in range(60, -1, -10)),
            _at(HINGE_X - S_PUSH, LIFT_Z)]


BRIEF = 'A silver laptop. Its lid turns on a hinge at the back of the base. A dark tab stands up at the front edge of the lid. To open the lid, press the closed fingers on the front of the tab and move them up and back along the arc of the lid edge.'  # the fixed items for a prompt with names only: no positions or sizes

LAPTOP = Scene(
    name="laptop", fixtures_brief=BRIEF, title="Laptop", objects=(), fixtures_xml=FIXTURES, fixtures_text=TEXT, fixtures=SHAPES,
    region=(0.30, 0.34, -0.38, -0.34), min_gap=0.1, joints={"laptop_lid": "laptop lid angle {:.2f} rad (0 = closed)"},
    tasks={t.name: t for t in (
        Task("open_laptop", "Open the laptop lid to at least 70 degrees.", _open, _open_plan),
        Task("close_laptop", "Close the laptop lid.", _closed, _close_plan, initial_qpos={"laptop_lid": 1.75}),
    )},
)
