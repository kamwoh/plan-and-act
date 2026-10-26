"""A stapler on the table. Its top arm turns on a hinge with a spring. A press gives an event."""
from __future__ import annotations

import math

from plan_and_act.scenes.base import Scene, Task, box, threshold_monitor
from plan_and_act.scenes.common import LIFT_Z
from plan_and_act.types import Action

# The stapler lies along x. The hinge is at the far end (+x), and the free front end points to the robot.
BASE_X0, BASE_X1, BASE_HY, BASE_H = 0.54, 0.70, 0.02, 0.015
POST_X0, POST_HY, POST_TOP = 0.67, 0.015, 0.039
HINGE_X, HINGE_Z = 0.685, 0.045
ARM_X0, ARM_HY, ARM_HZ = 0.54, 0.018, 0.006  # the arm runs from x ARM_X0 to the hinge, plus 5 mm
ARM_HX = (HINGE_X + 0.005 - ARM_X0) / 2
MAX_ANGLE, SPRINGREF = 0.15, -0.15  # rad: at MAX_ANGLE the arm front stops just above the base
FIRE, REARM = 0.10, 0.03  # rad
PRESS_X = 0.57  # the gripper pushes here, near the front of the arm
PRESS_Z = 0.035  # the fingertips are 1 cm below the TCP: this pushes the arm to its stop
BLACK, RED = "0.2 0.2 0.22 1", "0.8 0.1 0.12 1"

# Contact bits: the gripper and the table use 1. The base (2/1) and the arm (4/1) touch the gripper,
# but not each other, so that the arm can turn over the post. The spring holds the arm up against its stop
# at 0. A positive angle turns the front down. MJCF reads the hinge range and springref in degrees.
FIXTURES = "\n".join([
    '    <body name="stapler" pos="0 0 0">',
    (f'      <geom name="stapler_base" type="box" pos="{(BASE_X0 + BASE_X1) / 2:g} 0 {BASE_H / 2:g}" '
     f'size="{(BASE_X1 - BASE_X0) / 2:g} {BASE_HY:g} {BASE_H / 2:g}" rgba="{BLACK}" '
     'contype="2" conaffinity="1"/>'),
    (f'      <geom name="stapler_post" type="box" pos="{(POST_X0 + BASE_X1) / 2:g} 0 {(BASE_H + POST_TOP) / 2:g}" '
     f'size="{(BASE_X1 - POST_X0) / 2:g} {POST_HY:g} {(POST_TOP - BASE_H) / 2:g}" rgba="{BLACK}" '
     'contype="2" conaffinity="1"/>'),
    f'      <body name="stapler_arm" pos="{HINGE_X:g} 0 {HINGE_Z:g}">',
    (f'        <joint name="stapler_arm" type="hinge" axis="0 -1 0" range="0 {math.degrees(MAX_ANGLE):g}" '
     f'stiffness="1" springref="{math.degrees(SPRINGREF):g}" damping="0.05" frictionloss="0.01" '
     'solreflimit="0.004 1"/>'),
    (f'        <geom name="stapler_arm" type="box" pos="{0.005 - ARM_HX:g} 0 0" '
     f'size="{ARM_HX:g} {ARM_HY:g} {ARM_HZ:g}" rgba="{RED}" mass="0.1" '
     'contype="4" conaffinity="1"/>'),
    "      </body>",
    "    </body>",
])
SHAPES = (  # the arm up, at angle 0
    box("stapler_base", ((BASE_X0 + BASE_X1) / 2, 0, BASE_H / 2), (BASE_X1 - BASE_X0, 2 * BASE_HY, BASE_H), BLACK),
    box("stapler_post", ((POST_X0 + BASE_X1) / 2, 0, (BASE_H + POST_TOP) / 2),
        (BASE_X1 - POST_X0, 2 * POST_HY, POST_TOP - BASE_H), BLACK),
    box("stapler_arm", (HINGE_X + 0.005 - ARM_HX, 0, HINGE_Z), (2 * ARM_HX, 2 * ARM_HY, 2 * ARM_HZ), RED),
)
TEXT = (f"A stapler lies on the table along x, at x {BASE_X0:.2f} to {BASE_X1:.2f}, y {-ARM_HY:.3f} to "
        f"{ARM_HY:.3f}. It has a black base, {BASE_H * 100:.1f} cm high, and a red top arm. The arm turns on "
        f"a hinge at its far end, at x {HINGE_X:.3f}. A spring holds the arm up, with its top at "
        f"z {HINGE_Z + ARM_HZ:.3f}. The front end of the arm, at x {ARM_X0:.2f}, is free. "
        f"Push the front of the arm down about 2 cm to press the stapler. "
        "Close the gripper first and push with the fingertips.")


def press(x: float = PRESS_X) -> list[Action]:
    """Close the gripper, go above the arm, push it down to its stop and go back up."""
    return [Action("close", expect="nothing"), Action("move_to", (x, 0.0, LIFT_Z), yaw=0.0),
            Action("move_to", (x, 0.0, PRESS_Z)), Action("move_to", (x, 0.0, LIFT_Z))]


BRIEF = 'A stapler lies on the table along x. It has a black base and a red top arm. The arm turns on a hinge at its far (+x) end, and a spring holds it up. Close the gripper first, then push the front of the arm down with the fingertips.'  # the fixed items for a prompt with names only: no positions or sizes

STAPLER = Scene(
    name="stapler", fixtures_brief=BRIEF, title="Stapler", objects=(), fixtures_xml=FIXTURES, fixtures_text=TEXT, fixtures=SHAPES,
    region=(0.35, 0.45, -0.1, 0.1), min_gap=0.1, joints={"stapler_arm": "stapler arm angle {:.3f} rad (0 = up)"},
    monitors=(threshold_monitor("stapler_arm", FIRE, REARM, "stapler pressed"),),
    tasks={t.name: t for t in (
        Task("press_stapler", "Press the stapler down.", lambda w: "stapler pressed" in w.events,
             lambda w: press()),
    )},
)
