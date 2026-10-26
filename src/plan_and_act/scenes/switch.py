"""A rocker switch on a small box. It stays on or off until the gripper pushes its other end down."""
from __future__ import annotations

import math

from plan_and_act.scenes.base import Scene, Task, box
from plan_and_act.scenes.common import LIFT_Z
from plan_and_act.types import Action

BOX_X, BOX_Y, BOX_HX, BOX_HY, BOX_H = 0.62, 0.0, 0.045, 0.035, 0.04
ROCK_HX, ROCK_HY, ROCK_HZ = 0.03, 0.02, 0.006  # the rocker, along x
PIVOT_Z = BOX_H + ROCK_HZ  # the hinge is at the rocker centre: a flat rocker rests on the box top
TILT = 0.2  # rad: the rocker turns this far from flat to each side
ON = 2 * TILT  # the joint angle when the switch is on: the far (+x) end is down
COM_DZ = 0.03  # the centre of mass is this far above the hinge, so that the rocker falls to the nearer end
PRESS_DX = 0.02  # the gripper pushes this far out from the hinge
PRESS_Z = PIVOT_Z + ROCK_HZ + 0.007  # the fingertips are 1 cm below the TCP: they push the end a little past flat
BOX_RGBA, ROCK_RGBA = "0.85 0.85 0.82 1", "0.15 0.15 0.17 1"

# Contact bits: the gripper and the table use 1. The box (2/1) and the rocker (4/1) touch the gripper,
# but not each other, so that the low end of the rocker can sink into the box top. At angle 0 the switch is off
# and the near (-x) end is down. The joint range is 0 to ON. MJCF reads the angles in degrees.
# The rocker has no spring. The high centre of mass gives two stable ends, and the range holds it there.
FIXTURES = "\n".join([
    f'    <body name="switch_box" pos="{BOX_X:g} {BOX_Y:g} {BOX_H / 2:g}">',
    (f'      <geom name="switch_box" type="box" size="{BOX_HX:g} {BOX_HY:g} {BOX_H / 2:g}" rgba="{BOX_RGBA}" '
     'contype="2" conaffinity="1"/>'),
    f'      <body name="switch_rocker" pos="0 0 {PIVOT_Z - BOX_H / 2:g}" euler="0 {-math.degrees(TILT):g} 0">',
    (f'        <joint name="switch" type="hinge" axis="0 1 0" range="0 {math.degrees(ON):g}" damping="0.002" '
     'frictionloss="0.0002" solreflimit="0.004 1"/>'),
    f'        <inertial pos="0 0 {COM_DZ:g}" mass="0.05" diaginertia="1e-5 1e-5 1e-5"/>',
    (f'        <geom name="switch_rocker" type="box" size="{ROCK_HX:g} {ROCK_HY:g} {ROCK_HZ:g}" rgba="{ROCK_RGBA}" '
     'contype="4" conaffinity="1"/>'),
    "      </body>",
    "    </body>",
])
# The rocker is tilted at the start. Its box is the bounds of the tilted rocker.
_EX = ROCK_HX * math.cos(TILT) + ROCK_HZ * math.sin(TILT)
_EZ = ROCK_HX * math.sin(TILT) + ROCK_HZ * math.cos(TILT)
SHAPES = (
    box("switch_box", (BOX_X, BOX_Y, BOX_H / 2), (2 * BOX_HX, 2 * BOX_HY, BOX_H), BOX_RGBA),
    box("switch_rocker", (BOX_X, BOX_Y, PIVOT_Z), (2 * _EX, 2 * ROCK_HY, 2 * _EZ), ROCK_RGBA),
)
TEXT = (f"A small white box, {BOX_H * 100:.0f} cm high, at x {BOX_X - BOX_HX:.3f} to {BOX_X + BOX_HX:.3f}, "
        f"y {BOX_Y - BOX_HY:.3f} to {BOX_Y + BOX_HY:.3f}. On top is a black rocker switch, "
        f"{2 * ROCK_HX * 100:.0f} cm long along x and {2 * ROCK_HY * 100:.0f} cm wide. It turns on a hinge "
        f"along y at x {BOX_X:.2f}, z {PIVOT_Z:.3f}. When the switch is off, the far (+x) end is up. "
        "When the switch is on, the near (-x) end is up. Push the high end down to flip the switch. "
        "The switch stays where it is put. Close the gripper first and push with the fingertips.")


def angle(w) -> float:
    return w.joint_pos("switch")


def _free(w) -> bool:
    return "switch_rocker" not in w.gripper_contacts()


def flip(side: float) -> list[Action]:
    """Close the gripper, go above one end of the rocker, push it down and go back up.
    A side of +1 pushes the far (+x) end down and turns the switch on. A side of -1 turns it off."""
    x = BOX_X + side * PRESS_DX
    return [Action("close", expect="nothing"), Action("move_to", (x, BOX_Y, LIFT_Z), yaw=90.0),
            Action("move_to", (x, BOX_Y, PRESS_Z)), Action("move_to", (x, BOX_Y, LIFT_Z))]


BRIEF = 'A small white box with a black rocker switch on top. When the switch is off, the far (+x) end is up. When it is on, the near (-x) end is up. Push the high end down to flip it. Close the gripper first and push with the fingertips.'  # the fixed items for a prompt with names only: no positions or sizes

SWITCH = Scene(
    name="switch", fixtures_brief=BRIEF, title="Switch", objects=(), fixtures_xml=FIXTURES, fixtures_text=TEXT, fixtures=SHAPES,
    region=(0.35, 0.45, -0.1, 0.1), min_gap=0.1, joints={"switch": f"switch angle {{:.3f}} rad (0 = off, {ON:g} = on)"},
    tasks={t.name: t for t in (
        Task("switch_on", "Turn the switch on.", lambda w: angle(w) > 0.75 * ON and _free(w), lambda w: flip(1.0)),
        Task("switch_off", "Turn the switch off.", lambda w: angle(w) < 0.25 * ON and _free(w), lambda w: flip(-1.0),
             initial_qpos={"switch": ON}),
    )},
)
