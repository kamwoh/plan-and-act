"""A microwave oven with a door on a vertical hinge and a start button, and a box of noodles on the table."""
from __future__ import annotations

import itertools

import numpy as np

from plan_and_act.scenes.base import Monitor, ObjSpec, Scene, Task, box, cylinder
from plan_and_act.types import Action

FOOD = ObjSpec("noodles", "box", (0.025, 0.025, 0.025), "0.95 0.8 0.25 1", "box of noodles, 5 cm cube", mass=0.03)

# The oven body. The front face is at x FRONT, toward the robot. The cavity is open at the front.
FRONT, BACK, MW_H = 0.52, 0.74, 0.13
IN_HY, WALL, FLOOR_Z, CEIL_Z = 0.125, 0.01, 0.02, 0.12  # the cavity: half width, wall, floor top, ceiling
OUT_HY = IN_HY + WALL  # 0.135
SLOT = 0.09  # the top panel stops this far behind the front, so that the hand can reach the cavity front
CTRL_Y0, CTRL_Y1 = -OUT_HY - 0.08, -OUT_HY  # the control box, at the right of the cavity
# The door turns about a vertical axis at the left front corner. Its frame is at the hinge, door angle 0 (closed).
HINGE_Y = OUT_HY
DOOR_T, DOOR_W, DOOR_Z0 = 0.02, 2 * OUT_HY, 0.002  # it hangs 2 mm above the table
MAX_OPEN = 110.0  # degrees: the MJCF range is in degrees, qpos is in radians
OPEN_AT, CLOSED_AT = np.radians(70.0), np.radians(3.0)
# The handle: a vertical bar near the free edge of the door. Two posts hold its lower part. Its top part is free,
# so that a closed gripper can go down between the bar and the door.
HANDLE_S, STANDOFF, BAR_R, POST_R = DOOR_W - 0.03, 0.045, 0.006, 0.005  # S: the distance from the hinge
BAR_Z0, BAR_Z1, POST_Z = MW_H - 0.10, MW_H, (MW_H - 0.09, MW_H - 0.05)
# The start button on top of the control box.
BTN_X, BTN_Y, CAP_R, CAP_HH, TRAVEL = FRONT + 0.04, (CTRL_Y0 + CTRL_Y1) / 2, 0.015, 0.006, 0.008
FIRE, REARM = 0.7 * TRAVEL, 0.3 * TRAVEL
WHITE, GREY, DARK = "0.92 0.92 0.9 1", "0.55 0.56 0.58 1", "0.2 0.2 0.22 1"
GLASS, GREEN = "0.25 0.3 0.33 1", "0.1 0.75 0.2 1"
LIFT_Z = 0.25
FINGER_HX = 0.011  # half the thickness of the closed fingers, across the closing axis


def _box(name: str, lo, hi, rgba: str, bits: str) -> str:
    """A box geom from its low and high corners."""
    c = [(a + b) / 2 for a, b in zip(lo, hi, strict=True)]
    h = [(b - a) / 2 for a, b in zip(lo, hi, strict=True)]
    return (f'      <geom name="{name}" type="box" pos="{c[0]:g} {c[1]:g} {c[2]:g}" '
            f'size="{h[0]:g} {h[1]:g} {h[2]:g}" rgba="{rgba}" {bits}/>')


# The body parts, in world coordinates: name -> (low corner, high corner, colour).
BODY = {
    "microwave_floor": ((FRONT, -OUT_HY, 0.0), (BACK, OUT_HY, FLOOR_Z), WHITE),
    "microwave_left": ((FRONT, IN_HY, FLOOR_Z), (BACK, OUT_HY, MW_H), WHITE),
    "microwave_right": ((FRONT, -OUT_HY, FLOOR_Z), (BACK, -IN_HY, MW_H), WHITE),
    "microwave_back": ((BACK - WALL, -IN_HY, FLOOR_Z), (BACK, IN_HY, CEIL_Z), WHITE),
    "microwave_top": ((FRONT + SLOT, -IN_HY, CEIL_Z), (BACK, IN_HY, MW_H), WHITE),
    "microwave_controls": ((FRONT, CTRL_Y0, 0.0), (BACK, CTRL_Y1, MW_H), GREY),
}
# The door panel and the bar axis, in the door frame at the hinge.
DOOR_LO, DOOR_HI = (-DOOR_T, -DOOR_W, DOOR_Z0), (0.0, 0.0, MW_H)
_HX = -DOOR_T - STANDOFF

# Contact bits: the gripper, the food and the table use 1. The body (2/1), the door and the button (4/1)
# touch the gripper and the food, but not each other.
_FIXED, _MOVING = 'contype="2" conaffinity="1"', 'contype="4" conaffinity="1"'
FIXTURES = "\n".join([
    '    <body name="microwave" pos="0 0 0">',
    *(_box(n, lo, hi, c, _FIXED) for n, (lo, hi, c) in BODY.items()),
    "    </body>",
    f'    <body name="microwave_door" pos="{FRONT:g} {HINGE_Y:g} 0">',
    (f'      <joint name="microwave_door" type="hinge" axis="0 0 -1" range="0 {MAX_OPEN:g}" damping="0.3" '
     'frictionloss="0.2" armature="0.05"/>'),
    _box("microwave_door", DOOR_LO, DOOR_HI, GLASS, _MOVING),
    (f'      <geom name="microwave_handle" type="cylinder" fromto="{_HX:g} {-HANDLE_S:g} {BAR_Z0:g} '
     f'{_HX:g} {-HANDLE_S:g} {BAR_Z1:g}" size="{BAR_R:g}" rgba="{DARK}" {_MOVING}/>'),
    *(f'      <geom name="microwave_post_{i}" type="cylinder" fromto="{-DOOR_T:g} {-HANDLE_S:g} {z:g} '
      f'{_HX:g} {-HANDLE_S:g} {z:g}" size="{POST_R:g}" rgba="{DARK}" {_MOVING}/>' for i, z in enumerate(POST_Z)),
    '      <inertial pos="-0.01 -0.135 0.065" mass="0.5" diaginertia="0.004 0.001 0.004"/>',
    "    </body>",
    f'    <body name="start_button" pos="{BTN_X:g} {BTN_Y:g} {MW_H + CAP_HH:g}">',
    (f'      <joint name="start_button" type="slide" axis="0 0 -1" range="0 {TRAVEL:g}" stiffness="300" '
     'springref="0" damping="2"/>'),
    (f'      <geom name="start_button" type="cylinder" size="{CAP_R:g} {CAP_HH:g}" rgba="{GREEN}" mass="0.01" '
     f'{_MOVING}/>'),
    "    </body>",
])


def _shape(name: str, lo, hi, rgba: str, at=(0.0, 0.0, 0.0)):
    c = [(a + b) / 2 + o for a, b, o in zip(lo, hi, at, strict=True)]
    return box(name, c, [b - a for a, b in zip(lo, hi, strict=True)], rgba)


_HINGE = (FRONT, HINGE_Y, 0.0)
SHAPES = (  # the door closed, the button up
    *(_shape(n, lo, hi, c) for n, (lo, hi, c) in BODY.items()),
    _shape("microwave_door", DOOR_LO, DOOR_HI, GLASS, _HINGE),
    cylinder("microwave_handle", (FRONT + _HX, HINGE_Y - HANDLE_S, (BAR_Z0 + BAR_Z1) / 2), BAR_R, BAR_Z1 - BAR_Z0,
             DARK),
    *(box(f"microwave_post_{i}", (FRONT + _HX / 2 - DOOR_T / 2, HINGE_Y - HANDLE_S, z), (STANDOFF, 2 * POST_R,
                                                                                           2 * POST_R), DARK)
      for i, z in enumerate(POST_Z)),
    cylinder("start_button", (BTN_X, BTN_Y, MW_H + CAP_HH), CAP_R, 2 * CAP_HH, GREEN),
)
TEXT = (f"A white microwave oven at x {FRONT:.2f} to {BACK:.2f}, y {CTRL_Y0:.3f} to {OUT_HY:.3f}, "
        f"{MW_H * 100:.0f} cm tall. Its cavity is open toward the robot (-x): x {FRONT:.2f} to {BACK - WALL:.2f}, "
        f"y {-IN_HY:.3f} to {IN_HY:.3f}, floor at z {FLOOR_Z:.2f}, ceiling at z {CEIL_Z:.2f}. The top panel starts "
        f"{SLOT * 100:.0f} cm behind the front, so the top of the cavity front is open. The door is {DOOR_T * 100:.0f} "
        f"cm thick and covers the cavity front, at x {FRONT - DOOR_T:.2f} to {FRONT:.2f}. It turns on a vertical "
        f"hinge at x {FRONT:.2f}, y {HINGE_Y:.3f}, and it opens toward the robot and to the left (+y), up to "
        f"{MAX_OPEN:.0f} degrees. The handle is a dark vertical bar at x {FRONT + _HX:.3f}, y {HINGE_Y - HANDLE_S:.3f} "
        f"when the door is closed, {STANDOFF * 100:.1f} cm in front of the door, from z {BAR_Z0:.2f} to {BAR_Z1:.2f}. "
        f"Two posts hold it at z {POST_Z[0]:.2f} and {POST_Z[1]:.2f}. Above z {POST_Z[1] + POST_R:.3f}, a closed "
        f"gripper can go down between the bar and the door. A grey control box is at the right, y {CTRL_Y0:.3f} to "
        f"{CTRL_Y1:.3f}. A green start button, {2 * CAP_R * 100:.0f} cm wide, is on top of it at x {BTN_X:.2f}, "
        f"y {BTN_Y:.3f}. Its top is at z {MW_H + 2 * CAP_HH:.3f}. Push it down about 1 cm with the closed fingers. "
        "The oven starts only when the door is closed.")


def door_angle(w) -> float:
    return w.joint_pos("microwave_door")


def in_microwave(w, name: str) -> bool:
    """True when every corner of the box around the object is inside the cavity, behind the door."""
    o = next(o for o in w.objects if o.name == name)
    body = w.data.body(name)
    corners = np.array(list(itertools.product(*((-d / 2, d / 2) for d in o.dims))))
    x, y, z = (corners @ body.xmat.reshape(3, 3).T + body.xpos).T
    return bool((x > FRONT).all() and (x < BACK - WALL).all() and (abs(y) < IN_HY).all()
                and (z > FLOOR_Z - 0.003).all() and (z < CEIL_Z).all())


def _closed(w) -> bool:
    return bool(door_angle(w) < CLOSED_AT) and "microwave_door" not in w.gripper_contacts()


def _food_in(w) -> bool:
    return in_microwave(w, FOOD.name) and FOOD.name not in w.gripper_contacts()


STARTED = "microwave started"


def start_monitor() -> Monitor:
    """A press of the start button gives one event. The oven starts only when the door is closed. The event
    also says if the food is inside."""
    armed = True

    def check(w) -> str | None:
        nonlocal armed
        q = w.joint_pos("start_button")
        if q < REARM:
            armed = True
        if not (armed and q > FIRE):
            return None
        armed = False
        if door_angle(w) >= CLOSED_AT:
            return "start button pressed with the door open"
        return STARTED if in_microwave(w, FOOD.name) else "microwave started empty"
    return check


def _cooked(w) -> bool:
    return STARTED in w.events and _food_in(w) and _closed(w)


def _at(p, z: float, yaw: float | None = None) -> Action:
    return Action("move_to", (float(p[0]), float(p[1]), float(z)), yaw=yaw)


def _door_point(angle_deg: float, s: float, n: float) -> np.ndarray:
    """The (x, y) of a point that turns with the door: s from the hinge along the door, n in front of the
    inner face of the door. The door opens clockwise, seen from above."""
    a = np.radians(angle_deg)
    x, y = -n, -s
    return np.array([FRONT + x * np.cos(a) + y * np.sin(a), HINGE_Y - x * np.sin(a) + y * np.cos(a)])


HOOK_Z = BAR_Z1 - 0.02  # the tcp: the fingertips are 3 cm below the bar top, 1.5 cm above the upper post
OPEN_TO = 100  # degrees
PULL = 0.006  # the fingers aim this far into the bar, so that they keep pulling it


def _open_plan(w) -> list[Action]:
    """Hook the closed gripper between the bar and the door, then pull the bar around the hinge.
    The gripper turns with the door, so that its thin side stays across the gap. Then lift out."""
    n = DOOR_T + STANDOFF - BAR_R - FINGER_HX + PULL
    path = [(_door_point(a, HANDLE_S, n), 90.0 - a) for a in range(0, OPEN_TO + 1, 10)]
    end = path[-1][0]
    hook = _door_point(0, HANDLE_S, DOOR_T + (STANDOFF - BAR_R) / 2)  # the middle of the gap
    return [Action("close", expect="nothing"), _at(hook, LIFT_Z, 90.0), _at(hook, HOOK_Z),
            *(_at(p, HOOK_Z, yaw) for p, yaw in path[1:]), _at(end, LIFT_Z)]


GRIP_DOWN = 0.012  # the tcp is this far below the top of the box, above its centre, so that the box hangs still
DROP = 0.008  # the box bottom is this high above the cavity floor when the gripper lets go


def _food_plan(w) -> list[Action]:
    """Grasp the box near its top and carry it into the cavity, until the hand is 1 cm in front of the top
    panel. There the whole box is behind the door. Let go and back away."""
    p = w.object_pos(FOOD.name)
    grip = float(p[2]) + FOOD.half_height - GRIP_DOWN
    carry = FLOOR_Z + DROP + 2 * FOOD.half_height - GRIP_DOWN
    out, drop = (FRONT - 0.10, 0.0), (FRONT + 0.045, 0.0)  # the hand front stops 1 cm before the top panel
    return [Action("open"), _at(p, LIFT_Z, -90.0), _at(p, grip), Action("close"), _at(p, LIFT_Z),
            _at(out, LIFT_Z), _at(out, carry), _at(drop, carry), Action("open"), _at(out, carry),
            _at(out, LIFT_Z)]


PUSH_S, PRESS = 0.15, 0.0  # where the fingers push the door shut, and how far they aim into it
PUSH_Z = 0.06
PUSH_END = 10  # degrees: the sweep ends here. The door then shuts, and the fingers do not push the frame.


def _close_plan(w) -> list[Action]:
    """Put the closed fingers outside the open door, then sweep them around the hinge to push it shut."""
    n = DOOR_T + FINGER_HX - PRESS
    start = _door_point(OPEN_TO + 15, PUSH_S, n + 0.03)
    path = [(_door_point(a, PUSH_S, n), 90.0 - a) for a in range(OPEN_TO + 10, PUSH_END - 1, -10)]
    return [Action("close", expect="nothing"), _at(start, LIFT_Z, 90.0 - OPEN_TO - 15), _at(start, PUSH_Z),
            *(_at(p, PUSH_Z, yaw) for p, yaw in path), _at(path[-1][0], LIFT_Z)]


def _press_plan(w) -> list[Action]:
    """Above the start button, down onto it, back up. The gripper must be closed first."""
    b = (BTN_X, BTN_Y)
    return [Action("close", expect="nothing"), _at(b, LIFT_Z), _at(b, MW_H + 2 * CAP_HH - TRAVEL), _at(b, LIFT_Z)]


BRIEF = 'A white microwave oven. Its cavity opens toward the robot (-x), and the top of the cavity front is open. The door turns on a vertical hinge at its left (+y) side and opens toward the robot. The handle is a dark vertical bar in front of the door. A closed gripper can go down between the bar and the door, above the lower post. A green start button is on top of the grey control box at the right. The oven starts only when the door is closed.'  # the fixed items for a prompt with names only: no positions or sizes

MICROWAVE = Scene(
    name="microwave", fixtures_brief=BRIEF, title="Microwave", objects=(FOOD,), fixtures_xml=FIXTURES, fixtures_text=TEXT,
    fixtures=SHAPES, region=(0.40, 0.46, -0.36, -0.26), min_gap=0.1,
    joints={"microwave_door": "microwave door angle {:.2f} rad (0 = closed)",
            "start_button": "start button depth {:.3f} m"},
    monitors=(start_monitor,),
    tasks={t.name: t for t in (
        Task("open_door", "Open the microwave door to at least 70 degrees.",
             lambda w: bool(door_angle(w) >= OPEN_AT), _open_plan),
        Task("food_in_microwave", "Open the microwave door and put the box of noodles inside the microwave.", _food_in,
             lambda w: [*_open_plan(w), *_food_plan(w)]),
        Task("cook", "Put the box of noodles in the microwave, close the door and press the start button.", _cooked,
             lambda w: [*_open_plan(w), *_food_plan(w), *_close_plan(w), *_press_plan(w)]),
    )},
)
