"""A computer mouse and a mouse pad. The mouse has a left button on a spring. A click gives an event."""
from __future__ import annotations

from plan_and_act.scenes.base import Monitor, Scene, Task, box
from plan_and_act.scenes.common import LIFT_Z
from plan_and_act.types import Action

# The mouse is not an ObjSpec: an ObjSpec body has one geom and no joint inside. The mouse is a free body in
# the fixtures, with a fixed start pose and a child button on a slide joint. Its long side is along x.
MOUSE_X, MOUSE_Y = 0.42, -0.18  # the start centre
SHELL_HX, SHELL_HY, SHELL_HZ = 0.05, 0.03, 0.015
BTN_DX, BTN_DY, BTN_HX, BTN_HY, BTN_HZ = 0.027, 0.015, 0.02, 0.0135, 0.003  # left button at +y, right at -y
TRAVEL = 0.003  # how far the left button goes down
FIRE, REARM = 0.7 * TRAVEL, 0.3 * TRAVEL
TOP = 2 * SHELL_HZ + 2 * BTN_HZ  # the top of the buttons
PAD_X, PAD_Y, PAD_HX, PAD_HY = 0.60, 0.15, 0.09, 0.11
GRASP_DX = -0.005  # the fingers grip just behind the centre, clear of the buttons
GRASP_Z = 0.02
RELEASE_Z = GRASP_Z + 0.005
PRESS_Z = TOP - TRAVEL + 0.012  # the fingertips are 1 cm below the TCP, and the arm sags a little when it pushes
GREY, DARK, PAD = "0.75 0.75 0.78 1", "0.25 0.25 0.28 1", "0.15 0.35 0.6 1"
CLICK, CLICK_ON_PAD = "mouse clicked off the pad", "mouse clicked on the pad"

# The pad is visual only: contype and conaffinity 0, so nothing collides with it. The left button is a child
# body of the mouse, so the two never touch: MuJoCo does not test a body against its parent.
FIXTURES = "\n".join([
    f'    <body name="mouse_pad" pos="{PAD_X:g} {PAD_Y:g} 0.0006">',
    (f'      <geom name="mouse_pad" type="box" size="{PAD_HX:g} {PAD_HY:g} 0.0005" rgba="{PAD}" contype="0" '
     'conaffinity="0"/>'),
    "    </body>",
    f'    <body name="mouse" pos="{MOUSE_X:g} {MOUSE_Y:g} {SHELL_HZ:g}">',
    '      <freejoint name="mouse"/>',
    (f'      <geom name="mouse_shell" type="box" size="{SHELL_HX:g} {SHELL_HY:g} {SHELL_HZ:g}" rgba="{GREY}" '
     'mass="0.06" friction="1.5 0.01 0.001" condim="4"/>'),
    (f'      <geom name="mouse_right_button" type="box" pos="{BTN_DX:g} {-BTN_DY:g} {SHELL_HZ + BTN_HZ:g}" '
     f'size="{BTN_HX:g} {BTN_HY:g} {BTN_HZ:g}" rgba="{DARK}" mass="0.005"/>'),
    f'      <body name="mouse_button" pos="{BTN_DX:g} {BTN_DY:g} {SHELL_HZ + BTN_HZ:g}">',
    (f'        <joint name="mouse_button" type="slide" axis="0 0 -1" range="0 {TRAVEL:g}" stiffness="300" '
     'springref="-0.0005" damping="1" frictionloss="0.05" solreflimit="0.004 1"/>'),
    (f'        <geom name="mouse_button" type="box" size="{BTN_HX:g} {BTN_HY:g} {BTN_HZ:g}" rgba="{DARK}" '
     'mass="0.005"/>'),
    "      </body>",
    "    </body>",
])
SHAPES = (  # the mouse at its start pose
    box("mouse_pad", (PAD_X, PAD_Y, 0.0006), (2 * PAD_HX, 2 * PAD_HY, 0.001), PAD),
    box("mouse_shell", (MOUSE_X, MOUSE_Y, SHELL_HZ), (2 * SHELL_HX, 2 * SHELL_HY, 2 * SHELL_HZ), GREY),
    *(box(n, (MOUSE_X + BTN_DX, MOUSE_Y + dy, 2 * SHELL_HZ + BTN_HZ), (2 * BTN_HX, 2 * BTN_HY, 2 * BTN_HZ), DARK)
      for n, dy in (("mouse_right_button", -BTN_DY), ("mouse_button", BTN_DY))),
)
TEXT = (f"A blue mouse pad, flat on the table, at x {PAD_X - PAD_HX:.2f} to {PAD_X + PAD_HX:.2f}, "
        f"y {PAD_Y - PAD_HY:.2f} to {PAD_Y + PAD_HY:.2f}. A grey computer mouse lies on the table. "
        f"It is {2 * SHELL_HX * 100:.0f} cm long along x, {2 * SHELL_HY * 100:.0f} cm wide along y and "
        f"{TOP * 100:.1f} cm tall. At the start, its centre is at x {MOUSE_X:.2f}, y {MOUSE_Y:.2f}. "
        f"Its two buttons are on the +x half of its top. The left button is on the +y side, "
        f"{BTN_DX * 100:.1f} cm in +x and {BTN_DY * 100:.1f} cm in +y from the mouse centre. "
        "The mouse can move, but it does not turn by itself. To click, close the gripper first. "
        f"Then push the left button down about {TRAVEL * 1000:.0f} mm with the fingertips.")


def mouse_on_pad(w) -> bool:
    """True when the mouse centre is above the pad, the mouse rests flat, and the gripper does not touch it."""
    x, y, z = w.object_pos("mouse")
    flat = w.data.body("mouse").xmat[8] > 0.99  # the mouse z axis points up
    touch = {"mouse", "mouse_button"} & set(w.gripper_contacts())
    return bool(abs(x - PAD_X) <= PAD_HX and abs(y - PAD_Y) <= PAD_HY and z < 0.03 and flat and not touch)


def click_monitor() -> Monitor:
    """Gives an event once for each click. The event says if the mouse centre was above the pad."""
    armed = True

    def check(w) -> str | None:
        nonlocal armed
        q = w.joint_pos("mouse_button")
        if armed and q > FIRE:
            armed = False
            x, y, _ = w.object_pos("mouse")
            return CLICK_ON_PAD if abs(x - PAD_X) <= PAD_HX and abs(y - PAD_Y) <= PAD_HY else CLICK
        if q < REARM:
            armed = True
        return None
    return check


def _at(x: float, y: float, z: float, yaw: float | None = None) -> Action:
    return Action("move_to", (float(x), float(y), float(z)), yaw=yaw)


def _to_pad(w) -> list[Action]:
    """Grip the mouse across its width, just behind the centre. Carry it to the pad centre and let go."""
    x, y, _ = w.object_pos("mouse")
    gx, px = x + GRASP_DX, PAD_X + GRASP_DX
    return [Action("open"), _at(gx, y, LIFT_Z, 90.0), _at(gx, y, GRASP_Z), Action("close"), _at(gx, y, LIFT_Z),
            _at(px, PAD_Y, LIFT_Z), _at(px, PAD_Y, RELEASE_Z), Action("open"), _at(px, PAD_Y, LIFT_Z)]


def _click_on_pad(w) -> list[Action]:
    """Carry the mouse to the pad centre, then click it there."""
    return [*_to_pad(w), *click_at(PAD_X, PAD_Y)]


def click_at(x: float, y: float) -> list[Action]:
    """Close the gripper, go above the left button of a mouse centred at (x, y), push it down, go back up."""
    bx, by = x + BTN_DX, y + BTN_DY
    return [Action("close", expect="nothing"), _at(bx, by, LIFT_Z, 0.0), _at(bx, by, PRESS_Z), _at(bx, by, LIFT_Z)]


BRIEF = 'A blue mouse pad lies flat on the table. A grey computer mouse lies on the table. Its two buttons are on the +x half of its top, and the left button is on the +y side. The mouse can move. To click, close the gripper first, then push the left button down with the fingertips.'  # the fixed items for a prompt with names only: no positions or sizes

MOUSE = Scene(
    name="mouse", fixtures_brief=BRIEF, title="Mouse", objects=(), fixtures_xml=FIXTURES, fixtures_text=TEXT, fixtures=SHAPES,
    region=(0.35, 0.45, -0.25, -0.1), min_gap=0.1, joints={"mouse_button": "mouse button depth {:.4f} m"},
    monitors=(click_monitor,),
    tasks={t.name: t for t in (
        Task("mouse_on_pad", "Put the mouse on the mouse pad.", mouse_on_pad, _to_pad),
        Task("click_on_pad", "Put the mouse on the mouse pad, then click its left button.",
             lambda w: CLICK_ON_PAD in w.events and mouse_on_pad(w), _click_on_pad),
    )},
)
