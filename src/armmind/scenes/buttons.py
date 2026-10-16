"""A low panel with three spring buttons: red, green and blue. A press gives an event."""
from __future__ import annotations

from armmind.scenes.base import Scene, Task, threshold_monitor
from armmind.types import Action

PANEL_X, PANEL_HX, PANEL_HY, PANEL_H = 0.62, 0.05, 0.15, 0.04
CAP_R, CAP_HH, TRAVEL = 0.02, 0.008, 0.012  # the cap: radius, half height; how far a button goes down
FIRE, REARM = 0.7 * TRAVEL, 0.3 * TRAVEL
LIFT_Z = 0.15
PRESS_Z = PANEL_H + 2 * CAP_HH - TRAVEL  # the fingertips go 1 cm below the TCP, so this pushes a cap to the bottom
BUTTONS = {  # colour: (y, rgba)
    "red": (-0.1, "0.9 0.1 0.1 1"),
    "green": (0.0, "0.1 0.75 0.2 1"),
    "blue": (0.1, "0.15 0.3 0.95 1"),
}

# Contact bits: the gripper and the table use 1. The panel (2/1) and the caps (4/1) touch the gripper,
# but not each other, so that a cap can sink into the panel.
FIXTURES = "\n".join([
    f'    <body name="button_panel" pos="{PANEL_X:g} 0 {PANEL_H / 2:g}">',
    (f'      <geom name="button_panel" type="box" size="{PANEL_HX:g} {PANEL_HY:g} {PANEL_H / 2:g}" '
    'rgba="0.3 0.3 0.33 1" contype="2" conaffinity="1"/>'),
    "    </body>",
    *(f'    <body name="{c}_button" pos="{PANEL_X:g} {y:g} {PANEL_H + CAP_HH:g}">\n'
      f'      <joint name="{c}_button" type="slide" axis="0 0 -1" range="0 {TRAVEL:g}" stiffness="300" '
      'springref="0" damping="2"/>\n'
      f'      <geom name="{c}_button" type="cylinder" size="{CAP_R:g} {CAP_HH:g}" rgba="{rgba}" mass="0.01" '
      'contype="4" conaffinity="1"/>\n'
      "    </body>" for c, (y, rgba) in BUTTONS.items()),
])
TEXT = (f"A low grey panel, {PANEL_H * 100:.0f} cm high, at x {PANEL_X - PANEL_HX:.2f} to {PANEL_X + PANEL_HX:.2f}, "
        f"y {-PANEL_HY:.2f} to {PANEL_HY:.2f}. On top are three round buttons, {2 * CAP_R * 100:.0f} cm wide, "
        f"at x {PANEL_X:.2f}: red at y -0.10, green at y 0.00, blue at y 0.10. The top of a button is at "
        f"z {PANEL_H + 2 * CAP_HH:.3f}. Close the gripper first, then push a button down about 1 cm "
        "with the fingertips. An open gripper straddles the button and lands on the panel.")


def presses(w) -> list[str]:
    """The colours of the button presses so far, in order."""
    return [e.removesuffix(" button pressed") for e in w.events if e.endswith(" button pressed")]


def press(colour: str) -> list[Action]:
    """Above the button, down onto it, back up. The gripper must be closed first."""
    y = BUTTONS[colour][0]
    return [Action("move_to", (PANEL_X, y, LIFT_Z)), Action("move_to", (PANEL_X, y, PRESS_Z)),
            Action("move_to", (PANEL_X, y, LIFT_Z))]


BUTTONS_SCENE = Scene(
    name="buttons", title="Buttons", objects=(), fixtures_xml=FIXTURES, fixtures_text=TEXT,
    region=(0.35, 0.45, -0.1, 0.1), min_gap=0.1, joints={f"{c}_button": f"{c} button depth {{:.3f}} m" for c in BUTTONS},
    monitors=tuple(threshold_monitor(f"{c}_button", FIRE, REARM, f"{c} button pressed") for c in BUTTONS),
    tasks={t.name: t for t in (
        Task("press_green", "Press the green button.", lambda w: "green" in presses(w),
             lambda w: [Action("close"), *press("green")]),
        Task("press_sequence", "Press the red button, then the green button, then the blue button.",
             lambda w: presses(w) == ["red", "green", "blue"],
             lambda w: [Action("close"), *(a for c in BUTTONS for a in press(c))]),
    )},
)
