"""The system prompt, the reply schema and the text of each step's message."""
from __future__ import annotations

from dataclasses import dataclass

from armmind.types import Observation, fmt

SYSTEM = """\
You control a Franka Panda robot arm in a physics simulation. Each step, you get two camera images \
and the state of the gripper. You reply with a short plan and a list of actions.

Coordinates are in metres, in the world frame:
- The robot base is at x=0, y=0. The table surface is at z=0.
- +x points away from the robot. +y points to the right in both images.
- In the top image, +x points down. In the front image, +x points toward the camera.
- Both images have a grid on the table, with labels for x and y. Use it to find positions.

The cubes are 4 cm wide. The centre of a cube on the table is at z=0.02.
The gripper position is the point between the two fingertips. The gripper always points down.
The gripper is 0.08 m wide when open. When it closes on a cube, the width is about 0.04 m. \
When it closes on nothing, the width is about 0.

Actions:
- {"type": "move_to", "xyz": [x, y, z]}: move the gripper to this point.
- {"type": "open"} and {"type": "close"}: open or close the gripper.
- {"type": "wait", "seconds": s}: wait for s seconds (at most 3).
The arm can reach x from 0.25 to 0.75, y from -0.40 to 0.40, z from 0.01 to 0.50. \
The arm moves a target outside this box to the nearest point inside it.
To grasp a cube: open, move 10 cm above it, move down to its centre, close, then move up.
Each result says if the gripper reached the target, was blocked, or could not reach it.

The simulation stops while you think. Use at most 8 actions in each step. \
Write "observation" in 2 short sentences and "plan" in 1 sentence. \
Write "notes" only when something worth remembering changed. \
Keep a "todo" list of 3 to 5 short sub-goals. \
Set "done" to true when you are sure the task is complete. The actions in the same reply still run first."""

REPLY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["observation", "plan", "actions", "todo"],
    "properties": {
        "observation": {"type": "string"},
        "plan": {"type": "string"},
        "actions": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type"],
                "properties": {
                    "type": {"enum": ["move_to", "open", "close", "wait"]},
                    "xyz": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "seconds": {"type": "number"},
                },
            },
        },
        "notes": {"type": "string"},
        "todo": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "done": {"type": "boolean"},
    },
}


@dataclass(frozen=True)
class StepSummary:
    step: int
    plan: str
    results: tuple[str, ...]


def build_user_text(instruction: str, step: int, max_steps: int, obs: Observation, notes: str,
                    todo: tuple[str, ...], history: list[StepSummary], check_failed: bool = False) -> str:
    lines = [
        f"Task: {instruction}",
        f"Step {step} of {max_steps}.",
        f"Gripper at {fmt(obs.tcp)}, width {obs.gripper_width:.3f} m.",
        f"Gripper touching: {', '.join(obs.contacts) if obs.contacts else 'nothing'}.",
    ]
    if obs.objects is not None:
        lines.append("Cube centres (debug): " + "; ".join(f"{n} {fmt(p)}" for n, p in obs.objects.items()))
    if history:
        lines.append("Recent steps:")
        lines += [f"- step {h.step}: {h.plan} -> {'; '.join(h.results) or 'no actions'}" for h in history]
    lines.append(f"Your notes: {notes or '(empty)'}")
    lines.append("Your to-do list: " + ("; ".join(todo) if todo else "(empty)"))
    if check_failed:
        lines.append("The automatic check says the task is not finished yet.")
    lines.append("Image 1: front camera. Image 2: top camera.")
    return "\n".join(lines)


def example_step_text() -> str:
    """A sample step message with invented values, so that the page can show how the code fills it in."""
    from armmind.tasks import TASKS
    obs = Observation(b"", b"", (0.3, 0.0, 0.4), 0.08, ())
    return build_user_text(TASKS["stack"].instruction, 1, 30, obs, "", (), [])
