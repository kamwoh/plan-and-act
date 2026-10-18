"""The system prompt, the reply schema and the text of each step's message."""
from __future__ import annotations

import json
from dataclasses import dataclass

from armmind.types import Observation, fmt

# Parts that the step prompt and the plan prompt share. Each part ends with a newline.
COORDS = """\
Coordinates are in metres, in the world frame:
- The robot base is at x=0, y=0. The table surface is at z=0.
- +x points away from the robot. +y points to the right in both images.
- In the top image, +x points down. In the front image, +x points toward the camera.
- Both images have a grid on the table, with labels for x and y. Use it to find positions.
"""
OBJECTS = """
Objects sit on the table, and the step message lists them with their sizes. \
The centre of an object on the table is at half its height.
"""
GRIPPER = """\
The gripper position is the point between the two fingertips. The gripper always points down.
The gripper is 0.08 m wide when open. When it closes on an object, the width is about the width of the object. \
When it closes on nothing, the width is about 0.
"""
YAW = """\
0 is the start orientation, and at 0 the fingers close along the x axis. Positive yaw turns the gripper \
counter-clockwise, seen from above."""
REACH = """\
The arm can reach x from 0.25 to 0.75, y from -0.40 to 0.40, z from 0.01 to 0.50. \
The arm moves a target outside this box to the nearest point inside it.
"""
GRASP = """\
To grasp an object: open, move 10 cm above it, move down to its centre height, close, then move up.
For an object taller than 7 cm, close the gripper 3.5 cm below its top, not at its centre.
An object can stand at an angle. Then turn the gripper with "yaw" so that the fingers close across \
its narrow side.
"""

SYSTEM = f"""\
You control a Franka Panda robot arm in a physics simulation. Each step, you get two camera images \
and the state of the gripper. You reply with a short plan and a list of actions.

{COORDS}{OBJECTS}{GRIPPER}
Actions:
- {{"type": "move_to", "xyz": [x, y, z]}}: move the gripper to this point. The gripper moves in a straight line.
  move_to takes an optional "yaw" in degrees (-90 to 90) that turns the gripper about the vertical axis. \
{YAW} Without "yaw", the gripper keeps its current yaw.
- {{"type": "open"}} and {{"type": "close"}}: open or close the gripper.
- {{"type": "wait", "seconds": s}}: wait for s seconds (at most 3).
{REACH}{GRASP}\
Each result says if the gripper reached the target, was blocked, or could not reach it.
The code stops your action list at the first failure: a blocked or unreachable move, a close on nothing, \
or a dropped object. You then get new images.
Add "expect": {{"holding": "<object>"}} or {{"holding": "nothing"}} to an action to say what the gripper \
should hold after it. A mismatch also stops the list.
To close the gripper on purpose with nothing inside, for example to press or push, \
add "expect": {{"holding": "nothing"}}.

The simulation stops while you think. Use at most 8 actions in each step. \
Write "observation" in 2 short sentences and "plan" in 1 sentence. \
Write "notes" only when something worth remembering changed. \
Keep a "todo" list of 3 to 5 short sub-goals. \
Write a "lessons" item only when an action had a result you did not expect. Write it as cause and effect, \
for example "Closing 9 cm up on the bottle slipped. 6 cm up held." \
Set "done" to true when you are sure the task is complete. The actions in the same reply still run first."""

PLAN_SYSTEM = f"""\
You plan a whole task for a Franka Panda robot arm in a physics simulation, before the arm moves. \
You get two camera images of the start and the state of the gripper. You reply in one message \
with what you believe the scene is, and with the keyframes of the whole task.

{COORDS}{OBJECTS.replace("the step message", "the message")}{GRIPPER}"yaw" is in degrees (-90 to 90) and turns the gripper about the vertical axis. {YAW}
{REACH}{GRASP}
Write your belief in "objects": one item for each object that you see. Name each object as the \
message does, for example "red cube". "shape" is "box" or "cylinder". "size" is [width along x, depth along y, \
height] in metres, before the yaw turn. "xyz" is the centre. "yaw" is the turn of the object in degrees, \
counter-clockwise seen from above.
Write the task as "keyframes", at most 40, in order. Each keyframe is a gripper point "xyz" and a "yaw". \
The arm moves in a straight line from one keyframe to the next. Then it opens or closes the gripper \
if "gripper" changes. The gripper is open at the start, at x 0.30, y 0.00, z 0.40, yaw 0. \
"holding" names the object that the gripper holds after the keyframe, or "nothing". \
Give each keyframe a short "note", for example "above the red cube". \
Add "moves" to a keyframe when the hand pushes, slides or knocks an object without holding it, \
and for any other object that you expect to move. Each item is {{"object": <name>, "xyz": [x, y, z], \
"yaw": <degrees>}}: the pose of the object at the end of the keyframe. Do not list the held object. \
Write a "summary" of the plan in 1 or 2 sentences.
The code scores your belief against the true scene and checks each keyframe. It can then run \
the keyframes once in the physics. You get no images while they run.
A re-plan message also has your last plan, the result of its execution and two end images. \
Compare your last plan with the end images and find what went wrong. Then give a new plan for the whole task \
from the start. The scene is back at its start.
Write a "lessons" item only in a re-plan, and only when the last plan had a result you did not expect. \
Write it as cause and effect, for example "A grasp 2 cm above the cube centre slipped. 1 cm held.\""""

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
                    "yaw": {"type": "number"},
                    "expect": {"type": "object", "additionalProperties": False, "required": ["holding"],
                               "properties": {"holding": {"type": "string"}}},
                },
            },
        },
        "notes": {"type": "string"},
        "todo": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "done": {"type": "boolean"},
        "lessons": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
}


_VEC3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
PLAN_REPLY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["objects", "keyframes", "summary"],
    "properties": {
        "objects": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "shape", "size", "xyz", "yaw"],
                "properties": {
                    "name": {"type": "string"},
                    "shape": {"enum": ["box", "cylinder"]},
                    "size": _VEC3,
                    "xyz": _VEC3,
                    "yaw": {"type": "number"},
                },
            },
        },
        "keyframes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 40,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["xyz", "yaw", "gripper", "holding", "note"],
                "properties": {
                    "xyz": _VEC3,
                    "yaw": {"type": "number"},
                    "gripper": {"enum": ["open", "closed"]},
                    "holding": {"type": "string"},
                    "note": {"type": "string"},
                    "moves": {"type": "array", "maxItems": 4, "items": {
                        "type": "object", "additionalProperties": False, "required": ["object", "xyz", "yaw"],
                        "properties": {"object": {"type": "string"}, "xyz": _VEC3, "yaw": {"type": "number"}}}},
                },
            },
        },
        "summary": {"type": "string"},
        "lessons": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
}


@dataclass(frozen=True)
class StepSummary:
    step: int
    plan: str
    results: tuple[str, ...]
    stopped: str | None = None  # e.g. "Stopped after action 3 of 6: the gripper closed on nothing."


IMAGE_TEXT = "Image 1: front camera. Image 2: top camera."


def build_user_text(scene: str, instruction: str, step: int, max_steps: int, obs: Observation, notes: str,
                    todo: tuple[str, ...], history: list[StepSummary], check_failed: bool = False,
                    image_text: str | None = None, lessons: list[str] | tuple[str, ...] = (),
                    stopped: str | None = None, skipped: bool = True) -> str:
    lines = [
        f"Scene: {scene}",
        f"Task: {instruction}",
        f"Step {step} of {max_steps}.",
        f"Gripper at {fmt(obs.tcp)}, yaw {round(obs.gripper_yaw)}°, width {obs.gripper_width:.3f} m.",
        f"Gripper touching: {', '.join(obs.contacts) if obs.contacts else 'nothing'}.",
    ]
    if stopped:
        head = "Your last action list stopped early" if skipped else "Your last action failed"
        lines.append(f"{head}: {stopped}. Look at the images and plan again. Write a lesson if it helps.")
    if obs.objects:
        yaws = obs.object_yaws or {}
        lines.append("Object centres (debug): " + "; ".join(
            f"{n} {fmt(p)}" + (f" yaw {round(yaws[n])}°" if n in yaws else "") for n, p in obs.objects.items()))
    if obs.joints:
        lines.append(f"Moving parts (debug): {'; '.join(obs.joints)}.")
    if obs.events:
        lines.append(f"Events: {'; '.join(obs.events)}.")
    if history:
        lines.append("Recent steps:")
        lines += [f"- step {h.step}: {h.plan} -> {'; '.join(h.results) or 'no actions'}"
                  + (f". {h.stopped}" if h.stopped else "") for h in history]
    unique = list(dict.fromkeys(lessons))
    lines += ["Lessons:", *(f"- {t}" for t in unique)] if unique else ["Lessons: (none)"]
    lines.append(f"Your notes: {notes or '(empty)'}")
    lines.append("Your to-do list: " + ("; ".join(todo) if todo else "(empty)"))
    if check_failed:
        lines.append("The automatic check says the task is not finished yet.")
    lines.append(image_text or IMAGE_TEXT)
    return "\n".join(lines)


def example_step_text() -> str:
    """A sample step message with invented values, so that the page can show how the code fills it in."""
    from armmind.scenes import SCENES
    cubes = SCENES["cubes"]
    obs = Observation(b"", b"", (0.3, 0.0, 0.4), 0.08, ())
    return build_user_text(cubes.describe(), cubes.tasks["stack"].instruction, 1, 30, obs, "", (), [])


@dataclass(frozen=True)
class Feedback:
    """What the last plan did in the physics, for a re-plan. It holds only what the model wrote, what the arm
    reported and the end images. It never holds the true poses or the scores of the page."""
    objects: tuple[dict, ...]  # the last belief, in the reply format
    keyframes: tuple[dict, ...]  # the last keyframes, in the reply format
    status: str  # the status of the execution, for example "stopped"
    check: bool  # the task has an automatic check
    success: bool
    stopped: dict | None  # {"after", "of", "reason"} when the execution stopped early
    stopped_action: str | None  # the action that stopped, for example "close"
    images: tuple[bytes, ...]  # the end images: front (or side), then top. Empty when they are missing
    parent: str  # the run id of the last plan
    round: int  # the round of the new plan
    end_step: int | None  # the step of the end images in the parent run
    ran: int = 0  # the actions that started before a cancel
    total: int = 0  # the actions of the whole plan


def _feedback_lines(f: Feedback, first_camera: str) -> list[str]:
    last = json.dumps({"objects": list(f.objects), "keyframes": list(f.keyframes)}, separators=(",", ":"),
                      ensure_ascii=False)
    lines = [f"This is a re-plan, round {f.round}. Your last plan ran once in the physics.",
             f"Your last plan: {last}",
             f"Execution status: {f.status}."]
    if f.status not in ("cancelled", "error"):  # the check runs only at the end of a whole execution
        lines.append(f"The automatic check says the task is {'solved' if f.success else 'not solved'}."
                     if f.check else "There is no automatic check for this task.")
    if f.status == "cancelled":
        lines.append(f"The person stopped the execution after action {f.ran} of {f.total}.")
    elif f.status == "error":
        lines.append("The execution failed with an error in the simulation.")
    elif f.stopped:
        action = f" ({f.stopped_action})" if f.stopped_action else ""
        lines.append(f"The execution stopped at action {f.stopped['after']} of {f.stopped['of']}{action}: "
                     f"{f.stopped['reason']}.")
    else:
        lines.append("All actions ran.")
    if len(f.images) == 2:
        lines.append(f"Images 3 and 4 show the end of the last execution. The scene is now back at its start, "
                     f"as in images 1 and 2. Image 3: the {first_camera} camera at the end of the last execution. "
                     "Image 4: the top camera at the end.")
    else:
        lines.append("There are no images of the end of the last execution.")
    return lines


def build_plan_text(scene: str, instruction: str, obs: Observation, image_text: str | None = None,
                    lessons: list[str] | tuple[str, ...] = (), feedback: Feedback | None = None) -> str:
    """The plan message: the scene, the task, the gripper and the lessons, as in step 1. A re-plan adds the
    feedback part. It never has the true positions, yaws or joints, even when the observation has them."""
    lines = [
        f"Scene: {scene}",
        f"Task: {instruction}",
        "Plan the whole task now, in one reply.",
        f"Gripper at {fmt(obs.tcp)}, yaw {round(obs.gripper_yaw)}°, width {obs.gripper_width:.3f} m.",
        f"Gripper touching: {', '.join(obs.contacts) if obs.contacts else 'nothing'}.",
    ]
    unique = list(dict.fromkeys(lessons))
    lines += ["Lessons:", *(f"- {t}" for t in unique)] if unique else ["Lessons: (none)"]
    images = image_text or IMAGE_TEXT
    lines.append(images)
    if feedback is not None:
        lines += _feedback_lines(feedback, "side" if "side camera" in images else "front")
    return "\n".join(lines)
