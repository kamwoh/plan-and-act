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
the keyframes once in the physics. You get no images while they run. \
After the run, a message can show camera frames from the run, next to the poses that you imagined.
A re-plan message has the result of the execution of your last plan. Your last plan is earlier in this chat. \
When a re-plan message starts a new chat, it also has your last plan. \
Compare your last plan with the images and find what went wrong.
In a re-plan from the start, the scene is back at its start. The message then also has two end images. \
Give a new plan for the whole task.
A re-plan that continues has no end images: the scene did not reset, so plan only what is left, from images 1 and 2 and the gripper now.
In a re-plan, set "done" to true when the images show that the task is complete. Then "keyframes" can be empty. \
Set "impossible" to a short reason only when a cause makes the task impossible with this arm and scene, \
for example an object that the scene does not have or a target out of reach. One failed attempt is not a reason. \
Then "keyframes" can be empty. In every other case, give at least 1 keyframe.
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
        "done": {"type": "boolean"},
        "impossible": {"type": "string"},
    },
}


# The verdict rules, shared by the verdict prompt and the prompt of a chain.
JUDGE = """\
Judge only from the images: the plan says what was meant to happen, not what happened.
Set "done" to true only when the images show that the task is complete. Give the "reason" in 1 short sentence."""

VERDICT_SYSTEM = """\
You judge if a robot arm finished a task in a physics simulation. You get the task, the plan that the arm \
ran, and two camera images of the scene after the run. """ + JUDGE

# One chat for a chain: the round 1 plan, then each re-plan and each verdict as the next message.
CHAIN_SYSTEM = PLAN_SYSTEM + """

After an execution, a message can ask for your verdict instead of a plan: did the arm finish the task? \
It has two camera images of the scene after the run. """ + JUDGE

VERDICT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["done", "reason"],
    "properties": {"done": {"type": "boolean"}, "reason": {"type": "string"}},
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
class Frame:
    """A front camera frame of an execution, with the poses that Claude imagined at that point of its plan.
    `imagined` comes from Claude's own plan, never from the true scene. None: the plan has no track."""
    keyframe: int  # from 1
    of: int  # the keyframes of the plan
    note: str  # the note of the keyframe in Claude's plan
    at: str  # "end": the end of the keyframe. "stop": the execution stopped inside it
    imagined: tuple[tuple[str, tuple[float, float, float], float | None], ...] | None  # name, xyz, yaw or None
    png: bytes


def frame_label(f: Frame, camera: str) -> str:
    """For example: 'Front camera after keyframe 2 of 5. Note: "above the cube". You imagined red cube at ...'
    The imagined poses are the poses at the end of the keyframe, also for a stop frame: its label says so."""
    stop = f.at == "stop"
    where = f"at the stop in keyframe {f.keyframe} of {f.of}" if stop else f"after keyframe {f.keyframe} of {f.of}"
    parts = [f"{camera.capitalize()} camera {where}.", f"Note: {json.dumps(f.note, ensure_ascii=False)}."]
    if f.imagined is not None and not f.imagined:
        parts.append(f"At the end of keyframe {f.keyframe}, no object moves in your plan." if stop
                     else "No object moves in your plan up to here.")
    lead = f"At the end of keyframe {f.keyframe}, you imagined" if stop else "You imagined"
    for name, xyz, yaw in f.imagined or ():
        parts.append(f"{lead} {name} at {fmt(xyz)}" + (f", yaw {round(yaw)}°." if yaw is not None else "."))
    return " ".join(parts)


def _frame_lines(frames: tuple[Frame, ...], first: int, camera: str) -> list[str]:
    """The labels of the frames of an execution. The frames are the images from number `first` on."""
    if not frames:
        return []
    last = first + len(frames) - 1
    head = f"Image {first} shows" if first == last else f"Images {first} to {last} show"
    return [f"{head} the {camera} camera during the execution, in order. Each label has the poses that you "
            + "imagined at that point of your plan.",
            *(f"- Image {first + i}: {frame_label(f, camera)}" for i, f in enumerate(frames)),
            "Compare each frame with the poses that you imagined, and name the first keyframe where they differ."]


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
    mode: str = "start"  # "start": the scene is back at its start. "continue": it goes on from the end state
    frames: tuple[Frame, ...] = ()  # the frames for Claude. They go after the images now and before `images`
    ran_keyframes: tuple[int, int] = (0, 0)  # (K, N) when only the first K of N keyframes ran by design
    instruction: str | None = None  # the new instruction of the person, when it replaces the earlier task
    executed: bool = True  # False: the last plan did not run, so there is no result. Only with a new instruction


def partial_line(k: int, n: int) -> str:
    """The sentence for an execution that ran only its first k keyframes ("Run per round")."""
    return f"Only the first {k} of your {n} keyframes ran, by design. The arm is now at the end of keyframe {k}."


def _feedback_lines(f: Feedback, first_camera: str, with_plan: bool = True, start_seen: bool = False) -> list[str]:
    """The result of the last execution. Without `with_plan`, the lines leave out the last plan: the chat has it.
    With `start_seen`, a re-plan from the start sends only the end images: the chat has the start images."""
    lines = _new_instruction_lines(f)
    lines.append(f"This is a re-plan, round {f.round}." + (" Your last plan ran once in the physics." if f.executed
                                                          else ""))
    if with_plan:
        last = json.dumps({"objects": list(f.objects), "keyframes": list(f.keyframes)}, separators=(",", ":"),
                          ensure_ascii=False)
        lines.append(f"Your last plan: {last}")
    if not f.executed:
        return [*lines, *_no_run_lines(f, start_seen)]
    task = "the earlier task" if f.instruction else "the task"
    lines.append(f"Execution status: {f.status}.")
    if f.status not in ("cancelled", "error"):  # the check runs only at the end of a whole execution
        lines.append(f"The automatic check says {task} is {'solved' if f.success else 'not solved'}."
                     if f.check else "There is no automatic check for this task.")
    if f.status == "cancelled":
        lines.append(f"The person stopped the execution after action {f.ran} of {f.total}." if f.ran
                     else f"The person stopped the execution before the first action of {f.total}.")
    elif f.status == "error":
        lines.append("The execution failed with an error in the simulation.")
    elif f.status == "partial":
        lines.append(partial_line(*f.ran_keyframes))
    elif f.stopped:
        action = f" ({f.stopped_action})" if f.stopped_action else ""
        kf = f.stopped.get("keyframe")  # runs from before v1.16 have no keyframe
        where = f" in keyframe {kf} of {len(f.keyframes)}," if isinstance(kf, int) else ""
        lines.append(f"The execution stopped{where} at action {f.stopped['after']} of {f.stopped['of']}{action}: "
                     f"{f.stopped['reason']}.")
    else:
        lines.append("All actions ran.")
    first = 1 if start_seen and f.mode != "continue" else 3  # the number of the first frame
    lines += _frame_lines(f.frames, first, first_camera)
    end = first + len(f.frames)  # the number of the first end image
    if f.mode == "continue":
        lines.append("The scene did not reset. Images 1 and 2 show the scene now, at the end of the last "
                     "execution. " + ("Plan the new task from here." if f.instruction
                                      else "Plan from here, and do only what is left of the task."))
    elif start_seen:
        lines.append("The scene is now back at its start, the same as the start images earlier in this chat.")
        lines.append(f"Image {end}: the {first_camera} camera at the end of the last execution. "
                     f"Image {end + 1}: the top camera at the end." if len(f.images) == 2
                     else "There are no images of the end of the last execution.")
    elif len(f.images) == 2:
        lines.append(f"Images {end} and {end + 1} show the end of the last execution. The scene is now back at its "
                     f"start, as in images 1 and 2. Image {end}: the {first_camera} camera at the end of the last "
                     f"execution. Image {end + 1}: the top camera at the end.")
    else:
        lines.append("There are no images of the end of the last execution.")
    return lines


def _new_instruction_lines(f: Feedback) -> list[str]:
    """The first line of a re-plan with a new instruction of the person, else nothing."""
    if not f.instruction:
        return []
    where = ("The scene continues from where the last execution ended." if f.executed
             else "The scene is the same as at the start of your last plan.")
    return [f"New instruction from the person: {f.instruction.rstrip('.')}. This replaces the earlier task. {where}"]


def _no_run_lines(f: Feedback, start_seen: bool) -> list[str]:
    """The images of a re-plan after a plan that did not run: the scene is as at the start of that plan."""
    if start_seen and f.mode != "continue":
        return ["The scene is the same as the start images earlier in this chat."]
    return ["Images 1 and 2 show the scene now."]


def _gripper_lines(obs: Observation, feedback: Feedback | None, gripper: str | None) -> list[str]:
    """What to plan, and the gripper now. Never the true positions, yaws or joints of the observation."""
    continues = feedback is not None and feedback.mode == "continue"
    what = ("Plan the new task now, in one reply." if feedback is not None and feedback.instruction
            else "Plan what is left of the task now, in one reply." if continues
            else "Plan the whole task now, in one reply.")
    lines = [
        what,
        f"Gripper at {fmt(obs.tcp)}, yaw {round(obs.gripper_yaw)}°, width {obs.gripper_width:.3f} m.",
        f"Gripper touching: {', '.join(obs.contacts) if obs.contacts else 'nothing'}.",
    ]
    if continues and gripper in ("open", "closed"):
        lines.append(f"The gripper is {gripper} now.")
    return lines


def build_plan_text(scene: str, instruction: str, obs: Observation, image_text: str | None = None,
                    lessons: list[str] | tuple[str, ...] = (), feedback: Feedback | None = None,
                    gripper: str | None = None) -> str:
    """The plan message: the scene, the task, the gripper and the lessons, as in step 1. A re-plan adds the
    feedback part. This is the first message of a chat: round 1, or a re-plan in a new chat (the summary).
    It never has the true positions, yaws or joints, even when the observation has them.
    A re-plan that continues also says if the gripper is open or closed (`gripper`)."""
    lines = [f"Scene: {scene}", f"Task: {instruction}", *_gripper_lines(obs, feedback, gripper)]
    unique = list(dict.fromkeys(lessons))
    lines += ["Lessons:", *(f"- {t}" for t in unique)] if unique else ["Lessons: (none)"]
    images = image_text or IMAGE_TEXT
    lines.append(images)
    if feedback is not None:
        lines += _feedback_lines(feedback, "side" if "side camera" in images else "front")
    return "\n".join(lines)


def build_replan_text(obs: Observation, image_text: str | None, feedback: Feedback,
                      gripper: str | None = None, start_seen: bool = False) -> str:
    """A re-plan message in the chat of its chain: only what is new. The chat has the scene, the task, the
    lessons and the last plan. This message has the result of the last execution, the gripper now, the mode
    and the images. It never has the true positions, yaws or joints. With `start_seen`, a re-plan from the
    start sends only the end images, because the chat has the start images of the same seed."""
    images = image_text or IMAGE_TEXT
    seen = start_seen and feedback.mode == "start"
    lines = _feedback_lines(feedback, "side" if "side camera" in images else "front", with_plan=False,
                            start_seen=seen)
    return "\n".join([*lines, *_gripper_lines(obs, feedback, gripper), *([] if seen else [images])])


def _verdict_image_lines(image_text: str | None, frames: tuple[Frame, ...]) -> list[str]:
    """The frames of the run, then the end images. Without frames, the end images are images 1 and 2."""
    images = image_text or IMAGE_TEXT
    if not frames:
        return ["The images show the scene after the run. " + images]
    camera = "side" if "side camera" in images else "front"
    end = len(frames) + 1
    return [*_frame_lines(frames, 1, camera),
            f"Images {end} and {end + 1} show the scene after the run: the {camera} camera, then the top camera."]


def _ran_lines(ran_keyframes: tuple[int, int]) -> list[str]:
    return [partial_line(*ran_keyframes)] if ran_keyframes[0] else []


def build_chain_verdict_text(image_text: str | None = None, frames: tuple[Frame, ...] = (),
                             ran_keyframes: tuple[int, int] = (0, 0)) -> str:
    """The verdict message in the chat of its chain. The chat has the task and the plan that ran.
    `ran_keyframes` is (K, N) after a run of only the first K of N keyframes."""
    return "\n".join(["Your last plan ran once in the physics. Give your verdict now, not a plan.",
                      *_ran_lines(ran_keyframes), *_verdict_image_lines(image_text, frames), "Is the task complete?"])


def build_verdict_text(scene: str, instruction: str, objects: tuple[dict, ...], keyframes: tuple[dict, ...],
                       image_text: str | None = None, frames: tuple[Frame, ...] = (),
                       ran_keyframes: tuple[int, int] = (0, 0)) -> str:
    """The message of the verdict call after an execution: the task, the plan that ran, the frames and the end
    images. It has only what the model wrote and the images: never the true poses, the scores or the auto-check."""
    plan = json.dumps({"objects": list(objects), "keyframes": list(keyframes)}, separators=(",", ":"),
                      ensure_ascii=False)
    return "\n".join([f"Scene: {scene}", f"Task: {instruction}", f"The plan that ran: {plan}",
                      *_ran_lines(ran_keyframes), *_verdict_image_lines(image_text, frames), "Is the task complete?"])
