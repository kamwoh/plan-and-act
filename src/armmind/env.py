"""The interface between the agent and a world, and the MuJoCo version of it."""
from __future__ import annotations

import asyncio
import base64
import bisect
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import mujoco
import numpy as np

from armmind.agent.rules import check_result
from armmind.plan.formats import Keyframe
from armmind.scenes import Scene, Task
from armmind.sim import state as sim_state
from armmind.sim.arm import GRIP_CLOSED, Arm
from armmind.sim.render import Cameras, jpeg, png
from armmind.sim.world import World
from armmind.types import Action, ActionResult, Observation


class Env(Protocol):
    instruction: str
    scene_name: str
    scene_text: str  # the scene description for the step message
    image_text: str | None  # names the two images in the step message; None means the default text
    has_check: bool
    check_name: str | None
    object_labels: dict[str, str]  # scene object name -> its label, e.g. "red_cube" -> "red cube, 4 cm"

    def observe(self) -> Observation: ...
    def act(self, action: Action) -> ActionResult: ...
    def success(self) -> bool: ...
    def close(self) -> None: ...


def keyframe_actions(keyframes, state: str = "open") -> list[Action]:
    """The actions of a plan: a move_to for each keyframe, then an open or a close when the gripper state
    changes. The open or close expects what the keyframe holds, or nothing."""
    out = []
    for k in keyframes:
        out.append(Action("move_to", tuple(k.xyz), yaw=k.yaw))
        if k.state != state:
            out.append(Action("close" if k.state == "closed" else "open", expect=k.carrying or "nothing"))
            state = k.state
    return out


def keyframe_ends(keyframes, state: str = "open") -> list[int]:
    """The index in keyframe_actions of the last action of each keyframe."""
    out, n = [], 0
    for k in keyframes:
        n += 1 if k.state == state else 2
        state = k.state
        out.append(n - 1)
    return out


def _r3(p) -> tuple[float, float, float]:
    return (round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3))


class MujocoEnv:
    """Make it and use it on one thread only (see SimThread). With `start`, a saved state file, the env starts
    from that state in place of the reset of the seed."""

    def __init__(self, scene: Scene, task: Task | None, seed: int, instruction: str, show_positions: bool = False,
                 on_frame: Callable[[str, bytes], None] | None = None, start: Path | None = None):
        self.scene, self.seed = scene, seed
        self.task, self.instruction, self.show_positions = task, instruction, show_positions
        self.scene_name, self.scene_text, self.image_text = scene.name, scene.describe(), scene.image_text
        self.has_check, self.check_name = task is not None, task.name if task else None
        self.object_labels = {o.name: o.label for o in scene.objects}
        self.world = World(scene)
        saved = sim_state.load(start, scene, seed) if start is not None else None
        self.world.reset(scene.layout(seed), task.initial_qpos if task else None)
        self.cams = Cameras(self.world)
        self._on_frame, self._n_frames = on_frame, 0
        self.arm = Arm(self.world, on_frame=self._frame if on_frame else None)
        if saved is None:
            self.arm.reset_ready()
            self._seen = 0  # events before this index went out in an earlier observation
            return
        sim_state.apply(saved, self.world.model, self.world.data)
        self.arm.resume(float(saved["yaw"]))
        self.world.resume([str(e) for e in saved["events"]])
        self._seen = len(self.world.events)  # the events of the last execution went out then

    def save_state(self, path: Path) -> None:
        """Save the full state now, for a round that continues from here. The forward pass first brings the
        poses up to date after the last step, so that this env and an env from the file show the same frame."""
        mujoco.mj_forward(self.world.model, self.world.data)
        sim_state.save(path, self.scene, self.seed, self.world.data, self.arm.yaw, self.world.events)

    def gripper_state(self) -> str:
        """"closed" when the last gripper command was a close, else "open"."""
        return "closed" if self.world.data.ctrl[7] == GRIP_CLOSED else "open"

    def _frame(self) -> None:
        """Send the front camera on every frame and the top camera on every third frame. No shadows, for speed."""
        self._on_frame("front", jpeg(self.cams.render("front", shadows=False)))
        if self._n_frames % 3 == 2:
            self._on_frame("top", jpeg(np.asarray(self.cams.with_grid("top", shadows=False))))
        self._n_frames += 1

    def observe(self) -> Observation:
        objects = {o.name: _r3(self.world.object_pos(o.name)) for o in self.world.objects} if self.show_positions else None
        yaws = ({o.name: round(self.world.object_yaw(o.name), 1) for o in self.world.objects}
                if self.show_positions else None)
        joints = (tuple(self.world.scene.joints[n].format(q) for n, q in self.world.joints().items())
                  if self.show_positions else None)
        events, self._seen = tuple(self.world.events[self._seen:]), len(self.world.events)
        return Observation(
            front_png=png(self.cams.with_grid("front")), top_png=png(self.cams.with_grid("top")),
            tcp=_r3(self.world.tcp()), gripper_width=round(self.world.gripper_width(), 3),
            contacts=tuple(self.world.gripper_contacts()), objects=objects,
            gripper_yaw=round(self.world.gripper_yaw(), 1), events=events, joints=joints, object_yaws=yaws)

    def act(self, action: Action) -> ActionResult:
        return self.arm.act(action)

    def success(self) -> bool:
        return self.task is not None and bool(self.task.success(self.world))

    def cameras(self) -> list[dict]:
        """The pose, the field of view and the image size of each camera, for the path overlay of the page."""
        from armmind.sim.planview import cameras
        return cameras(self.world, self.cams.w, self.cams.h)

    def plan_view(self, belief, keyframes, marks: list[dict] | None = None) -> dict:
        """The PlanView of a belief and its keyframes, from the true scene now (call it before any motion)."""
        from armmind.sim.planview import build_view
        return build_view(self.world, belief, keyframes, marks, self.cams.w, self.cams.h, self.gripper_state(),
                          self.arm.held())

    def check_plan(self, belief, keyframes) -> dict:
        """The scores of a plan against the true scene now (call it before any motion)."""
        from armmind.sim.planview import check_plan
        return check_plan(self.world, self.arm, belief, keyframes, self.gripper_state())

    def execute_plan(self, keyframes: tuple[Keyframe, ...], should_stop: Callable[[], bool],
                     on_action: Callable[[int, int], None] | None = None) -> dict:
        """Run the keyframes once, with the stop rules of a step. should_stop() between actions cancels the
        rest. on_action(index, count) runs before each action. The auto-check runs at the end when the
        instruction is a task of the scene. "true_start" and "true_after" hold the true object poses before the
        plan and after each keyframe that ran. "frames" holds one front camera frame at the end of each keyframe
        that ran, and one at a stop: {"keyframe" (from 1), "of", "note", "at" ("end" or "stop"), "png"}."""
        actions = keyframe_actions(keyframes, self.gripper_state())
        last_actions = keyframe_ends(keyframes, self.gripper_state())
        ends = set(last_actions)
        lines, stopped, cancelled, frames = [], None, False, []
        true_start, true_after = self._true_poses(), []  # the poses after each keyframe that ran
        for i, action in enumerate(actions):
            if should_stop():
                cancelled = True
                break
            if on_action is not None:
                on_action(i, len(actions))
            result, why = check_result(action, self.act(action), self.object_labels)
            lines.append(result.summary())
            if i in ends:
                true_after.append(self._true_poses())
            if i in ends or why is not None:
                k = bisect.bisect_left(last_actions, i)  # the keyframe of action i
                frames.append({"keyframe": k + 1, "of": len(keyframes), "note": keyframes[k].note,
                               "at": "end" if why is None else "stop", "png": png(self.cams.with_grid("front"))})
            if why is not None:
                stopped = {"after": i + 1, "of": len(actions), "keyframe": k + 1, "reason": why}
                lines += [f"{a.describe()}: skipped" for a in actions[i + 1:]]
                break
        success = self.has_check and not cancelled and self.success()
        status = ("cancelled" if cancelled else "success" if success else "stopped" if stopped
                  else "failed" if self.has_check else "done")
        return {"actions": [a.describe() for a in actions], "results": lines, "stopped": stopped,
                "success": bool(success), "status": status, "true_start": true_start, "true_after": true_after,
                "frames": frames}

    def _true_poses(self) -> dict[str, dict]:
        """The true pose of each scene object now: {"xyz", "yaw"}."""
        return {o.name: {"xyz": [round(float(v), 5) for v in self.world.object_pos(o.name)],
                         "yaw": round(float(self.world.object_yaw(o.name)), 2)} for o in self.world.objects}

    def close(self) -> None:
        self.cams.close()


def render_preview(scene: Scene, seed: int) -> dict:
    """The starting scene of a seed as base64 images. Call it on a fresh SimThread."""
    env = MujocoEnv(scene, None, seed, "")
    try:
        return {"front": base64.b64encode(jpeg(env.cams.render("front"))).decode(),
                "top": base64.b64encode(png(env.cams.with_grid("top"))).decode()}
    finally:
        env.close()


class SimThread:
    """Runs every call on one worker thread. MuJoCo's EGL renderer works only on the thread that made it."""

    def __init__(self):
        self._ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sim")

    async def call(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self._ex, fn, *args)

    def shutdown(self) -> None:
        self._ex.shutdown(wait=True)
