"""The interface between the agent and a world, and the MuJoCo version of it."""
from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import numpy as np

from armmind.scenes import Scene, Task
from armmind.sim.arm import Arm
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


def _r3(p) -> tuple[float, float, float]:
    return (round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3))


class MujocoEnv:
    """Make it and use it on one thread only (see SimThread)."""

    def __init__(self, scene: Scene, task: Task | None, seed: int, instruction: str, show_positions: bool = False,
                 on_frame: Callable[[str, bytes], None] | None = None):
        self.task, self.instruction, self.show_positions = task, instruction, show_positions
        self.scene_name, self.scene_text, self.image_text = scene.name, scene.describe(), scene.image_text
        self.has_check, self.check_name = task is not None, task.name if task else None
        self.object_labels = {o.name: o.label for o in scene.objects}
        self.world = World(scene)
        self.world.reset(scene.layout(seed), task.initial_qpos if task else None)
        self.cams = Cameras(self.world)
        self._on_frame, self._n_frames = on_frame, 0
        self.arm = Arm(self.world, on_frame=self._frame if on_frame else None)
        self.arm.reset_ready()
        self._seen = 0  # events before this index went out in an earlier observation

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

    def plan_view(self, belief, keyframes, marks: list[dict] | None = None) -> dict:
        """The PlanView of a belief and its keyframes, from the true scene now (call it before any motion)."""
        from armmind.sim.planview import build_view
        return build_view(self.world, belief, keyframes, marks, self.cams.w, self.cams.h)

    def check_plan(self, belief, keyframes) -> dict:
        """The scores of a plan against the true scene now (call it before any motion)."""
        from armmind.sim.planview import check_plan
        return check_plan(self.world, self.arm, belief, keyframes)

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
