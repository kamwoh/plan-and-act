"""The interface between the agent and a world, and the MuJoCo version of it."""
from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import numpy as np

from armmind.sim.arm import Arm
from armmind.sim.render import Cameras, jpeg, png
from armmind.sim.world import CUBES, World
from armmind.tasks import Task, random_layout
from armmind.types import Action, ActionResult, Observation


class Env(Protocol):
    instruction: str
    has_check: bool
    check_name: str | None

    def observe(self) -> Observation: ...
    def act(self, action: Action) -> ActionResult: ...
    def success(self) -> bool: ...
    def close(self) -> None: ...


def _r3(p) -> tuple[float, float, float]:
    return (round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3))


class MujocoEnv:
    """Make it and use it on one thread only (see SimThread)."""

    def __init__(self, task: Task | None, seed: int, instruction: str, show_positions: bool = False,
                 on_frame: Callable[[str, bytes], None] | None = None):
        self.task, self.instruction, self.show_positions = task, instruction, show_positions
        self.has_check, self.check_name = task is not None, task.name if task else None
        self.world = World()
        self.world.reset(random_layout(seed))
        self.cams = Cameras(self.world)
        self._on_frame, self._n_frames = on_frame, 0
        self.arm = Arm(self.world, on_frame=self._frame if on_frame else None)
        self.arm.reset_ready()

    def _frame(self) -> None:
        """Send the front camera on every frame and the top camera on every third frame. No shadows, for speed."""
        self._on_frame("front", jpeg(self.cams.render("front", shadows=False)))
        if self._n_frames % 3 == 2:
            self._on_frame("top", jpeg(np.asarray(self.cams.with_grid("top", shadows=False))))
        self._n_frames += 1

    def observe(self) -> Observation:
        objects = {n: _r3(self.world.cube_pos(n)) for n in CUBES} if self.show_positions else None
        return Observation(
            front_png=png(self.cams.with_grid("front")), top_png=png(self.cams.with_grid("top")),
            tcp=_r3(self.world.tcp()), gripper_width=round(self.world.gripper_width(), 3),
            contacts=tuple(self.world.gripper_contacts()), objects=objects)

    def act(self, action: Action) -> ActionResult:
        return self.arm.act(action)

    def success(self) -> bool:
        return self.task is not None and bool(self.task.success(self.world))

    def close(self) -> None:
        self.cams.close()


def render_preview(seed: int) -> dict:
    """The starting scene of a seed as base64 images. Call it on a fresh SimThread."""
    env = MujocoEnv(None, seed, "")
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
