"""Test doubles for the env and the model."""
from __future__ import annotations

import asyncio

from armmind.llm.protocol import Final, LLMError, Usage
from armmind.types import Action, ActionResult, Observation


class FakeEnv:
    instruction = "Touch the red cube with the gripper."
    scene_name = "cubes"
    scene_text = "Objects: red cube, 4 cm; blue cube, 4 cm."
    image_text = None

    def __init__(self, succeed_after: int | None = None, has_check: bool = True):
        self.has_check = has_check
        self.check_name = "reach" if has_check else None
        self.acts: list[Action] = []
        self.succeed_after = succeed_after
        self.closed = False

    def observe(self) -> Observation:
        return Observation(b"front", b"top", (0.3, 0.0, 0.4), 0.08, ())

    def act(self, action: Action) -> ActionResult:
        self.acts.append(action)
        return ActionResult(action, "reached", (0.3, 0.0, 0.4), 0.08)

    def success(self) -> bool:
        return self.succeed_after is not None and len(self.acts) >= self.succeed_after

    def close(self) -> None:
        self.closed = True


def good(actions=({"type": "close"},), **extra):
    body = {"observation": "o", "plan": "p", "actions": list(actions), "todo": ["t"], **extra}
    return [Final("", body), Usage("fake", 0.1)]


def bad():
    return [LLMError("schema", "answer did not match")]


async def direct(fn, *args):
    return fn(*args)


class Recorder:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def __call__(self, kind: str, data: dict) -> None:
        self.events.append((kind, data))

    def kinds(self) -> list[str]:
        return [k for k, _ in self.events]


class HangingConversation:
    def __init__(self):
        self.cancelled = asyncio.Event()

    async def send(self, msg, schema=None):
        await self.cancelled.wait()
        yield LLMError("cancelled", "cancelled")

    async def cancel(self):
        self.cancelled.set()

    async def close(self):
        pass


class HangingBackend:
    name = "hanging"

    async def open(self, system):
        self.conv = HangingConversation()
        return self.conv


class DeafConversation(HangingConversation):
    async def cancel(self):
        pass


class DeafBackend:
    name = "deaf"

    async def open(self, system):
        self.conv = DeafConversation()
        return self.conv
