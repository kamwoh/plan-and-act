"""Scripted backend for tests: replays prepared event lists and records what was sent."""
from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator

from armmind.llm.protocol import Final, LLMError, LLMEvent, Message, Usage


class FakeConversation:
    def __init__(self, backend: FakeBackend):
        self.backend = backend

    async def send(self, msg: Message, schema: dict | None = None) -> AsyncIterator[LLMEvent]:
        self.backend.sent.append((msg, schema))
        if not self.backend.replies:
            yield LLMError("backend", "fake backend has no more replies")
            return
        for ev in self.backend.replies.pop(0):
            yield ev

    async def cancel(self) -> None:
        self.backend.cancelled += 1

    async def close(self) -> None:
        self.backend.closed += 1


class FakeBackend:
    name = "fake"

    def __init__(self, replies: list[list[LLMEvent]]):
        self.replies = list(replies)
        self.sent: list[tuple[Message, dict | None]] = []
        self.systems: list[str] = []
        self.cancelled = 0
        self.closed = 0

    async def open(self, system: str) -> FakeConversation:
        self.systems.append(system)
        return FakeConversation(self)


DEMO_REPLIES = [  # invented replies for a demo run of the cubes scene
    {"observation": "The red cube is left of the centre. The gripper is high above the table.",
     "plan": "Open and move above the red cube.",
     "actions": [{"type": "open"}, {"type": "move_to", "xyz": [0.5, -0.1, 0.15]}],
     "todo": ["move above the red cube", "grasp it", "lift it"]},
    {"observation": "The gripper is above the red cube.", "plan": "Go down and close on the cube.",
     "actions": [{"type": "move_to", "xyz": [0.5, -0.1, 0.02]}, {"type": "close", "expect": {"holding": "red cube"}}],
     "todo": ["grasp it", "lift it"], "notes": "The cube is at about x 0.50, y -0.10.",
     "lessons": ["Closing 2 cm up on a 4 cm cube held it."]},
    {"observation": "The gripper holds something.", "plan": "Lift the cube.",
     "actions": [{"type": "move_to", "xyz": [0.5, -0.1, 0.15]}], "todo": ["lift it"]},
]


DEMO_PLAN = {  # an invented plan for seed 0 of the cubes scene: push the green cube, then pick up the red cube
    "objects": [
        {"name": "red cube", "shape": "box", "size": [0.04, 0.04, 0.04], "xyz": [0.56, -0.11, 0.02], "yaw": 0},
        {"name": "blue cube", "shape": "box", "size": [0.04, 0.04, 0.04], "xyz": [0.41, -0.24, 0.02], "yaw": 10},
        {"name": "green cube", "shape": "box", "size": [0.04, 0.04, 0.04], "xyz": [0.60, 0.21, 0.02], "yaw": 0},
    ],
    "keyframes": [
        {"xyz": [0.6, 0.27, 0.15], "yaw": 0, "gripper": "closed", "holding": "nothing",
         "note": "above, right of the green cube, fingers closed"},
        {"xyz": [0.6, 0.27, 0.02], "yaw": 0, "gripper": "closed", "holding": "nothing", "note": "down beside it"},
        {"xyz": [0.6, 0.19, 0.02], "yaw": 0, "gripper": "closed", "holding": "nothing", "note": "push it 5 cm",
         "moves": [{"object": "green cube", "xyz": [0.6, 0.16, 0.02], "yaw": 0}]},
        {"xyz": [0.6, 0.19, 0.15], "yaw": 0, "gripper": "closed", "holding": "nothing", "note": "up"},
        {"xyz": [0.56, -0.11, 0.15], "yaw": 0, "gripper": "open", "holding": "nothing", "note": "above the red cube"},
        {"xyz": [0.56, -0.11, 0.02], "yaw": 0, "gripper": "closed", "holding": "red cube", "note": "grasp it"},
        {"xyz": [0.56, -0.11, 0.2], "yaw": 0, "gripper": "closed", "holding": "red cube", "note": "lift it"},
    ],
    "summary": "Push the green cube 5 cm to the left, then grasp the red cube from above and lift it 18 cm.",
}


class DemoConversation:
    def __init__(self, backend: DemoBackend):
        self.backend = backend

    async def send(self, msg: Message, schema: dict | None = None) -> AsyncIterator[LLMEvent]:
        await asyncio.sleep(self.backend.delay_s)
        if schema is not None and "keyframes" in schema.get("properties", {}):
            yield Final("", DEMO_PLAN)
            yield Usage("demo", self.backend.delay_s, 2400, 400, 0.006, 0, 0)
            return
        step = re.search(r"Step (\d+) of", msg.text)  # the same reply for the same step in each run
        body = DEMO_REPLIES[(int(step.group(1)) - 1 if step else 0) % len(DEMO_REPLIES)]
        yield Final("", body)
        yield Usage("demo", self.backend.delay_s, 1800, 120, 0.004, 0, 0)

    async def cancel(self) -> None:
        pass

    async def close(self) -> None:
        pass


class DemoBackend:
    """A model stand-in for a manual check of the page. It waits, then gives the next invented reply."""
    name = "demo"

    def __init__(self, delay_s: float = 2.0):
        self.delay_s = delay_s

    async def open(self, system: str) -> DemoConversation:
        return DemoConversation(self)
