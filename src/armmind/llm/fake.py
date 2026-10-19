"""Scripted backend for tests: replays prepared event lists and records what was sent."""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator

from armmind.llm.protocol import Final, LLMError, LLMEvent, Message, Usage


def lost_chat(session: str) -> LLMError:
    return LLMError("resume", f"No conversation found with session ID: {session}")


def _reply_text(ev: Final) -> str:
    return json.dumps(ev.parsed, ensure_ascii=False) if ev.parsed is not None else ev.text


class Chats:
    """The saved chats of a fake or demo backend, as the CLI keeps its sessions: session id -> messages."""

    def __init__(self, prefix: str):
        self.prefix, self.chats, self.count = prefix, {}, 0

    def open(self, chain: str | None, resume: str | None) -> tuple[str | None, list[Message], str | None]:
        """The session id, the messages so far and the id of a lost session. A chat without a chain is not saved."""
        if chain is None:
            return None, [], None
        if resume is not None:
            return (resume, self.chats[resume], None) if resume in self.chats else (None, [], resume)
        self.count += 1
        session = f"{self.prefix}-{self.count}"
        self.chats[session] = []
        return session, self.chats[session], resume


class FakeConversation:
    def __init__(self, backend: FakeBackend, chain: str | None = None, resume: str | None = None):
        self.backend = backend
        self.session_id, self.history, self._lost = backend.store.open(chain, resume)

    async def send(self, msg: Message, schema: dict | None = None) -> AsyncIterator[LLMEvent]:
        self.backend.sent.append((msg, schema))
        if self._lost is not None:  # the CLI fails before the model sees the message
            yield lost_chat(self._lost)
            return
        self.history.append(msg)
        self.backend.seen.append(tuple(self.history))
        if not self.backend.replies:
            yield LLMError("backend", "fake backend has no more replies")
            return
        for ev in self.backend.replies.pop(0):
            if isinstance(ev, Final):
                self.history.append(Message("assistant", _reply_text(ev)))
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
        self.seen: list[tuple[Message, ...]] = []  # for each call that reached the model: its whole chat
        self.systems: list[str] = []
        self.opens: list[dict] = []
        self.store = Chats("fake-session")
        self.cancelled = 0
        self.closed = 0

    @property
    def chats(self) -> dict[str, list[Message]]:
        return self.store.chats

    async def open(self, system: str, chain: str | None = None, resume: str | None = None) -> FakeConversation:
        self.systems.append(system)
        self.opens.append({"system": system, "chain": chain, "resume": resume})
        return FakeConversation(self, chain, resume)


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


# Round 1 of the demo lifts the red cube too little, so the pick check fails. The re-plan lifts it enough.
DEMO_PLAN_LOW = {**DEMO_PLAN, "keyframes": [*DEMO_PLAN["keyframes"][:-1],
                                            {**DEMO_PLAN["keyframes"][-1], "xyz": [0.56, -0.11, 0.08]}],
                 "summary": "Push the green cube 5 cm to the left, then grasp the red cube and lift it 6 cm."}
DEMO_LESSON = "A lift to z 0.08 left the cube under the 10 cm goal. Lift to z 0.20."
DEMO_REPLAN = {**DEMO_PLAN, "lessons": [DEMO_LESSON]}
# A re-plan that continues: the red cube is still in the hand, 6 cm up. Lift it higher.
DEMO_CONTINUE = {
    "objects": [{**DEMO_PLAN["objects"][0], "xyz": [0.56, -0.11, 0.07]}, DEMO_PLAN["objects"][1],
                {**DEMO_PLAN["objects"][2], "xyz": [0.6, 0.16, 0.02]}],
    "keyframes": [{"xyz": [0.56, -0.11, 0.2], "yaw": 0, "gripper": "closed", "holding": "red cube",
                   "note": "lift it higher"}],
    "summary": "The red cube is still in the hand. Lift it to z 0.20.",
    "lessons": [DEMO_LESSON],
}


# A re-plan of a task without an auto-check: the images show that it is complete.
DEMO_DONE = {"objects": DEMO_CONTINUE["objects"], "keyframes": [], "done": True,
             "summary": "The images show that the task is complete."}


def demo_plan(text: str) -> dict:
    """The demo reply to a plan message."""
    if "This is a re-plan" not in text:
        return DEMO_PLAN_LOW
    if "There is no automatic check" in text:
        return DEMO_DONE
    return DEMO_CONTINUE if "did not reset" in text else DEMO_REPLAN


def demo_verdict(text: str, last_plan: dict | None = None) -> dict:
    """The demo reply to a verdict call: not done when the last keyframe holds an object less than 10 cm up.
    The plan is in the message of a new chat. Else it is the last plan of the chat (`last_plan`)."""
    line = next((x for x in text.splitlines() if x.startswith("The plan that ran: ")), "")
    try:
        plan = json.loads(line.removeprefix("The plan that ran: ")) if line else last_plan
        last = plan["keyframes"][-1]
        low = last.get("holding", "nothing") != "nothing" and float(last["xyz"][2]) < 0.1
    except (ValueError, KeyError, IndexError, TypeError):
        low = False
    if low:
        return {"done": False, "reason": "The cube is in the gripper, but less than 10 cm above the table."}
    return {"done": True, "reason": "The images show that the task is complete."}


class DemoConversation:
    def __init__(self, backend: DemoBackend, chain: str | None = None, resume: str | None = None):
        self.backend = backend
        self.session_id, self.history, self._lost = backend.store.open(chain, resume)

    def _last_plan(self) -> dict | None:
        for m in reversed(self.history):
            if m.role == "assistant" and '"keyframes"' in m.text:
                return json.loads(m.text)
        return None

    async def send(self, msg: Message, schema: dict | None = None) -> AsyncIterator[LLMEvent]:
        await asyncio.sleep(self.backend.delay_s)
        if self._lost is not None:  # the demo keeps its chats in memory: a server restart loses them
            yield lost_chat(self._lost)
            return
        self.history.append(msg)
        if schema is not None and set(schema.get("required", [])) == {"done", "reason"}:
            final, usage = Final("", demo_verdict(msg.text, self._last_plan())), Usage(
                "demo", self.backend.delay_s, 1200, 40, 0.002, 0, 0)
        elif schema is not None and "keyframes" in schema.get("properties", {}):
            final, usage = Final("", demo_plan(msg.text)), Usage("demo", self.backend.delay_s, 2400, 400, 0.006, 0, 0)
        else:
            step = re.search(r"Step (\d+) of", msg.text)  # the same reply for the same step in each run
            body = DEMO_REPLIES[(int(step.group(1)) - 1 if step else 0) % len(DEMO_REPLIES)]
            final, usage = Final("", body), Usage("demo", self.backend.delay_s, 1800, 120, 0.004, 0, 0)
        self.history.append(Message("assistant", _reply_text(final)))
        yield final
        yield usage

    async def cancel(self) -> None:
        pass

    async def close(self) -> None:
        pass


class DemoBackend:
    """A model stand-in for a manual check of the page. It waits, then gives the next invented reply."""
    name = "demo"

    def __init__(self, delay_s: float = 2.0):
        self.delay_s = delay_s
        self.store = Chats("demo-session")

    async def open(self, system: str, chain: str | None = None, resume: str | None = None) -> DemoConversation:
        return DemoConversation(self, chain, resume)
