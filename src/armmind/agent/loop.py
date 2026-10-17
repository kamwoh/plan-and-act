"""The agent loop: observe, ask the model, act. The physics stands still while the model thinks."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

from armmind.agent.parse import Reply, ReplyError, parse_reply
from armmind.agent.prompts import REPLY_SCHEMA, SYSTEM, StepSummary, build_user_text
from armmind.env import Env
from armmind.llm.protocol import Conversation, Final, LLMBackend, LLMError, Message, Usage
from armmind.store.lessons import LessonStore
from armmind.types import Action, ActionResult, fmt

Emit = Callable[[str, dict], None]
Call = Callable[..., Awaitable]


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 30
    history_steps: int = 5
    max_actions: int = 8


def _usage_dict(u: Usage | None) -> dict | None:
    if u is None:
        return None
    return {"in": u.in_tokens, "cache_write": u.cache_write_tokens, "cache_read": u.cache_read_tokens,
            "out": u.out_tokens, "cost_usd": u.cost_usd, "latency_s": u.latency_s}


def _sum_usage(attempts: list[dict]) -> dict | None:
    """Add up the usage of all attempts of a step (a missing number counts as 0); None if no attempt has usage."""
    parts = [a["usage"] for a in attempts if a["usage"] is not None]
    if not parts:
        return None
    total = {k: sum(p[k] or 0 for p in parts) for k in parts[0]}
    total["cost_usd"], total["latency_s"] = round(total["cost_usd"], 6), round(total["latency_s"], 3)
    return total


CLOSED_ON_NOTHING_M = 0.005  # a close that ends below this width, with no object, closed on nothing


def _resolve(name: str, labels: dict[str, str]) -> str | None:
    """The object name for a name or a label, matched without case. "nothing" gives ""."""
    want = name.strip().lower()
    if want == "nothing":
        return ""
    for obj, label in labels.items():
        if want in {obj.lower(), obj.lower().replace("_", " "), label.lower(), label.split(",")[0].strip().lower()}:
            return obj
    return None


def check_result(action: Action, result: ActionResult, labels: dict[str, str]) -> tuple[ActionResult, str | None]:
    """Compare a result with what the model wanted. Gives the result, maybe with a note, and the reason to stop
    the action list, or None."""
    held = result.held_after if result.held_after is not None else tuple(c for c in result.contacts if c != "table")
    before = result.held_before or ()
    expect = _resolve(action.expect, labels) if action.expect is not None else None
    if action.expect is not None and expect is None:
        note = f"unknown object {action.expect!r} in expect, ignored"
        result = replace(result, note=f"{result.note}, {note}" if result.note else note)
    if action.type == "move_to" and result.status in ("blocked", "unreachable"):
        return result, f"the move to {fmt(action.xyz)} was {result.status}"
    if action.type == "close" and not held and result.gripper_width < CLOSED_ON_NOTHING_M and expect != "":
        return result, "the gripper closed on nothing"
    if action.type != "open" and (dropped := [n for n in before if n not in held]):
        why = f"dropped {', '.join(dropped)}"
        return replace(result, note=f"{result.note}, {why}" if result.note else why), why
    if expect is not None and (expect not in held if expect else bool(held)):
        return result, f"expected to hold {expect or 'nothing'}, holds {', '.join(held) or 'nothing'}"
    return result, None


class Agent:
    def __init__(self, backend: LLMBackend, config: AgentConfig | None = None, system: str | None = None,
                 store: LessonStore | None = None, use_lessons: bool = True, run_id: str = ""):
        self.backend, self.cfg = backend, config or AgentConfig()
        self.store, self.use_lessons, self.run_id = store, use_lessons, run_id
        self.saved_lessons: tuple[str, ...] = ()  # from earlier runs of the scene, oldest first
        self.run_lessons: list[str] = []  # the new lessons of this run, oldest first, without duplicates
        self.system = system or SYSTEM
        self._stop = asyncio.Event()
        self._conv: Conversation | None = None

    async def stop(self) -> None:
        self._stop.set()
        if self._conv is not None:
            await self._conv.cancel()

    async def run(self, env: Env, call: Call, emit: Emit) -> str:
        history: deque[StepSummary] = deque(maxlen=self.cfg.history_steps)
        self.run_lessons = []
        self.saved_lessons = tuple(self.store.texts(env.scene_name)) if self.store and self.use_lessons else ()
        notes, todo, status, check_failed, stopped_reason = "", (), "step_limit", False, None
        labels = getattr(env, "object_labels", None) or {}
        emit("run_start", {"scene": env.scene_name, "instruction": env.instruction, "max_steps": self.cfg.max_steps,
                           "check": env.check_name, "system_prompt": self.system,
                           "use_lessons": self.use_lessons, "saved_lessons": list(self.saved_lessons)})
        for n in range(1, self.cfg.max_steps + 1):
            if self._stop.is_set():
                status = "stopped"
                break
            obs = await call(env.observe)
            emit("observation", {"step": n, "front_png": obs.front_png, "top_png": obs.top_png})
            text = build_user_text(env.scene_text, env.instruction, n, self.cfg.max_steps, obs, notes, todo, list(history),
                                   check_failed, env.image_text, [*self.saved_lessons, *self.run_lessons],
                                   stopped_reason)
            self._conv = await self.backend.open(self.system)
            attempts: list[dict] = []
            try:
                reply, error, usage = await self._ask(Message("user", text, (obs.front_png, obs.top_png)), attempts)
            finally:
                conv, self._conv = self._conv, None
                await conv.close()
            if self._stop.is_set():
                status = "stopped"
                break
            detail = {"prompt": text, "attempts": attempts, "usage": _sum_usage(attempts)}
            if reply is None and error is not None and error.kind != "schema":
                emit("llm_error", {"step": n, "error_kind": error.kind, "message": error.message, **detail})
                status = "llm_error"
                break
            if reply is None:
                emit("bad_reply", {"step": n, "error": error.message if error else "no reply", **detail})
                continue
            emit("reply", {"step": n, "plan": reply.plan, "actions": [a.describe() for a in reply.actions]})
            results, stopped = [], None
            for i, action in enumerate(reply.actions):
                if self._stop.is_set():
                    break
                result, why = check_result(action, await call(env.act, action), labels)
                results.append(result)
                if await call(env.success):
                    break
                if why is not None:
                    stopped = {"after": i + 1, "of": len(reply.actions), "reason": why}
                    break
            lines = [r.summary() for r in results]
            if stopped:
                lines += [f"{a.describe()}: skipped" for a in reply.actions[stopped["after"]:]]
            stopped_reason = stopped["reason"] if stopped else None
            if reply.notes is not None:
                notes = reply.notes
            for lesson in reply.lessons:
                if lesson not in self.run_lessons:
                    self.run_lessons.append(lesson)
            todo = reply.todo or todo
            history.append(StepSummary(n, reply.plan, tuple(lines), stopped and
                                       f"Stopped after action {stopped['after']} of {stopped['of']}: "
                                       f"{stopped['reason']}."))
            solved = await call(env.success) if env.has_check else False
            emit("step", {
                "step": n, "observation": reply.observation, "plan": reply.plan,
                "actions": [a.describe() for a in reply.actions], "results": lines, "stopped": stopped,
                "truncated": reply.truncated, "notes": notes, "todo": list(todo), "success": solved, "done": reply.done,
                "lessons": list(reply.lessons),
                "latency_s": usage.latency_s if usage else None, "cost_usd": usage.cost_usd if usage else None,
                **detail,
            })
            if solved:
                status = "success"
                break
            if self._stop.is_set():
                status = "stopped"
                break
            if reply.done and not env.has_check:
                status = "done"
                break
            check_failed = reply.done  # the model says it is done, but the check disagrees
        else:
            if self._stop.is_set():
                status = "stopped"
        self._save_lessons(env.scene_name, status)
        emit("run_end", {"status": status})
        return status

    def _save_lessons(self, scene: str, status: str) -> None:
        """Save the new lessons of the run. A failed write gives a warning and does not stop the run."""
        if self.store is None or not self.run_lessons:
            return
        try:
            self.store.append(scene, self.run_lessons, run=self.run_id, outcome=status)
        except OSError as e:
            logging.getLogger(__name__).warning("cannot save the lessons of scene %s: %s", scene, e)

    async def _ask(self, msg: Message, attempts: list[dict]) -> tuple[Reply | None, LLMError | None, Usage | None]:
        reply, error, usage = await self._send(msg, attempts)
        if error is not None and error.kind == "schema" and not self._stop.is_set():
            retry = Message("user", f"Your last reply was not valid: {error.message[:300]}. "
                                    "Reply again in the required format.")
            reply, error, usage = await self._send(retry, attempts)
        return reply, error, usage

    async def _send(self, msg: Message, attempts: list[dict]) -> tuple[Reply | None, LLMError | None, Usage | None]:
        """Send one message and record the attempt (what went out, what came back) in `attempts`."""
        reply, error, usage, final = await self._send_raw(msg)
        shown = None
        if final is not None:
            shown = json.dumps(final.parsed, ensure_ascii=False, indent=2) if final.parsed is not None else final.text
        attempts.append({"message": msg.text, "images": len(msg.images), "reply": shown,
                         "error": error.message if error else None, "usage": _usage_dict(usage)})
        return reply, error, usage

    async def _send_raw(self, msg: Message) -> tuple[Reply | None, LLMError | None, Usage | None, Final | None]:
        if self._stop.is_set():
            return None, LLMError("cancelled", "stopped"), None, None
        final, usage, error = None, None, None

        async def consume() -> None:
            nonlocal final, usage, error
            async for ev in self._conv.send(msg, REPLY_SCHEMA):
                if isinstance(ev, Final):
                    final = ev
                elif isinstance(ev, Usage):
                    usage = ev
                elif isinstance(ev, LLMError):
                    error = ev

        stream = asyncio.create_task(consume())
        stopper = asyncio.create_task(self._stop.wait())
        try:
            await asyncio.wait({stream, stopper}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stopper.cancel()
            if not stream.done():
                stream.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stream
        if not stream.done() or stream.cancelled():
            return None, LLMError("cancelled", "stopped"), usage, final
        stream.result()
        if error is not None:
            return None, error, usage, final
        if final is None or final.parsed is None:
            return None, LLMError("schema", "no structured answer"), usage, final
        try:
            return parse_reply(final.parsed, self.cfg.max_actions), None, usage, final
        except ReplyError as e:
            return None, LLMError("schema", str(e)), usage, final
