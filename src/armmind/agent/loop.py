"""The agent loop: observe, ask the model, act. The physics stands still while the model thinks."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from armmind.agent.parse import PlanReply, Reply, ReplyError, parse_plan, parse_reply
from armmind.agent.prompts import (
    PLAN_REPLY_SCHEMA,
    PLAN_SYSTEM,
    REPLY_SCHEMA,
    SYSTEM,
    Feedback,
    StepSummary,
    build_plan_text,
    build_user_text,
)
from armmind.agent.rules import check_result
from armmind.env import Env
from armmind.llm.protocol import Conversation, Final, LLMBackend, LLMError, Message, Usage
from armmind.store.lessons import LessonStore

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




MAX_RUN_LESSONS = 30  # the newest lessons of this run that go into each step message


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
        self.run_lessons = []
        self.saved_lessons = tuple(self.store.texts(env.scene_name)) if self.store and self.use_lessons else ()
        start = {"scene": env.scene_name, "instruction": env.instruction, "max_steps": self.cfg.max_steps,
                 "check": env.check_name, "system_prompt": self.system,
                 "use_lessons": self.use_lessons, "saved_lessons": list(self.saved_lessons)}
        if hasattr(env, "cameras"):  # the page draws the planned path on the camera images with these
            try:
                start["cameras"] = await call(env.cameras)
            except Exception as e:  # noqa: BLE001  the run goes on without the overlay
                logging.getLogger(__name__).warning("no camera data for the path overlay: %s: %s",
                                                    type(e).__name__, e)
        emit("run_start", start)
        status = "error"  # kept when a step raises
        try:
            status = await self._steps(env, call, emit)
        finally:
            self._save_lessons(env.scene_name, status)
        emit("run_end", {"status": status})
        return status

    async def _steps(self, env: Env, call: Call, emit: Emit) -> str:
        history: deque[StepSummary] = deque(maxlen=self.cfg.history_steps)
        notes, todo, status, check_failed, stopped, stopped_reason = "", (), "step_limit", False, None, None
        labels = getattr(env, "object_labels", None) or {}
        for n in range(1, self.cfg.max_steps + 1):
            if self._stop.is_set():
                status = "stopped"
                break
            obs = await call(env.observe)
            emit("observation", {"step": n, "front_png": obs.front_png, "top_png": obs.top_png,
                                 "tcp": list(obs.tcp), "yaw": obs.gripper_yaw})
            text = build_user_text(env.scene_text, env.instruction, n, self.cfg.max_steps, obs, notes, todo, list(history),
                                   check_failed, env.image_text,
                                   [*self.saved_lessons, *self.run_lessons[-MAX_RUN_LESSONS:]], stopped_reason,
                                   skipped=bool(stopped and stopped["after"] < stopped["of"]))
            self._conv = await self.backend.open(self.system)
            attempts: list[dict] = []
            try:
                reply, error, usage = await self._ask(Message("user", text, (obs.front_png, obs.top_png)), attempts,
                                                      lambda m, i, n=n: emit("ask", {"step": n, "prompt": m.text,
                                                                                    "attempt": i}))
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
            emit("reply", {"step": n, "observation": reply.observation, "plan": reply.plan,
                           "actions": [a.describe() for a in reply.actions],
                           "action_data": [a.to_dict() for a in reply.actions], "notes": reply.notes,
                           "todo": list(reply.todo), "lessons": list(reply.lessons), "usage": _usage_dict(usage)})
            results, stopped = [], None
            for i, action in enumerate(reply.actions):
                if self._stop.is_set():
                    break
                emit("action", {"step": n, "index": i, "of": len(reply.actions)})  # the page marks the action that runs
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
            if reply.done and not stopped and not env.has_check:
                status = "done"
                break
            check_failed = reply.done  # the model says it is done, but the check disagrees
        else:
            if self._stop.is_set():
                status = "stopped"
        return status

    async def plan(self, env: Env, call: Call, emit: Emit, seed: int = 0, feedback: Feedback | None = None,
                   auto_rounds: int | None = None) -> str:
        """One model call: the belief of the scene and the keyframes of the whole task. The code then scores
        the plan and builds the view of the page. Nothing moves. With feedback, this is a re-plan: a later round
        that sees the last plan, its result and its end images. auto_rounds is set in a round of the Auto loop."""
        self.saved_lessons = tuple(self.store.texts(env.scene_name)) if self.store and self.use_lessons else ()
        emit("run_start", {"mode": "plan", "scene": env.scene_name, "instruction": env.instruction, "seed": seed,
                           "run_id": self.run_id, "max_steps": 1, "check": env.check_name,
                           "system_prompt": PLAN_SYSTEM, "use_lessons": self.use_lessons,
                           "saved_lessons": list(self.saved_lessons),
                           "round": feedback.round if feedback else 1, "parent": feedback.parent if feedback else None,
                           "end_step": feedback.end_step if feedback else None, "auto_rounds": auto_rounds,
                           "round_mode": feedback.mode if feedback else None})
        status = "error"
        try:
            status = await self._plan(env, call, emit, feedback)
        finally:
            emit("run_end", {"status": status})
        return status

    async def _plan(self, env: Env, call: Call, emit: Emit, feedback: Feedback | None) -> str:
        obs = await call(env.observe)
        emit("observation", {"step": 1, "front_png": obs.front_png, "top_png": obs.top_png})
        gripper = await call(env.gripper_state) if hasattr(env, "gripper_state") else None
        text = build_plan_text(env.scene_text, env.instruction, obs, env.image_text, self.saved_lessons, feedback,
                               gripper)
        images = (obs.front_png, obs.top_png, *(feedback.images if feedback else ()))
        self._conv = await self.backend.open(PLAN_SYSTEM)
        attempts: list[dict] = []
        try:
            reply, error, _ = await self._ask(
                Message("user", text, images), attempts,
                lambda m, i: emit("ask", {"step": 1, "prompt": m.text, "attempt": i}), PLAN_REPLY_SCHEMA, parse_plan)
        finally:
            conv, self._conv = self._conv, None
            await conv.close()
        if self._stop.is_set():
            return "stopped"
        detail = {"prompt": text, "attempts": attempts, "usage": _sum_usage(attempts)}
        if reply is None and error is not None and error.kind != "schema":
            emit("llm_error", {"step": 1, "error_kind": error.kind, "message": error.message, **detail})
            return "llm_error"
        if reply is None:
            emit("bad_reply", {"step": 1, "error": error.message if error else "no reply", **detail})
            return "bad_reply"
        self._save_plan_lessons(env.scene_name, reply.lessons, feedback)
        scores = await call(env.check_plan, reply.objects, reply.keyframes)
        view = await call(env.plan_view, reply.objects, reply.keyframes, scores["marks"])
        emit("plan", {"step": 1, "summary": reply.summary, "objects": [o.to_dict() for o in reply.objects],
                      "keyframes": [k.to_dict() for k in reply.keyframes], "warnings": list(reply.warnings),
                      "lessons": list(reply.lessons), "scores": scores, "view": view, **detail})
        return "planned"

    def _save_plan_lessons(self, scene: str, lessons: tuple[str, ...], feedback: Feedback | None) -> None:
        """Save the lessons of a re-plan with the status of the execution that the feedback describes.
        A round 1 plan saw no result, so its lessons are not saved."""
        if not lessons:
            return
        if feedback is None:
            logging.getLogger(__name__).info("a round 1 plan wrote %d lessons: they are not saved", len(lessons))
            return
        if self.store is None or not self.use_lessons:
            return
        try:
            self.store.append(scene, list(lessons), run=self.run_id, outcome=feedback.status)
        except OSError as e:
            logging.getLogger(__name__).warning("cannot save the lessons of scene %s: %s", scene, e)

    def _save_lessons(self, scene: str, status: str) -> None:
        """Save the new lessons of the run. A failed write gives a warning and does not stop the run."""
        if self.store is None or not self.run_lessons:
            return
        try:
            self.store.append(scene, self.run_lessons, run=self.run_id, outcome=status)
        except OSError as e:
            logging.getLogger(__name__).warning("cannot save the lessons of scene %s: %s", scene, e)

    async def _ask(self, msg: Message, attempts: list[dict], on_ask: Callable[[Message, int], None],
                   schema: dict = REPLY_SCHEMA, parse: Callable | None = None
                   ) -> tuple[Reply | PlanReply | None, LLMError | None, Usage | None]:
        """Send the message, and once more after a reply in the wrong format. on_ask runs before each send."""
        parse = parse or (lambda obj: parse_reply(obj, self.cfg.max_actions))
        on_ask(msg, 1)
        reply, error, usage = await self._send(msg, attempts, schema, parse)
        if error is not None and error.kind == "schema" and not self._stop.is_set():
            retry = Message("user", f"Your last reply was not valid: {error.message[:300]}. "
                                    "Reply again in the required format.")
            on_ask(retry, 2)
            reply, error, usage = await self._send(retry, attempts, schema, parse)
        return reply, error, usage

    async def _send(self, msg: Message, attempts: list[dict], schema: dict, parse: Callable
                    ) -> tuple[Reply | PlanReply | None, LLMError | None, Usage | None]:
        """Send one message and record the attempt (what went out, what came back) in `attempts`."""
        reply, error, usage, final = await self._send_raw(msg, schema, parse)
        shown = None
        if final is not None:
            shown = json.dumps(final.parsed, ensure_ascii=False, indent=2) if final.parsed is not None else final.text
        attempts.append({"message": msg.text, "images": len(msg.images), "reply": shown,
                         "error": error.message if error else None, "usage": _usage_dict(usage)})
        return reply, error, usage

    async def _send_raw(self, msg: Message, schema: dict, parse: Callable
                        ) -> tuple[Reply | PlanReply | None, LLMError | None, Usage | None, Final | None]:
        if self._stop.is_set():
            return None, LLMError("cancelled", "stopped"), None, None
        final, usage, error = None, None, None

        async def consume() -> None:
            nonlocal final, usage, error
            async for ev in self._conv.send(msg, schema):
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
            return parse(final.parsed), None, usage, final
        except ReplyError as e:
            return None, LLMError("schema", str(e)), usage, final
