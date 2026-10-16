"""The agent loop: observe, ask the model, act. The physics stands still while the model thinks."""
from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from armmind.agent.parse import Reply, ReplyError, parse_reply
from armmind.agent.prompts import REPLY_SCHEMA, SYSTEM, StepSummary, build_user_text
from armmind.env import Env
from armmind.llm.protocol import Conversation, Final, LLMBackend, LLMError, Message, Usage

Emit = Callable[[str, dict], None]
Call = Callable[..., Awaitable]


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 30
    history_steps: int = 5
    max_actions: int = 8


class Agent:
    def __init__(self, backend: LLMBackend, config: AgentConfig | None = None):
        self.backend, self.cfg = backend, config or AgentConfig()
        self._stop = asyncio.Event()
        self._conv: Conversation | None = None

    async def stop(self) -> None:
        self._stop.set()
        if self._conv is not None:
            await self._conv.cancel()

    async def run(self, env: Env, call: Call, emit: Emit) -> str:
        history: deque[StepSummary] = deque(maxlen=self.cfg.history_steps)
        notes, todo, status, check_failed = "", (), "step_limit", False
        emit("run_start", {"instruction": env.instruction, "max_steps": self.cfg.max_steps,
                           "check": env.check_name})
        for n in range(1, self.cfg.max_steps + 1):
            if self._stop.is_set():
                status = "stopped"
                break
            obs = await call(env.observe)
            emit("observation", {"step": n, "front_png": obs.front_png, "top_png": obs.top_png})
            text = build_user_text(env.instruction, n, self.cfg.max_steps, obs, notes, todo, list(history), check_failed)
            self._conv = await self.backend.open(SYSTEM)
            try:
                reply, error, usage = await self._ask(Message("user", text, (obs.front_png, obs.top_png)))
            finally:
                conv, self._conv = self._conv, None
                await conv.close()
            if self._stop.is_set():
                status = "stopped"
                break
            if reply is None and error is not None and error.kind != "schema":
                emit("llm_error", {"step": n, "error_kind": error.kind, "message": error.message})
                status = "llm_error"
                break
            if reply is None:
                emit("bad_reply", {"step": n, "error": error.message if error else "no reply"})
                continue
            emit("reply", {"step": n, "plan": reply.plan, "actions": [a.describe() for a in reply.actions]})
            results = []
            for action in reply.actions:
                if self._stop.is_set():
                    break
                results.append(await call(env.act, action))
                if await call(env.success):
                    break
            if reply.notes is not None:
                notes = reply.notes
            todo = reply.todo or todo
            history.append(StepSummary(n, reply.plan, tuple(r.summary() for r in results)))
            solved = await call(env.success) if env.has_check else False
            emit("step", {
                "step": n, "observation": reply.observation, "plan": reply.plan,
                "actions": [a.describe() for a in reply.actions], "results": [r.summary() for r in results],
                "truncated": reply.truncated, "notes": notes, "todo": list(todo), "success": solved, "done": reply.done,
                "latency_s": usage.latency_s if usage else None, "cost_usd": usage.cost_usd if usage else None,
            })
            if solved:
                status = "success"
                break
            if reply.done and not env.has_check:
                status = "done"
                break
            check_failed = reply.done  # the model says it is done, but the check disagrees
        else:
            if self._stop.is_set():
                status = "stopped"
        emit("run_end", {"status": status})
        return status

    async def _ask(self, msg: Message) -> tuple[Reply | None, LLMError | None, Usage | None]:
        reply, error, usage = await self._send(msg)
        if error is not None and error.kind == "schema" and not self._stop.is_set():
            retry = Message("user", f"Your last reply was not valid: {error.message[:300]}. "
                                    "Reply again in the required format.")
            reply, error, usage = await self._send(retry)
        return reply, error, usage

    async def _send(self, msg: Message) -> tuple[Reply | None, LLMError | None, Usage | None]:
        if self._stop.is_set():
            return None, LLMError("cancelled", "stopped"), None
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
            return None, LLMError("cancelled", "stopped"), usage
        stream.result()
        if error is not None:
            return None, error, usage
        if final is None or final.parsed is None:
            return None, LLMError("schema", "no structured answer"), usage
        try:
            return parse_reply(final.parsed, self.cfg.max_actions), None, usage
        except ReplyError as e:
            return None, LLMError("schema", str(e)), usage
