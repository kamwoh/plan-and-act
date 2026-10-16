"""HTTP and WebSocket server. One run at a time. Frames and events go to every open page."""
from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from armmind.agent.loop import Agent, AgentConfig
from armmind.env import Env, SimThread
from armmind.llm.protocol import LLMBackend
from armmind.store.runlog import RunLog
from armmind.tasks import TASKS, Task

WEB_DIR = Path(__file__).resolve().parents[3] / "web"
EnvFactory = Callable[[Task, int, bool, Callable[[bytes], None]], Env]


class RunRequest(BaseModel):
    task: str
    seed: int = Field(0, ge=0)
    max_steps: int = Field(30, ge=1, le=200)
    show_positions: bool = False
    instruction: str | None = Field(None, max_length=500)  # replaces the task's text, the success check stays


class Hub:
    def __init__(self):
        self.clients: set = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    async def publish(self, msg: dict) -> None:
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except Exception:  # noqa: BLE001  a closed tab must not stop the run
                self.clients.discard(ws)

    def publish_threadsafe(self, msg: dict) -> None:
        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(self.publish(msg), self.loop)


class Pacer:
    """Sleeps on the sim thread between frames, so that the motion plays at real time."""

    def __init__(self, dt: float = 1 / 30):
        self.dt, self._last = dt, None

    def __call__(self) -> None:
        now = time.monotonic()
        if self._last is not None and (wait := self.dt - (now - self._last)) > 0:
            time.sleep(wait)
            now += wait
        self._last = now


def default_env_factory(task: Task, seed: int, show_positions: bool, on_frame: Callable[[bytes], None]) -> Env:
    from armmind.env import MujocoEnv
    return MujocoEnv(task, seed, show_positions=show_positions, on_frame=on_frame)


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


class Runner:
    def __init__(self, backend: LLMBackend, env_factory: EnvFactory, runs_dir: Path, hub: Hub):
        self.backend, self.env_factory, self.runs_dir, self.hub = backend, env_factory, runs_dir, hub
        self.task: asyncio.Task | None = None
        self.agent: Agent | None = None
        self.run_dir: Path | None = None

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    async def start(self, req: RunRequest, task: Task) -> Path:
        if self.running:
            raise HTTPException(409, "a run is already active")
        log = RunLog(self.runs_dir)
        self.run_dir = log.dir
        pacer = Pacer()

        def on_frame(jpg: bytes) -> None:
            self.hub.publish_threadsafe({"type": "frame", "jpg": _b64(jpg)})
            pacer()

        def emit(kind: str, data: dict) -> None:
            msg = {"type": "event", **log.event(kind, data)}
            for key, value in data.items():
                if isinstance(value, bytes):
                    msg[key] = _b64(value)
            self.hub.publish_threadsafe(msg)

        self.agent = Agent(self.backend, AgentConfig(max_steps=req.max_steps))
        agent = self.agent

        async def go() -> None:
            sim, env = SimThread(), None
            try:
                env = await sim.call(self.env_factory, task, req.seed, req.show_positions, on_frame)
                if req.instruction and req.instruction.strip():
                    env.instruction = req.instruction.strip()
                await agent.run(env, sim.call, emit)
            except Exception as e:  # noqa: BLE001  report and end the run, never kill the server
                emit("run_end", {"status": "error", "message": f"{type(e).__name__}: {e}"})
            finally:
                try:
                    if env is not None:
                        await sim.call(env.close)
                except Exception as e:  # noqa: BLE001  record it, then finish the cleanup
                    log.event("cleanup_error", {"message": f"{type(e).__name__}: {e}"})
                finally:
                    try:
                        sim.shutdown()
                    finally:
                        log.close()

        self.task = asyncio.create_task(go())
        return log.dir

    async def stop(self) -> None:
        if self.agent is not None:
            await self.agent.stop()
        if self.task is not None:
            await asyncio.wait({self.task}, timeout=5)


def create_app(backend: LLMBackend, env_factory: EnvFactory | None = None, runs_dir: Path = Path("runs"),
               web_dir: Path = WEB_DIR) -> FastAPI:
    hub = Hub()
    runner = Runner(backend, env_factory or default_env_factory, Path(runs_dir), hub)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.loop = asyncio.get_running_loop()
        yield
        await runner.stop()

    app = FastAPI(lifespan=lifespan)

    @app.get("/api/tasks")
    def tasks() -> list[dict]:
        return [{"name": t.name, "instruction": t.instruction} for t in TASKS.values()]

    @app.get("/api/state")
    def state() -> dict:
        return {"running": runner.running, "run_dir": str(runner.run_dir) if runner.run_dir else None}

    @app.post("/api/run")
    async def run(req: RunRequest) -> dict:
        task = TASKS.get(req.task)
        if task is None:
            raise HTTPException(404, f"unknown task {req.task!r}")
        return {"run_dir": str(await runner.start(req, task))}

    @app.post("/api/stop")
    async def stop() -> dict:
        await runner.stop()
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(sock: WebSocket) -> None:
        await sock.accept()
        hub.clients.add(sock)
        try:
            while True:
                await sock.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(sock)

    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    app.state.runner, app.state.hub = runner, hub
    return app
