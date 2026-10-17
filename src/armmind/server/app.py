"""HTTP and WebSocket server. One run at a time. Frames and events go to every open page."""
from __future__ import annotations

import asyncio
import base64
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from armmind.agent.loop import Agent, AgentConfig
from armmind.agent.prompts import SYSTEM, example_step_text
from armmind.env import Env, SimThread
from armmind.llm.protocol import LLMBackend
from armmind.scenes import SCENES, Scene, Task
from armmind.sim.assets import AssetsMissing
from armmind.sim.meshes import warm
from armmind.store.lessons import LessonChanged, LessonStore
from armmind.store.runlog import RunLog

WEB_DIR = Path(__file__).resolve().parents[3] / "web"
EnvFactory = Callable[[Scene, Task | None, str, int, bool, Callable[[str, bytes], None]], Env]
Preview = Callable[[Scene, int], dict]


class RunRequest(BaseModel):
    instruction: str = Field(min_length=1, max_length=500)
    scene: str = "cubes"
    seed: int = Field(0, ge=0)
    max_steps: int = Field(30, ge=1, le=200)
    show_positions: bool = False
    system_prompt: str | None = Field(None, max_length=10000)
    use_lessons: bool = True

    @field_validator("instruction")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("the instruction is empty")
        return v

    @field_validator("system_prompt")
    @classmethod
    def _strip_prompt(cls, v: str | None) -> str | None:
        return (v or "").strip() or None  # empty means: use the default


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


def default_env_factory(scene: Scene, task: Task | None, instruction: str, seed: int, show_positions: bool,
                        on_frame: Callable[[str, bytes], None]) -> Env:
    from armmind.env import MujocoEnv
    return MujocoEnv(scene, task, seed, instruction, show_positions=show_positions, on_frame=on_frame)


def default_preview(scene: Scene, seed: int) -> dict:
    from armmind.env import render_preview
    return render_preview(scene, seed)


def _same_origin(headers) -> bool:
    """True when the request has no Origin header, or the Origin host:port equals the Host header."""
    origin = headers.get("origin")
    return origin is None or urlsplit(origin).netloc == headers.get("host")


def _scene(name: str) -> Scene:
    if name not in SCENES:
        raise HTTPException(404, f"unknown scene {name!r}")
    return SCENES[name]


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


class Runner:
    def __init__(self, backend: LLMBackend, env_factory: EnvFactory, runs_dir: Path, hub: Hub, lessons: LessonStore):
        self.backend, self.env_factory, self.runs_dir, self.hub = backend, env_factory, runs_dir, hub
        self.lessons = lessons
        self.task: asyncio.Task | None = None
        self.agent: Agent | None = None
        self.run_dir: Path | None = None

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    async def start(self, req: RunRequest, scene: Scene, task: Task | None) -> Path:
        if self.running:
            raise HTTPException(409, "a run is already active")
        log = RunLog(self.runs_dir)
        self.run_dir = log.dir
        pacer = Pacer()

        def on_frame(cam: str, jpg: bytes) -> None:
            self.hub.publish_threadsafe({"type": "frame", "cam": cam, "jpg": _b64(jpg)})
            if cam == "front":  # the front image comes first in every frame, so this paces once per frame
                pacer()

        def emit(kind: str, data: dict) -> None:
            msg = {"type": "event", **log.event(kind, data)}
            for key, value in data.items():
                if isinstance(value, bytes):
                    msg[key] = _b64(value)
            self.hub.publish_threadsafe(msg)

        self.agent = Agent(self.backend, AgentConfig(max_steps=req.max_steps), system=req.system_prompt,
                           store=self.lessons, use_lessons=req.use_lessons, run_id=log.dir.name)
        agent = self.agent

        async def go() -> None:
            sim, env = SimThread(), None
            try:
                env = await sim.call(self.env_factory, scene, task, req.instruction, req.seed, req.show_positions, on_frame)
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
               web_dir: Path = WEB_DIR, preview: Preview | None = None, lessons_dir: Path | None = None) -> FastAPI:
    hub = Hub()
    lessons = LessonStore(Path(lessons_dir) if lessons_dir is not None else Path(runs_dir).parent / "lessons")
    runner = Runner(backend, env_factory or default_env_factory, Path(runs_dir), hub, lessons)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.loop = asyncio.get_running_loop()
        if env_factory is None:  # the real server: measure the scanned objects early, without a wait
            threading.Thread(target=warm, args=({m for sc in SCENES.values() for m in sc.meshes},),
                             daemon=True, name="warm-meshes").start()
        yield
        await runner.stop()

    app = FastAPI(lifespan=lifespan)

    def task_list(scene: Scene) -> list[dict]:
        return [{"name": t.name, "instruction": t.instruction} for t in scene.tasks.values()]

    @app.get("/api/tasks")
    def tasks() -> list[dict]:
        return task_list(SCENES["cubes"])  # kept for old pages: the cubes scene

    @app.get("/api/scenes")
    def scenes() -> list[dict]:
        return [{"name": s.name, "title": s.title, "description": s.describe(), "tasks": task_list(s),
                 "cameras": s.camera_labels} for s in SCENES.values()]

    @app.get("/api/state")
    def state() -> dict:
        return {"running": runner.running, "run_dir": str(runner.run_dir) if runner.run_dir else None}

    def same_origin(request: Request) -> None:
        if not _same_origin(request.headers):
            raise HTTPException(403, "cross-site request refused")

    @app.post("/api/run", dependencies=[Depends(same_origin)])
    async def run(req: RunRequest) -> dict:
        scene = _scene(req.scene)
        task = next((t for t in scene.tasks.values() if t.instruction == req.instruction), None)
        return {"run_dir": str(await runner.start(req, scene, task))}

    @app.get("/api/prompt")
    def prompt() -> dict:
        return {"system": SYSTEM, "step_example": example_step_text()}

    @app.get("/api/preview")
    async def scene_preview(scene: str = "cubes", seed: int = Query(0, ge=0)) -> dict:
        chosen = _scene(scene)
        if runner.running:
            raise HTTPException(409, "a run is active")
        sim = SimThread()  # a fresh thread, because the EGL context belongs to the thread that made it
        try:
            return await sim.call(preview or default_preview, chosen, seed)
        except AssetsMissing as e:  # the Panda model or a scanned object is not downloaded
            raise HTTPException(503, str(e)) from e
        finally:
            sim.shutdown()

    @app.get("/api/lessons")
    def get_lessons(scene: str) -> list[dict]:
        return lessons.load(_scene(scene).name)

    @app.delete("/api/lessons", dependencies=[Depends(same_origin)])
    def delete_lessons(scene: str, index: int | None = None, text: str | None = None) -> dict:
        name = _scene(scene).name
        if runner.running:
            raise HTTPException(409, "a run is active")
        try:
            lessons.delete(name, index, text)
        except IndexError as e:
            raise HTTPException(404, str(e)) from e
        except LessonChanged as e:  # another tab or a run end changed the list: reload it
            raise HTTPException(409, str(e)) from e
        return {"ok": True}

    @app.post("/api/stop", dependencies=[Depends(same_origin)])
    async def stop() -> dict:
        await runner.stop()
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(sock: WebSocket) -> None:
        if not _same_origin(sock.headers):
            await sock.close(code=1008)
            return
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
