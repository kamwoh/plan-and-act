"""HTTP and WebSocket server. One run at a time. Frames and events go to every open page."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import functools
import logging
import shutil
import threading
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from armmind.agent.loop import BRANCH_CHAT, Agent, AgentConfig, Chain
from armmind.agent.parse import Verdict
from armmind.agent.prompts import SYSTEM, Feedback, Frame, example_step_text
from armmind.env import Env, SimThread, keyframe_actions
from armmind.llm.protocol import LLMBackend
from armmind.plan.formats import Keyframe
from armmind.plan.frames import pick_frames
from armmind.plan.scores import prediction
from armmind.plan.track import moved_after, wrap_deg
from armmind.scenes import SCENES, Scene, Task
from armmind.sim.assets import AssetsMissing
from armmind.sim.meshes import warm
from armmind.sim.state import END_STATE, START_STATE, state_problem
from armmind.store.lessons import LessonChanged, LessonStore
from armmind.store.runlog import RunLog
from armmind.store.runs import RunStore, valid_run_id

WEB_DIR = Path(__file__).resolve().parents[3] / "web"
# (scene, task, instruction, seed, show_positions, on_frame). A round that continues adds start=<state file>.
EnvFactory = Callable[..., Env]
Preview = Callable[[Scene, int], dict]


MAX_ROUND = 30  # a plan and at most 29 re-plans
MAX_RUN_KEYFRAMES = 30  # the largest "Run per round"
RoundMode = Literal["continue", "start"]  # a re-plan goes on from the end state, or starts at the reset


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


class PlanRequest(BaseModel):
    instruction: str = Field(min_length=1, max_length=500)
    scene: str = "cubes"
    seed: int = Field(0, ge=0)
    use_lessons: bool = True
    auto: bool = False  # the Auto loop: plan, execute and re-plan until the auto-check passes
    rounds: int = Field(3, ge=1, le=MAX_ROUND)  # the rounds of the Auto loop
    mode: RoundMode = "continue"  # the mode of each round of the Auto loop after round 1
    max_keyframes: int = Field(0, ge=0, le=MAX_RUN_KEYFRAMES)  # "Run per round" of the Auto loop. 0: all

    @field_validator("instruction")
    @classmethod
    def _strip(cls, v: str) -> str:
        return RunRequest._strip(v)


class ReplanRequest(BaseModel):
    mode: RoundMode = "continue"
    max_keyframes: int = Field(0, ge=0, le=MAX_RUN_KEYFRAMES)  # the page setting: a mode "start" needs 0
    instruction: str | None = Field(None, max_length=500)  # a new instruction: "Plan from here"
    auto: bool = False  # the Auto loop from this round on
    rounds: int = Field(3, ge=1, le=MAX_ROUND)  # the rounds of the Auto loop

    @field_validator("instruction")
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        return (v or "").strip() or None  # empty means: keep the instruction of the round


class ExecuteRequest(BaseModel):
    max_keyframes: int = Field(0, ge=0, le=MAX_RUN_KEYFRAMES)  # run only the first K keyframes. 0: all


START_WITH_K = "Run per round needs a re-plan that continues. Turn Continue on, or set Run per round to 0"
NEW_FROM_START = "A new instruction always continues from where the arm is now. Use Plan for a fresh start"


def run_keyframes(keyframes: tuple, k: int) -> tuple:
    """The keyframes that an execution runs: the first k, or all when k is 0 or not below the count."""
    return keyframes[:k] if 0 < k < len(keyframes) else keyframes


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
                        on_frame: Callable[[str, bytes], None], start: Path | None = None) -> Env:
    from armmind.env import MujocoEnv
    return MujocoEnv(scene, task, seed, instruction, show_positions=show_positions, on_frame=on_frame, start=start)


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


def _prediction(view: dict | None, out: dict) -> dict | None:
    """The prediction score of an execution, or None when the view or the true poses are not there."""
    if not isinstance(view, dict) or "true_after" not in out or not {"track", "believed"} <= set(view):
        return None  # no view, or a view older than the plan player
    try:
        return prediction(view["track"], view["believed"], out["true_start"], out["true_after"],
                          view.get("true_match") or {})
    except Exception:  # a broken view: the execution result stays, and the log has the cause
        logging.getLogger(__name__).exception("cannot score the prediction of the plan")
        return None


def _save_frames(folder: Path, step: int, frames: list[dict], end_images: bool, of: int | None = None) -> list[dict]:
    """Write the frames of an execution as image files of the run folder. Gives the frames for exec_end, in
    order, with the file name in place of the image. "sent" marks the frames that go to Claude.
    `of` is the keyframe count of the whole plan, when the execution ran only its first keyframes."""
    sent = set(pick_frames(frames, end_images))  # before `of`: the end images show the last keyframe that ran
    out = []
    for i, f in enumerate(frames):
        name = f"step{step:03d}_kf{int(f['keyframe']):02d}{'_stop' if f['at'] == 'stop' else ''}.png"
        (folder / name).write_bytes(f["png"])
        out.append({"keyframe": f["keyframe"], "of": of or f["of"], "note": f["note"], "at": f["at"], "file": name,
                    "sent": i in sent})
    return out


def plan_round(start: dict) -> int:
    """The round of a plan run. Plans from before v1.12 have no round: they are round 1."""
    r = start.get("round")
    return r if isinstance(r, int) and not isinstance(r, bool) and r >= 1 else 1


def _reply_format(plan: dict) -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    """The belief and the keyframes of a saved plan event, in the format of the plan reply."""
    objects = tuple({k: o.get(k) for k in ("name", "shape", "size", "xyz", "yaw")} for o in plan.get("objects") or [])
    keyframes = []
    for k in plan["keyframes"]:
        d = {"xyz": k.get("xyz"), "yaw": k.get("yaw"), "gripper": k.get("state"),
             "holding": k.get("carrying") or "nothing", "note": k.get("note", "")}
        if k.get("moves"):
            d["moves"] = [{"object": m.get("name"), "xyz": m.get("xyz"), "yaw": m.get("yaw")} for m in k["moves"]]
        keyframes.append(d)
    return objects, tuple(keyframes)


def _imagined(view: dict | None) -> list[tuple] | None:
    """For each keyframe of a plan view: (name, xyz, yaw) of each entity that Claude's plan carries or moves up
    to its end, from Claude's own belief and keyframes. The yaw is None for a round entity. None: no track."""
    try:
        believed = list(view["believed"])
        shapes = {e["name"]: e.get("shape") for e in believed}
        return [tuple((n, tuple(round(float(v), 3) for v in p["xyz"]),
                       None if shapes.get(n) in ("cylinder", "sphere") else wrap_deg(float(p["yaw"])))
                      for n, p in poses.items()) for poses in moved_after(view["track"], believed)]
    except (KeyError, TypeError, ValueError, AttributeError):  # a view without a track, or an old one
        return None


def _sent_frames(found: dict, history: RunStore) -> tuple[Frame, ...]:
    """The frames of the last execution of a saved plan that went to Claude, in order, with the poses that
    Claude imagined at each. A frame without its image file is left out."""
    end = (found["execution"] or {}).get("end") or {}
    listed = [f for f in end.get("frames") or [] if isinstance(f, dict) and f.get("sent")]
    imagined = _imagined(found["plan"].get("view")) if listed else None
    out = []
    for f in listed:
        path = history.image(found["folder"].name, f.get("file") if isinstance(f.get("file"), str) else "")
        try:
            k = int(f["keyframe"])
            png = path.read_bytes() if path is not None else None
        except (KeyError, TypeError, ValueError, OSError):
            continue
        if png is not None:
            poses = imagined[k - 1] if imagined is not None and 1 <= k <= len(imagined) else None
            out.append(Frame(k, int(f.get("of") or k), str(f.get("note") or ""), "stop" if f.get("at") == "stop"
                             else "end", poses, png))
    return tuple(out)


def _ran_keyframes(end: dict) -> tuple[int, int]:
    """(K, N) when an execution ran only its first K of N keyframes by design, else (0, 0)."""
    if end.get("status") != "partial":
        return (0, 0)
    return int(end.get("ran_keyframes") or 0), int(end.get("keyframes") or 0)


def _feedback(found: dict, history: RunStore, mode: str = "start") -> Feedback | None:
    """The feedback of the last execution of a saved plan, or None when it has not run to its end. It reads
    only the model's own plan, the status, the stop and the end images: never the scores or the true poses.
    In the mode "continue", the new round starts at the end, so its own images are the end images."""
    ex = found["execution"]
    if ex is None or ex["end"] is None:
        return None
    end, actions = ex["end"], (ex["step"] or {}).get("actions") or []
    run_id = found["folder"].name
    stopped = end.get("stopped") if isinstance(end.get("stopped"), dict) else None
    stopped_action = None
    if stopped:
        try:
            stopped_action = str(actions[int(stopped["after"]) - 1])
        except (KeyError, IndexError, TypeError, ValueError):
            stopped_action = None
    images: tuple[bytes, ...] = ()
    paths = [history.image(run_id, end.get(k) if isinstance(end.get(k), str) else "") for k in ("front_png", "top_png")]
    if all(paths) and mode != "continue":
        try:
            images = tuple(path.read_bytes() for path in paths)
        except OSError:
            images = ()
    objects, keyframes = _reply_format(found["plan"])
    total = len(actions) or len(ex["start"].get("action_data") or [])
    return Feedback(objects, keyframes, str(end.get("status") or "error"), bool(ex["start"].get("check")),
                    bool(end.get("success")), stopped, stopped_action, images, run_id,
                    plan_round(found["start"]) + 1, end.get("step") if images else None,
                    ran=ex["actions"], total=total, mode=mode, frames=_sent_frames(found, history),
                    ran_keyframes=_ran_keyframes(end))


def _no_run_feedback(found: dict, status: str = "not run") -> Feedback:
    """The feedback of a saved plan without an end state: it did not run, or its execution ended with "error".
    It has the plan only, and the status. The new round starts where that plan started."""
    objects, keyframes = _reply_format(found["plan"])
    mode = "continue" if found["start"].get("round_mode") == "continue" else "start"
    return Feedback(objects, keyframes, status, False, False, None, None, (), found["folder"].name,
                    plan_round(found["start"]) + 1, None, mode=mode, executed=False)


def chain_of(found: dict) -> Chain:
    """The chat of the chain of a saved plan. A plan from before v1.14 has no chain: it starts one."""
    chain = found["start"].get("chain")
    if not (isinstance(chain, str) and valid_run_id(chain)):
        chain = found["folder"].name
    return Chain(chain, found.get("session"), found.get("chat_images", 0), found.get("chat_start", False))


def branch_chain(found: dict, history: RunStore) -> Chain:
    """The chat for a re-plan or a verdict of a saved plan. When the plan already has a child round, its saved
    chat also holds that round, even a stopped one. A new chat with a summary then starts a branch."""
    chain, run_id = chain_of(found), found["folder"].name
    if any(r["parent"] == run_id for r in history.runs()):
        return Chain(chain.id, note=BRANCH_CHAT)
    return chain


def start_state(found: dict) -> Path | None:
    """The state file that a saved plan starts from: its START_STATE in a round that continues, else None."""
    return found["folder"] / START_STATE if found["start"].get("round_mode") == "continue" else None


def saved_plan(found: dict) -> tuple[Scene, Task | None, str, int, tuple[Keyframe, ...]] | None:
    """The scene, the task, the instruction, the seed and the keyframes of a saved plan, or None."""
    start, plan = found["start"], found["plan"]
    scene = SCENES.get(start.get("scene") or "")
    try:
        keyframes = tuple(Keyframe.from_dict(k) for k in plan["keyframes"])
        seed = int(start.get("seed", 0))
    except (KeyError, TypeError, ValueError):
        return None
    if scene is None or not keyframes:
        return None
    instruction = start.get("instruction") or ""
    task = next((t for t in scene.tasks.values() if t.instruction == instruction), None)
    return scene, task, instruction, seed, keyframes


def _copy_state(state: Path, folder: Path) -> Path:
    """Copy the end state of the parent into the run folder of a round that continues. Gives the copy."""
    target = folder / START_STATE
    shutil.copyfile(state, target)
    return target


AUTO_VIEW_S = 20.0  # the Auto loop waits for the imagined video of a plan, at most this long, before Execute


def _start_gripper(view: dict | None) -> str:
    """The gripper at the start of a plan, from its view: "closed" in a round that continues closed."""
    try:
        return "closed" if view["track"]["start"]["state"] == "closed" else "open"
    except (KeyError, TypeError):
        return "open"


def _view_seconds(view: dict | None, limit: float) -> float:
    try:
        return max(0.0, min(limit, float(view["track"]["duration"])))
    except (KeyError, TypeError, ValueError):
        return 0.0


class Runner:
    def __init__(self, backend: LLMBackend, env_factory: EnvFactory, runs_dir: Path, hub: Hub, lessons: LessonStore,
                 history: RunStore):
        self.backend, self.env_factory, self.runs_dir, self.hub = backend, env_factory, runs_dir, hub
        self.lessons, self.history = lessons, history
        self.task: asyncio.Task | None = None
        self._stopper: Callable[[], Awaitable[None]] | None = None  # stops the active run, plan, execution or loop
        self.run_dir: Path | None = None
        self.auto_view_s = AUTO_VIEW_S
        self.auto = False  # True while the Auto loop runs

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    async def start(self, req: RunRequest, scene: Scene, task: Task | None) -> Path:
        if self.running:
            raise HTTPException(409, "a run is already active")
        log = RunLog(self.runs_dir)
        agent = Agent(self.backend, AgentConfig(max_steps=req.max_steps), system=req.system_prompt,
                      store=self.lessons, use_lessons=req.use_lessons, run_id=log.dir.name)
        self._launch(log, agent.stop, self._session(log, scene, task, req.instruction, req.seed, req.show_positions,
                                                    agent.run))
        return log.dir

    def _plan_part(self, log: RunLog, req: PlanRequest, feedback: Feedback | None, auto_rounds: int | None,
                   chain: Chain | None = None):
        """The agent and the session body of one plan round. A re-plan continues the chat of `chain`."""
        agent = Agent(self.backend, store=self.lessons, use_lessons=req.use_lessons, run_id=log.dir.name,
                      chain=chain)
        return agent, lambda env, call, emit: agent.plan(env, call, emit, seed=req.seed, feedback=feedback,
                                                         auto_rounds=auto_rounds)

    async def start_plan(self, req: PlanRequest, scene: Scene, task: Task | None,
                         feedback: Feedback | None = None, state: Path | None = None,
                         chain: Chain | None = None) -> Path:
        """A run of kind "plan": one model call, no motion. The env never gets the true positions.
        With feedback, it is a re-plan of the run feedback.parent. With req.auto, it starts the Auto loop: after
        a parent, its first round is that re-plan.
        With `state`, the end state of the parent, the round continues: the state file goes into the new run
        folder, and the env of the plan and of each Execute starts from it."""
        if self.running:
            raise HTTPException(409, "a run is already active")
        log = RunLog(self.runs_dir)
        if req.auto:
            self._start_loop(log, req, scene, task, feedback, state, chain)
            return log.dir
        start = _copy_state(state, log.dir) if state is not None else None
        agent, body = self._plan_part(log, req, feedback, None, chain)
        self._launch(log, agent.stop, self._session(log, scene, task, req.instruction, req.seed, False, body,
                                                    start=start))
        return log.dir

    def _execute_part(self, folder: Path, scene: Scene, task: Task | None, instruction: str, seed: int,
                      keyframes: tuple[Keyframe, ...], step: int, view: dict | None, round_no: int = 1,
                      round_mode: str | None = None, verdict_next: bool = False, max_keyframes: int = 0):
        """The stopper, the before hook and the session body of one execution of a saved plan. The body saves
        the end state in END_STATE of the folder, also after a stop. With max_keyframes K, it runs only the
        first K keyframes. A run of them without a stop and without a solved check has the status "partial"."""
        cancel = threading.Event()
        run = run_keyframes(keyframes, max_keyframes)
        partial = len(run) < len(keyframes)

        async def stopper() -> None:
            cancel.set()

        def before(emit) -> None:  # before the env, so that a failed start still has a pair of events
            # A file of an earlier execution must not stand for this one, also when the env fails to start.
            (folder / END_STATE).unlink(missing_ok=True)
            actions = keyframe_actions(run, _start_gripper(view))
            emit("exec_start", {"run_id": folder.name, "scene": scene.name, "instruction": instruction, "seed": seed,
                                "check": task.name if task else None, "keyframes": len(run),
                                "plan_keyframes": len(keyframes), "step": step,
                                "round": round_no, "round_mode": round_mode,
                                "action_data": [a.to_dict() for a in actions]})

        async def body(env, call, emit) -> None:
            loop = asyncio.get_running_loop()

            def on_action(index: int, count: int) -> None:  # on the sim thread: emit on the event loop
                loop.call_soon_threadsafe(emit, "action", {"step": step, "index": index, "of": count})

            end_state = folder / END_STATE
            out = await call(env.execute_plan, run, cancel.is_set, on_action)
            if partial and out["status"] in ("done", "failed"):  # the first K keyframes ran by design
                out["status"] = "partial"
            if hasattr(env, "save_state"):
                try:
                    await call(env.save_state, end_state)
                except Exception as e:  # noqa: BLE001  the result stays, a re-plan then cannot continue
                    logging.getLogger(__name__).warning("cannot save the end state: %s: %s", type(e).__name__, e)
            emit("step", {"step": step, "plan": "Run the saved keyframes once.", "observation": "",
                          "actions": out["actions"], "results": out["results"], "stopped": out["stopped"],
                          "truncated": False, "notes": "", "todo": [], "lessons": [], "success": out["success"],
                          "done": False, "prompt": "", "attempts": [], "usage": None})
            images = {}  # the end images, for a re-plan
            try:
                obs = await call(env.observe)
                images = {"front_png": obs.front_png, "top_png": obs.top_png}
            except Exception as e:  # noqa: BLE001  the result stays, a re-plan then has no end images
                logging.getLogger(__name__).warning("no end images of the execution: %s: %s", type(e).__name__, e)
            # After a partial run, the next re-plan judges it: no verdict call.
            judged = verdict_next and bool(images) and out["status"] not in ("cancelled", "error", "partial")
            try:
                frames = _save_frames(folder, step, out.get("frames") or [], bool(images), len(keyframes))
            except (OSError, KeyError, TypeError, ValueError) as e:  # the result stays, without the frames
                logging.getLogger(__name__).warning("cannot save the frames: %s: %s", type(e).__name__, e)
                frames = []
            emit("exec_end", {"status": out["status"], "success": out["success"], "stopped": out["stopped"],
                              "prediction": _prediction(view, out), "step": step, "verdict_next": judged,
                              "frames": frames, "ran_keyframes": len(run), "keyframes": len(keyframes), **images})

        return stopper, before, body

    async def start_execute(self, folder: Path, scene: Scene, task: Task | None, instruction: str, seed: int,
                            keyframes: tuple[Keyframe, ...], step: int, view: dict | None = None,
                            round_no: int = 1, start: Path | None = None, round_mode: str | None = None,
                            max_keyframes: int = 0) -> None:
        """Run the keyframes of a saved plan once, from a fresh reset of its scene and seed, or from the state
        file `start`. The events go to the end of the plan's own log. With the plan view, exec_end has the
        prediction score. Then one short verdict call asks Claude if the end images show the task complete."""
        if self.running:
            raise HTTPException(409, "a run is already active")
        log = RunLog.append_to(folder)
        exec_stop, before, body = self._execute_part(folder, scene, task, instruction, seed, keyframes, step, view,
                                                     round_no, round_mode, verdict_next=True,
                                                     max_keyframes=max_keyframes)
        part: list[Callable[[], Awaitable[None]]] = [exec_stop]
        stopped = threading.Event()

        async def stopper() -> None:
            stopped.set()
            await part[0]()

        async def job() -> None:
            await self._session(log, scene, task, instruction, seed, False, body, end_kind="exec_end",
                                before=before, start=start)
            found = self.history.plan(folder.name)
            end = found["execution"]["end"] if found and found["execution"] else None
            if end and end.get("verdict_next"):
                await self._judge(folder, scene, instruction, part, stopped.is_set)

        self._launch(log, stopper, job())

    async def _judge(self, folder: Path, scene: Scene, instruction: str, part: list[Callable[[], Awaitable[None]]],
                     should_stop: Callable[[], bool]) -> Verdict | None:
        """The verdict call after the last execution of a plan, in the chat of its chain. It sends the end
        images: no true values. A new chat also gets the task and the plan. It always writes a verdict event to
        the run log, with "error" when no verdict came: a stop, an execution without an end, a model error or
        any other error. Gives the verdict, or None. part[0] becomes its stopper."""
        verdict, detail, fields = None, {}, {"check": None, "round": None, "step": None}
        log = RunLog.append_to(folder)
        emit = self._emitter(log)
        try:
            found = self.history.plan(folder.name)
            ex = found["execution"] if found else None
            end = ex["end"] if ex else None
            if not end:
                raise LookupError("the plan has no finished execution")
            fields = {"check": bool(end.get("success")) if ex["start"].get("check") else None,
                      "round": plan_round(found["start"]), "step": end.get("step")}
            if end.get("status") in ("cancelled", "error"):
                raise LookupError(f"the execution ended with {end.get('status')}")
            paths = [self.history.image(folder.name, end.get(k) if isinstance(end.get(k), str) else "")
                     for k in ("front_png", "top_png")]
            if not all(paths):
                raise LookupError("the execution has no end images")
            images = tuple(p.read_bytes() for p in paths)
            objects, keyframes = _reply_format(found["plan"])
            agent = Agent(self.backend, run_id=folder.name, chain=branch_chain(found, self.history))
            part[:] = [agent.stop]
            if should_stop():
                detail = {"error": "stopped"}
            else:
                emit("verdict_start", {"session_id": None})  # a server that dies in the call leaves this mark
                verdict, detail = await agent.verdict(scene.describe(), instruction, objects, keyframes, images,
                                                      scene.image_text, emit, _sent_frames(found, self.history),
                                                      _ran_keyframes(end))
        except Exception as e:  # noqa: BLE001  the execution result stays, the verdict says why it is missing
            detail = {"error": f"{type(e).__name__}: {e}"}
        try:
            emit("verdict", {"done": verdict.done if verdict else None,
                             "reason": verdict.reason if verdict else "", **fields, **detail})
        finally:
            log.close()
        return verdict

    def _start_loop(self, log: RunLog, req: PlanRequest, scene: Scene, task: Task | None,
                    first: Feedback | None = None, first_state: Path | None = None,
                    first_chain: Chain | None = None) -> None:
        """The Auto loop as one job: plan, execute, then re-plan, until the task is solved or the rounds end.
        With an auto-check, the check is the judge. Without one, Claude is. Claude judges each execution: the
        "done" of the next re-plan, or a verdict call after the last execution. A done re-plan also ends a loop
        with an auto-check, because Claude has no more to try. loop_end has Claude's verdict and the check of
        each execution, and the count of agreements.
        Each round has its own run folder. The last round gets a loop_end event. Stop ends the loop.
        With `first`, the loop starts with a re-plan of that parent: from `first_state`, in `first_chain`.
        The loop does not judge the execution of the parent. req.rounds counts the rounds of this loop."""
        stop = asyncio.Event()
        base = first.round - 1 if first is not None else 0  # the round before the first round of the loop
        part: list[Callable[[], Awaitable[None]]] = []  # the stopper of the active plan or execution

        async def stopper() -> None:
            stop.set()
            if part:
                await part[0]()

        async def loop() -> None:
            feedback, rounds_run, success, reason, folder = first, 0, False, "", log.dir
            start = None  # the state file of a round that continues
            chain = first_chain  # the chat of the loop: round 1 starts it, each later round continues it
            verdicts: list[dict] = []  # for each judged execution: {"round", "claude", "reason", "check"}
            pending: dict | None = None  # the last execution, until the next re-plan judges it
            impossible: dict | None = None  # {"round", "reason"} when Claude says that the task is not possible

            async def judge_last(judged: Path, round_no: int, check: bool | None, why: str) -> tuple[bool, str]:
                """The verdict call on the last execution: gives the success and the reason of the loop end.
                Without an auto-check, Claude's verdict is the success."""
                verdict = await self._judge(judged, scene, req.instruction, part, stop.is_set)
                if stop.is_set():
                    return success, "stopped"
                if verdict is not None:
                    verdicts.append({"round": round_no, "claude": verdict.done, "reason": verdict.reason,
                                     "check": check})
                if task is not None:
                    return success, why
                if verdict is not None and verdict.done:
                    return True, "Claude says the task is done"
                return False, f"{why}, and {'Claude says not done' if verdict else 'the verdict call failed'}"

            try:
                while True:
                    rounds_run += 1
                    plan_log = log if rounds_run == 1 else RunLog(self.runs_dir)
                    continues = feedback is not None and feedback.mode == "continue"
                    if rounds_run == 1:
                        start = _copy_state(first_state, plan_log.dir) if first_state is not None else None
                    else:
                        start = _copy_state(folder / END_STATE, plan_log.dir) if continues else None
                    folder = self.run_dir = plan_log.dir
                    agent, body = self._plan_part(plan_log, req, feedback, req.rounds, chain)
                    part[:] = [agent.stop]
                    status = await self._session(plan_log, scene, task, req.instruction, req.seed, False, body,
                                                 start=start)
                    if stop.is_set():
                        reason = "stopped"
                        break
                    found = self.history.plan(folder.name)
                    if found:
                        chain = chain_of(found)
                    if pending is not None and status == "planned" and found:
                        verdicts.append({k: pending[k] for k in ("round", "check")} | {
                            "claude": bool(found["plan"].get("done")), "reason": found["plan"].get("summary") or ""})
                    elif pending is not None:  # the re-plan failed: a verdict call judges the execution before it
                        success, reason = await judge_last(pending["folder"], pending["round"], pending["check"],
                                                           f"the plan ended with {status or 'error'}")
                        break
                    pending = None
                    if status == "planned" and found and found["plan"].get("done"):
                        success = task is None  # with a check, the last execution was not solved
                        reason = ("Claude says the task is done" if success
                                  else "Claude says done, the auto-check says not solved")
                        break
                    if status == "planned" and found and (why := found["plan"].get("impossible")):
                        impossible = {"round": base + rounds_run, "reason": why}
                        reason = f"Claude says the task is not possible: {why}"
                        break
                    saved = saved_plan(found) if found else None
                    if status != "planned" or saved is None:
                        reason = f"the plan ended with {status or 'error'}"
                        break
                    with contextlib.suppress(TimeoutError):  # the page shows the imagined video meanwhile
                        await asyncio.wait_for(stop.wait(), _view_seconds(found["plan"].get("view"), self.auto_view_s))
                    if stop.is_set():
                        reason = "stopped"
                        break
                    exec_stop, before, body = self._execute_part(folder, scene, task, req.instruction, req.seed,
                                                                 saved[4], 2, found["plan"].get("view"),
                                                                 base + rounds_run,
                                                                 feedback.mode if feedback else None,
                                                                 max_keyframes=req.max_keyframes)
                    part[:] = [exec_stop]
                    exec_log = RunLog.append_to(folder)
                    await self._session(exec_log, scene, task, req.instruction, req.seed, False, body,
                                        end_kind="exec_end", before=before, start=start)
                    found = self.history.plan(folder.name)
                    result = _feedback(found, self.history, req.mode) if found else None
                    if stop.is_set():
                        reason = "stopped"
                        break
                    if result is None or (result.status == "error" and rounds_run >= req.rounds):
                        reason = "the execution failed"
                        break
                    if result.status == "error":  # no end state: the next round starts from the start
                        feedback, pending = _feedback(found, self.history, "start"), None
                        continue
                    check = result.success if task else None
                    if result.success:
                        success, reason = True, "the auto-check passed"
                    elif rounds_run >= req.rounds:
                        reason = f"the {req.rounds} rounds ended" + (" without success" if task else "")
                    elif req.mode == "continue" and (problem := state_problem(folder / END_STATE, scene, req.seed)):
                        reason = f"cannot continue: {problem}"
                    else:  # the next re-plan judges this execution
                        feedback, pending = result, {"round": base + rounds_run, "check": check, "folder": folder}
                        continue
                    success, reason = await judge_last(folder, base + rounds_run, check, reason)  # the last one
                    break
            except Exception as e:  # noqa: BLE001  end the loop with a reason, never kill the server
                reason = f"error: {type(e).__name__}: {e}"
            finally:
                end_log = RunLog.append_to(folder)
                try:
                    compared = [v for v in verdicts if v["check"] is not None]
                    self._emitter(end_log)("loop_end", {
                        "rounds_run": rounds_run, "rounds_max": req.rounds, "success": success, "reason": reason,
                        "judge": "check" if task else "claude", "verdicts": verdicts,
                        "claude": verdicts[-1] if verdicts else None,
                        "agree": sum(v["claude"] == v["check"] for v in compared), "compared": len(compared),
                        "impossible": impossible})
                finally:
                    end_log.close()
                    self.auto = False

        self.auto = True  # before the launch, so /api/state never shows the loop as running without auto
        self._launch(log, stopper, loop())

    def _emitter(self, log: RunLog) -> Callable[[str, dict], None]:
        """Write an event to the log and send it to every page. Bytes go to the page as base64."""
        def emit(kind: str, data: dict) -> None:
            msg = {"type": "event", **log.event(kind, data)}
            for key, value in data.items():
                if isinstance(value, bytes):
                    msg[key] = _b64(value)
            self.hub.publish_threadsafe(msg)
        return emit

    def _launch(self, log: RunLog, stopper: Callable[[], Awaitable[None]], job: Awaitable) -> None:
        """Start the job as the one active task. Stop calls the stopper."""
        self.run_dir, self._stopper = log.dir, stopper
        self.task = asyncio.create_task(job)

    async def _session(self, log: RunLog, scene: Scene, task: Task | None, instruction: str, seed: int,
                       show_positions: bool, body: Callable[[Env, Callable, Callable[[str, dict], None]], Awaitable],
                       end_kind: str = "run_end", before: Callable[[Callable[[str, dict], None]], None] | None = None,
                       start: Path | None = None):
        """Make the env on a fresh sim thread, run `body`, and clean up. Gives what `body` gives. An error ends
        with an `end_kind` event and gives None. The log is closed at the end. With `start`, a state file, the
        env starts from that state."""
        pacer = Pacer()

        def on_frame(cam: str, jpg: bytes) -> None:
            self.hub.publish_threadsafe({"type": "frame", "cam": cam, "jpg": _b64(jpg)})
            if cam == "front":  # the front image comes first in every frame, so this paces once per frame
                pacer()

        emit = self._emitter(log)
        sim, env = SimThread(), None
        try:
            if before is not None:
                before(emit)
            make = self.env_factory if start is None else functools.partial(self.env_factory, start=start)
            env = await sim.call(make, scene, task, instruction, seed, show_positions, on_frame)
            return await body(env, sim.call, emit)
        except Exception as e:  # noqa: BLE001  report and end the run, never kill the server
            emit(end_kind, {"status": "error", "message": f"{type(e).__name__}: {e}"})
            return None
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

    async def stop(self) -> None:
        if self._stopper is not None:
            await self._stopper()
        if self.task is not None:
            await asyncio.wait({self.task}, timeout=5)


def create_app(backend: LLMBackend, env_factory: EnvFactory | None = None, runs_dir: Path = Path("runs"),
               web_dir: Path = WEB_DIR, preview: Preview | None = None, lessons_dir: Path | None = None) -> FastAPI:
    hub = Hub()
    lessons = LessonStore(Path(lessons_dir) if lessons_dir is not None else Path(runs_dir).parent / "lessons")
    history = RunStore(Path(runs_dir))
    runner = Runner(backend, env_factory or default_env_factory, Path(runs_dir), hub, lessons, history)

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
        return {"running": runner.running, "run_dir": str(runner.run_dir) if runner.run_dir else None,
                "auto": runner.running and runner.auto}

    def same_origin(request: Request) -> None:
        if not _same_origin(request.headers):
            raise HTTPException(403, "cross-site request refused")

    @app.post("/api/run", dependencies=[Depends(same_origin)])
    async def run(req: RunRequest) -> dict:
        scene = _scene(req.scene)
        task = next((t for t in scene.tasks.values() if t.instruction == req.instruction), None)
        return {"run_dir": str(await runner.start(req, scene, task))}

    @app.post("/api/plan", dependencies=[Depends(same_origin)])
    async def plan(req: PlanRequest) -> dict:
        if req.auto and req.max_keyframes > 0 and req.mode == "start":
            raise HTTPException(409, START_WITH_K)
        scene = _scene(req.scene)
        task = next((t for t in scene.tasks.values() if t.instruction == req.instruction), None)
        folder = await runner.start_plan(req, scene, task)
        return {"run_dir": str(folder), "run_id": folder.name}

    @app.post("/api/plan/{run_id}/execute", dependencies=[Depends(same_origin)])
    async def execute(run_id: str, body: ExecuteRequest | None = None) -> dict:
        found = history.plan(run_id)
        if found is None:
            raise HTTPException(404, "no such plan")
        if why := found["plan"].get("impossible"):
            raise HTTPException(409, f"Claude says the task is not possible: {why}. Nothing executes")
        saved = saved_plan(found)
        if saved is None and found["plan"].get("done") and not found["plan"]["keyframes"]:
            raise HTTPException(404, "Claude says the task is done: the plan has no keyframes to execute")
        if saved is None:
            raise HTTPException(404, "the plan cannot run")
        scene, task, instruction, seed, keyframes = saved
        start = start_state(found)
        if start is not None and (problem := state_problem(start, scene, seed)):
            raise HTTPException(409, f"the plan continues from a saved state, but {problem}")
        await runner.start_execute(found["folder"], scene, task, instruction, seed, keyframes,
                                   step=2 + found["executions"], view=found["plan"].get("view"),
                                   round_no=plan_round(found["start"]), start=start,
                                   round_mode=found["start"].get("round_mode"),
                                   max_keyframes=(body or ExecuteRequest()).max_keyframes)
        return {"run_id": run_id}

    @app.post("/api/plan/{run_id}/replan", dependencies=[Depends(same_origin)])
    async def replan(run_id: str, body: ReplanRequest | None = None) -> dict:
        body = body or ReplanRequest()
        if body.mode == "start" and body.max_keyframes > 0:
            raise HTTPException(409, START_WITH_K)
        found = history.plan(run_id)
        if found is None:
            raise HTTPException(404, "no such plan")
        last = (found["execution"] or {}).get("end") or {}
        mode = "continue" if last.get("status") == "partial" else body.mode  # a partial run always continues
        if runner.running:
            raise HTTPException(409, "a run is already active")
        start = found["start"]
        if plan_round(start) >= MAX_ROUND:
            raise HTTPException(409, f"this plan is round {plan_round(start)}, the last round of {MAX_ROUND}")
        old_instruction = (start.get("instruction") or "").strip()
        new = body.instruction if body.instruction not in (None, old_instruction) else None
        if new is not None and body.mode == "start":
            raise HTTPException(409, NEW_FROM_START)
        feedback = _feedback(found, history, mode)
        if new is not None and (feedback is None or feedback.status == "error"):
            # No end state: the round starts where that plan started
            feedback = _no_run_feedback(found, feedback.status if feedback else "not run")
        if feedback is None:
            raise HTTPException(404, "the plan has no finished execution yet: execute it first")
        if new is not None:
            feedback = dataclasses.replace(feedback, instruction=new)
        scene = SCENES.get(start.get("scene") or "")
        try:
            req = PlanRequest(instruction=new or old_instruction, scene=start.get("scene") or "",
                              seed=int(start.get("seed", 0)), use_lessons=start.get("use_lessons") is not False,
                              auto=body.auto, rounds=min(body.rounds, MAX_ROUND - plan_round(start)),
                              mode=body.mode, max_keyframes=body.max_keyframes)
        except (TypeError, ValueError):
            req = None
        if scene is None or req is None:
            raise HTTPException(404, "the plan cannot run")
        task = next((t for t in scene.tasks.values() if t.instruction == req.instruction), None)
        state = None
        if not feedback.executed:
            state = start_state(found)
            if state is not None and (problem := state_problem(state, scene, req.seed)):
                raise HTTPException(409, f"the plan started from a saved state, but {problem}")
        elif feedback.mode == "continue":
            state = found["folder"] / END_STATE
            if problem := state_problem(state, scene, req.seed):
                raise HTTPException(409, f"cannot continue from the last execution: {problem}. Re-plan from the start")
        folder = await runner.start_plan(req, scene, task, feedback, state, branch_chain(found, history))
        return {"run_dir": str(folder), "run_id": folder.name}

    @app.post("/api/plan/{rest:path}")
    def no_plan(rest: str) -> None:
        raise HTTPException(404, "no such plan")  # for example an id with a slash in it

    @app.get("/api/plan/{run_id}/view")
    def plan_view(run_id: str) -> dict:
        plan = history.plan(run_id)
        if plan is None:
            raise HTTPException(404, "no such plan")
        return plan["plan"]["view"]

    def image_url(run_id: str, name: str) -> str:
        return f"/api/runs/{run_id}/images/{name}"

    @app.get("/api/runs")
    def list_runs() -> list[dict]:
        return history.runs(active=runner.run_dir.name if runner.running and runner.run_dir else None)

    @app.get("/api/runs/{run_id}")
    def run_events(run_id: str) -> list[dict]:
        events = history.events(run_id, image_url)
        if events is None:
            raise HTTPException(404, "no such run")
        return events

    @app.get("/api/runs/{run_id}/chain")
    def run_chain(run_id: str) -> list[str]:
        """The run ids of the chain of a run, from round 1 to this run: the page shows them in one chat."""
        ids = history.chain(run_id)
        if ids is None:
            raise HTTPException(404, "no such run")
        return ids

    @app.get("/api/runs/{run_id}/images/{name}")
    def run_image(run_id: str, name: str) -> FileResponse:
        path = history.image(run_id, name)
        if path is None:
            raise HTTPException(404, "no such image")
        return FileResponse(path, media_type="image/png")

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
