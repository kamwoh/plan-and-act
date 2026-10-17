"""Command line: `armmind scripted` checks a task of a scene without a model."""
from __future__ import annotations

import argparse

from armmind.scenes import SCENES


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="armmind")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scripted", help="run the hard-coded plan for a task")
    s.add_argument("--scene", choices=list(SCENES), default="cubes")
    s.add_argument("--task", required=True, help="a task of the scene")
    s.add_argument("--seed", type=int, default=0)
    v = sub.add_parser("serve", help="start the web app")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8780)
    v.add_argument("--model", default=None, help="model name for the Claude CLI")
    v.add_argument("--runs", default="runs", help="folder for run logs")
    args = p.parse_args(argv)
    if args.cmd == "serve":
        if args.port == 8765:
            p.error("port 8765 is for ps1-mind. Use another port, for example 8780")
        return _serve(args.host, args.port, args.model, args.runs)
    scene = SCENES[args.scene]
    if args.task not in scene.tasks:
        p.error(f"scene {scene.name} has no task {args.task!r}. Choose from: {', '.join(scene.tasks)}")
    return _scripted(scene.name, args.task, args.seed)


def _serve(host: str, port: int, model: str | None, runs: str) -> int:
    import os
    from pathlib import Path

    import uvicorn

    from armmind.llm.claude_cli import ClaudeCLIBackend
    from armmind.server.app import create_app

    lessons_dir = None  # the default: next to the runs folder
    if os.environ.get("ARMMIND_FAKE_LLM") == "1":  # for tests and screenshots only: no model is called
        import tempfile

        from armmind.llm.fake import DemoBackend
        backend = DemoBackend()
        lessons_dir = Path(tempfile.mkdtemp(prefix="armmind-demo-lessons-"))  # keep invented lessons out
    else:
        backend = ClaudeCLIBackend(model=model, timeout_s=180)
    app = create_app(backend, runs_dir=Path(runs), lessons_dir=lessons_dir)
    print(f"arm-mind: open http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
    return 0


def _scripted(scene_name: str, task_name: str, seed: int) -> int:
    from armmind.sim.arm import Arm
    from armmind.sim.world import World

    scene = SCENES[scene_name]
    task = scene.tasks[task_name]
    w = World(scene)
    w.reset(scene.layout(seed), task.initial_qpos)
    arm = Arm(w)
    arm.reset_ready()
    for action in task.solution(w):
        print(arm.act(action).summary())
    ok = task.success(w)
    print("success" if ok else "failed")
    return 0 if ok else 1
