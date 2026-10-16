"""Command line: `armmind scripted` checks a task without a model."""
from __future__ import annotations

import argparse

from armmind.tasks import TASKS


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="armmind")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scripted", help="run the hard-coded plan for a task")
    s.add_argument("--task", choices=sorted(TASKS), required=True)
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
    return _scripted(args.task, args.seed)


def _serve(host: str, port: int, model: str | None, runs: str) -> int:
    from pathlib import Path

    import uvicorn

    from armmind.llm.claude_cli import ClaudeCLIBackend
    from armmind.server.app import create_app

    app = create_app(ClaudeCLIBackend(model=model, timeout_s=180), runs_dir=Path(runs))
    print(f"arm-mind: open http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
    return 0


def _scripted(task: str, seed: int) -> int:
    from armmind.policy.scripted import scripted_plan
    from armmind.sim.arm import Arm
    from armmind.sim.world import World

    w = World()
    w.reset(TASKS[task].layout(seed))
    arm = Arm(w)
    arm.reset_ready()
    for action in scripted_plan(task, w):
        print(arm.act(action).summary())
    ok = TASKS[task].success(w)
    print("success" if ok else "failed")
    return 0 if ok else 1
