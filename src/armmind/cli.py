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
    args = p.parse_args(argv)
    return _scripted(args.task, args.seed)


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
