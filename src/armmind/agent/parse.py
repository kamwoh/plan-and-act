"""Checks the model's JSON reply and turns it into actions."""
from __future__ import annotations

import math
from dataclasses import dataclass

from armmind.types import Action

MAX_WAIT_S = 3.0
MAX_LESSONS, LESSON_CHARS = 3, 200


class ReplyError(ValueError):
    pass


@dataclass(frozen=True)
class Reply:
    observation: str
    plan: str
    actions: tuple[Action, ...]
    notes: str | None
    todo: tuple[str, ...]
    truncated: bool = False
    done: bool = False
    lessons: tuple[str, ...] = ()


def _number(v) -> bool:
    if not isinstance(v, int | float) or isinstance(v, bool):
        return False
    try:
        return math.isfinite(v)
    except OverflowError:
        return False


def _action(i: int, a) -> Action:
    if not isinstance(a, dict):
        raise ReplyError(f"action {i} is not an object")
    kind = a.get("type")
    if kind == "move_to":
        xyz = a.get("xyz")
        if not (isinstance(xyz, list) and len(xyz) == 3 and all(_number(v) for v in xyz)):
            raise ReplyError(f"action {i}: move_to needs 'xyz' with 3 finite numbers")
        if "yaw" in a and not _number(a["yaw"]):
            raise ReplyError(f"action {i}: 'yaw' must be a finite number of degrees")
        yaw = float(a["yaw"]) if "yaw" in a else None  # the arm clamps the range
        return Action("move_to", (float(xyz[0]), float(xyz[1]), float(xyz[2])), yaw=yaw)
    if kind in ("open", "close"):
        return Action(kind)
    if kind == "wait":
        s = a.get("seconds", 0.5)
        if not _number(s):
            raise ReplyError(f"action {i}: wait needs 'seconds' as a number")
        return Action("wait", seconds=min(max(float(s), 0.0), MAX_WAIT_S))
    raise ReplyError(f"action {i}: unknown type {kind!r}")


def parse_reply(obj, max_actions: int = 8) -> Reply:
    if not isinstance(obj, dict):
        raise ReplyError("the reply is not a JSON object")
    raw = obj.get("actions")
    if not isinstance(raw, list):
        raise ReplyError("'actions' must be a list")
    actions = [_action(i, a) for i, a in enumerate(raw)]
    todo = obj.get("todo") or []
    if not isinstance(todo, list) or not all(isinstance(t, str) for t in todo):
        raise ReplyError("'todo' must be a list of strings")
    notes = obj.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ReplyError("'notes' must be a string")
    done = obj.get("done", False)
    if not isinstance(done, bool):
        raise ReplyError("'done' must be true or false")
    lessons = obj.get("lessons") or []
    if not isinstance(lessons, list) or not all(isinstance(t, str) for t in lessons):
        raise ReplyError("'lessons' must be a list of strings")
    lessons = [t.strip()[:LESSON_CHARS] for t in lessons if t.strip()][:MAX_LESSONS]
    return Reply(str(obj.get("observation", "")), str(obj.get("plan", "")), tuple(actions[:max_actions]),
                 notes, tuple(todo[:5]), len(actions) > max_actions, done, tuple(lessons))
