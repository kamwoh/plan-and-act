"""Checks the model's JSON reply and turns it into actions."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from armmind.plan.formats import Entity, Keyframe, Move, norm_name
from armmind.plan.track import wrap_deg
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


def _expect(i: int, a: dict) -> str | None:
    if "expect" not in a:
        return None
    e = a["expect"]
    if not (isinstance(e, dict) and set(e) == {"holding"} and isinstance(e["holding"], str)):
        raise ReplyError(f"action {i}: 'expect' must be {{\"holding\": \"<object>\"}}")
    return e["holding"].strip()


def _action(i: int, a) -> Action:
    if not isinstance(a, dict):
        raise ReplyError(f"action {i} is not an object")
    return replace(_kind(i, a), expect=_expect(i, a))


def _kind(i: int, a: dict) -> Action:
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
    return Reply(str(obj.get("observation", "")), str(obj.get("plan", "")), tuple(actions[:max_actions]),
                 notes, tuple(todo[:5]), len(actions) > max_actions, done, _lessons(obj))


def _lessons(obj: dict) -> tuple[str, ...]:
    """The optional "lessons" of a reply: at most MAX_LESSONS, each cut to LESSON_CHARS."""
    lessons = obj.get("lessons") or []
    if not isinstance(lessons, list) or not all(isinstance(t, str) for t in lessons):
        raise ReplyError("'lessons' must be a list of strings")
    return tuple([t.strip()[:LESSON_CHARS] for t in lessons if t.strip()][:MAX_LESSONS])


# ---- The plan reply ----
MAX_OBJECTS, MAX_KEYFRAMES = 12, 40
YAW_LIMIT = 90.0  # degrees, as move_to clamps it
NAME_CHARS, NOTE_CHARS, SUMMARY_CHARS = 80, 200, 1000
MAX_MOVES = 4  # moved objects in one keyframe


@dataclass(frozen=True)
class PlanReply:
    objects: tuple[Entity, ...]
    keyframes: tuple[Keyframe, ...]
    summary: str
    warnings: tuple[str, ...] = ()  # e.g. "keyframe 2: yaw clamped to 90"
    lessons: tuple[str, ...] = ()


def _vec(where: str, key: str, v, positive: bool = False) -> tuple[float, float, float]:
    if not (isinstance(v, list) and len(v) == 3 and all(_number(x) for x in v)) or (
            positive and not all(x > 0 for x in v)):
        kind = "3 finite numbers above 0" if positive else "3 finite numbers"
        raise ReplyError(f"{where}: '{key}' must be {kind}")
    return (float(v[0]), float(v[1]), float(v[2]))


def _field(where: str, d: dict, key: str):
    if key not in d:
        raise ReplyError(f"{where}: '{key}' is missing")
    return d[key]


def _text(where: str, d: dict, key: str) -> str:
    v = _field(where, d, key)
    if not isinstance(v, str):
        raise ReplyError(f"{where}: '{key}' must be a string")
    return v.strip()


def _yaw(where: str, d: dict) -> float:
    v = _field(where, d, "yaw")
    if not _number(v):
        raise ReplyError(f"{where}: 'yaw' must be a finite number of degrees")
    return float(v)


def _plan_object(i: int, o) -> Entity:
    where = f"object {i + 1}"
    if not isinstance(o, dict):
        raise ReplyError(f"{where} is not an object")
    name = _text(where, o, "name")
    if not name:
        raise ReplyError(f"{where}: 'name' is empty")
    shape = _field(where, o, "shape")
    if shape not in ("box", "cylinder"):
        raise ReplyError(f"{where}: 'shape' must be \"box\" or \"cylinder\"")
    return Entity(name[:NAME_CHARS], shape, _vec(where, "size", _field(where, o, "size"), positive=True),
                  _vec(where, "xyz", _field(where, o, "xyz")), _yaw(where, o))


def _moves(where: str, k: dict, carrying: str | None, names: dict[str, str]) -> tuple[Move, ...]:
    """The optional "moves" of a keyframe, with the believed object names. `names` maps norm_name to a name."""
    if "moves" not in k:
        return ()
    raw = k["moves"]
    if not isinstance(raw, list):
        raise ReplyError(f"{where}: 'moves' must be a list")
    if len(raw) > MAX_MOVES:
        raise ReplyError(f"{where}: 'moves' has {len(raw)} items, at most {MAX_MOVES} are allowed")
    out: list[Move] = []
    for j, m in enumerate(raw):
        at = f"{where}, move {j + 1}"
        if not isinstance(m, dict) or set(m) != {"object", "xyz", "yaw"}:
            raise ReplyError(f"{at}: each move must be {{\"object\": <name>, \"xyz\": [x, y, z], \"yaw\": <degrees>}}")
        if not isinstance(m["object"], str):
            raise ReplyError(f"{at}: 'object' must be a string")
        key = norm_name(m["object"])
        if key not in names:
            raise ReplyError(f"{at}: {m['object']!r} is not one of the 'objects'")
        if carrying is not None and key == norm_name(carrying):
            raise ReplyError(f"{at}: the keyframe holds {m['object']!r}. The held object follows the hand, "
                             "so do not list it in 'moves'")
        if any(norm_name(o.name) == key for o in out):
            raise ReplyError(f"{at}: {m['object']!r} is in 'moves' twice")
        out.append(Move(names[key], _vec(at, "xyz", m["xyz"]), wrap_deg(_yaw(at, m))))
    return tuple(out)


def _plan_keyframe(i: int, k, warnings: list[str], names: dict[str, str]) -> Keyframe:
    where = f"keyframe {i + 1}"
    if not isinstance(k, dict):
        raise ReplyError(f"{where} is not an object")
    xyz = _vec(where, "xyz", _field(where, k, "xyz"))
    want = _yaw(where, k)
    yaw = min(max(want, -YAW_LIMIT), YAW_LIMIT)
    if yaw != want:
        warnings.append(f"{where}: yaw clamped to {yaw:g}")
    state = _field(where, k, "gripper")
    if state not in ("open", "closed"):
        raise ReplyError(f"{where}: 'gripper' must be \"open\" or \"closed\"")
    holding = _text(where, k, "holding")
    carrying = None if holding.lower() in ("", "nothing") else holding[:NAME_CHARS]
    return Keyframe(xyz, yaw, state, carrying, _text(where, k, "note")[:NOTE_CHARS],
                    _moves(where, k, carrying, names))


def _released_not_moved(keyframes: tuple[Keyframe, ...]) -> tuple[Keyframe, ...]:
    """Refuse a move of the object that the same keyframe lets go: the player drops it by itself."""
    before = None
    for i, k in enumerate(keyframes):
        lets_go = before is not None and (k.carrying is None or norm_name(k.carrying) != norm_name(before))
        if lets_go and any(norm_name(m.name) == norm_name(before) for m in k.moves):
            raise ReplyError(f"keyframe {i + 1}, moves: keyframe {i + 1} lets go of {before!r}. A released "
                                 "object falls by itself: move it in a later keyframe")
        before = k.carrying
    return keyframes


def parse_plan(obj) -> PlanReply:
    """Check a plan reply and turn it into entities and keyframes. A bad value raises ReplyError."""
    if not isinstance(obj, dict):
        raise ReplyError("the reply is not a JSON object")
    objects = _field("the reply", obj, "objects")
    if not isinstance(objects, list):
        raise ReplyError("'objects' must be a list")
    if len(objects) > MAX_OBJECTS:
        raise ReplyError(f"'objects' has {len(objects)} items, at most {MAX_OBJECTS} are allowed")
    keyframes = _field("the reply", obj, "keyframes")
    if not isinstance(keyframes, list):
        raise ReplyError("'keyframes' must be a list")
    if not 1 <= len(keyframes) <= MAX_KEYFRAMES:
        raise ReplyError(f"'keyframes' must have 1 to {MAX_KEYFRAMES} items, not {len(keyframes)}")
    summary = _field("the reply", obj, "summary")
    if not isinstance(summary, str):
        raise ReplyError("'summary' must be a string")
    warnings: list[str] = []
    entities = tuple(_plan_object(i, o) for i, o in enumerate(objects))
    seen: dict[str, str] = {}
    for e in entities:  # one name for each object, so that a keyframe names exactly one
        key = norm_name(e.name)
        if key in seen:
            raise ReplyError(f"the object name {e.name!r} is there twice (also as {seen[key]!r}). "
                             "Give each object its own name")
        seen[key] = e.name
    names = {norm_name(e.name): e.name for e in entities}
    return PlanReply(entities,
                     _released_not_moved(tuple(_plan_keyframe(i, k, warnings, names) for i, k in enumerate(keyframes))),
                     summary.strip()[:SUMMARY_CHARS], tuple(warnings), _lessons(obj))
