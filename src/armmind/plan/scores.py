"""Scores of a plan: the belief against the true scene, and checks of each keyframe. Plain Python only."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace

from armmind.plan.formats import Entity, Keyframe, build_track, match_entities, norm_name
from armmind.plan.track import poses_after

SQUARE_TOL = 0.10  # two horizontal sizes within 10 % make a square footprint
SAMPLE_M = 0.01  # the path check looks at the agent every 1 cm
IGNORE_TARGET_M = 0.03  # the grasp target does not count in the last 3 cm before a close
GRASP_ANGLE_DEG = 20.0
MAX_GRASP_WIDTH = 0.075
GRASP_CENTRE_M = 0.02
COLLIDE_M = 0.002  # an overlap must be deeper than this: the boxes of scanned objects are a little loose


def _round(v: float, n: int = 2) -> float:
    return round(v, n) + 0.0  # + 0.0 turns -0.0 into 0.0


def is_round(e: Entity, round_names: set[str]) -> bool:
    return e.shape in ("cylinder", "sphere") or e.name in round_names


def symmetry(e: Entity, round_names: set[str]) -> float | None:
    """The turn in degrees after which the object looks the same: 90 for a square footprint, 180 for a box,
    and None for a round object."""
    if is_round(e, round_names):
        return None
    w, d = e.size[0], e.size[1]
    return 90.0 if abs(w - d) <= SQUARE_TOL * max(w, d) else 180.0


def _fold(diff: float, period: float) -> float:
    d = diff % period
    return min(d, period - d)


def yaw_error(true: Entity, believed_yaw: float, round_names: set[str]) -> float | None:
    period = symmetry(true, round_names)
    return None if period is None else _fold(believed_yaw - true.yaw, period)


def canonical(e: Entity) -> Entity:
    """The same box with the shorter horizontal side along x: [w, d] at yaw t equals [d, w] at t + 90."""
    if e.size[0] <= e.size[1]:
        return e
    return Entity(e.name, e.shape, (e.size[1], e.size[0], e.size[2]), e.xyz, e.yaw + 90.0, e.color)


def perception(believed: list[Entity], truth: list[Entity], aliases: dict[str, list[str]],
               round_names: set[str]) -> dict:
    """Errors of each matched object, the missing and the invented objects, and a summary."""
    match = match_entities([b.name for b in believed], {t.name: aliases.get(t.name, []) for t in truth})
    by_name = {b.name: b for b in believed}
    rows = []
    for t in truth:
        if t.name not in match:
            continue
        b, t = canonical(by_name[match[t.name]]), canonical(t)
        yaw = yaw_error(t, b.yaw, round_names)
        rows.append({"name": b.name, "true_name": t.name,
                     "xy_cm": _round(math.hypot(b.xyz[0] - t.xyz[0], b.xyz[1] - t.xyz[1]) * 100),
                     "z_cm": _round((b.xyz[2] - t.xyz[2]) * 100),
                     "size_cm": [_round((bs - ts) * 100) for bs, ts in zip(b.size, t.size, strict=True)],
                     "yaw_deg": None if yaw is None else _round(yaw, 1)})
    missing = [t.name for t in truth if t.name not in match]
    invented = [b.name for b in believed if b.name not in set(match.values())]
    xy = [r["xy_cm"] for r in rows]
    return {"rows": rows, "missing": missing, "invented": invented,
            "summary": {"mean_xy_cm": _round(sum(xy) / len(xy)) if xy else 0.0, "max_xy_cm": max(xy, default=0.0),
                        "matched": len(rows), "missing": len(missing), "invented": len(invented)}}


# ---- Boxes that turn only about z ----
Box = tuple[tuple[float, float, float], tuple[float, float, float], float]  # centre, full size, yaw in degrees


def _corners(box: Box) -> list[tuple[float, float]]:
    (x, y, _), (w, d, _), yaw = box
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return [(x + c * u - s * v, y + s * u + c * v) for u, v in ((w / 2, d / 2), (-w / 2, d / 2), (-w / 2, -d / 2),
                                                                 (w / 2, -d / 2))]


def boxes_overlap(a: Box, b: Box, margin: float = 1e-4) -> bool:
    """True when the two boxes share a volume deeper than `margin` (a touch is not an overlap)."""
    if abs(a[0][2] - b[0][2]) >= (a[1][2] + b[1][2]) / 2 - margin:
        return False
    ca, cb = _corners(a), _corners(b)
    for yaw in (a[2], a[2] + 90.0, b[2], b[2] + 90.0):  # the separating axis test in the plane
        ax = (math.cos(math.radians(yaw)), math.sin(math.radians(yaw)))
        pa = [p[0] * ax[0] + p[1] * ax[1] for p in ca]
        pb = [p[0] * ax[0] + p[1] * ax[1] for p in cb]
        if min(max(pa), max(pb)) - max(min(pa), min(pb)) <= margin:
            return False
    return True


def agent_boxes(parts: list[dict], xyz, yaw: float, gap: float, slide_axis=(1.0, 0.0, 0.0)) -> list[tuple[str, Box]]:
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    out = []
    for p in parts:
        lx, ly, lz = (p["xyz"][i] + slide_axis[i] * p.get("slide", 0) * gap / 2 for i in range(3))
        centre = (xyz[0] + c * lx - s * ly, xyz[1] + s * lx + c * ly, xyz[2] + lz)
        out.append((p["name"], (centre, tuple(p["size"]), yaw)))
    return out


def _entity_box(e: Entity, pose) -> Box:
    if e.shape in ("cylinder", "sphere"):
        return (pose[0], (e.size[0], e.size[1], e.size[2]), 0.0)
    return (pose[0], e.size, pose[1])


def _narrow_width(e: Entity) -> float:
    return min(e.size[0], e.size[1])


def plan_checks(start: dict, keyframes, truth: list[Entity], aliases: dict[str, list[str]], round_names: set[str],
                parts: list[dict], reach: Callable[[tuple, float], str | None], *, open_state: str,
                gap: Callable[[str, Entity | None], float],
                believed_to_true: dict[str, str] | None = None, belief: list[Entity] | None = None) -> list[dict]:
    """One mark for each keyframe: {"ok", "checks": {name: "ok" or a reason}, "info", "holds"}.

    The environment gives these rules:
    - reach(xyz, yaw): None, or the reason why it cannot reach the pose.
    - open_state: the state that holds nothing. A change away from it grasps, a change back to it releases.
    - gap(state, held): the slide gap of the agent parts in a state, with the held true entity or None.
    "holds" is the true object in the agent after the keyframe, or None.
    The path into a keyframe may touch the objects in the moves of that keyframe. After it, each moved true
    object shifts by the predicted displacement: the move against the believed pose before the keyframe in
    `belief`, as the page player has it. Without a belief, it goes to the move pose."""
    by_name = {t.name: t for t in truth}
    poses = {t.name: (t.xyz, t.yaw) for t in truth}  # where each true object is, as the plan moves them
    keys = {name: {norm_name(name), *(norm_name(a) for a in aliases.get(name, []) if a)} for name in by_name}
    believed_to_true = {norm_name(b): t for b, t in (believed_to_true or {}).items()}

    def true_name(name: str | None) -> str | None:
        if name is None:
            return None
        n = norm_name(name)
        return next((t for t, k in keys.items() if n in k), believed_to_true.get(n))

    def closes(before: str, k: Keyframe) -> bool:
        return before == open_state and k.state != open_state

    xyz, yaw, state = tuple(start["xyz"]), float(start["yaw"]), start["state"]
    held, local = None, None  # the true object in the hand, and its pose in the hand frame
    keyframes = list(keyframes)
    believed = _believed_before(start, keyframes, belief or [])
    marks = []
    for i, k in enumerate(keyframes):
        k: Keyframe
        checks, info = {}, ""
        why = reach(k.xyz, k.yaw)
        checks["reach"] = why or "ok"
        closing = closes(state, k)
        target = true_name(k.carrying) if closing else None
        # A descent that stops open, with a close in place at the next keyframe, may touch its target too.
        nxt = keyframes[i + 1] if i + 1 < len(keyframes) else None
        near = target or (true_name(nxt.carrying) if nxt is not None and closes(k.state, nxt) else None)
        moved = {t for m in k.moves if (t := true_name(m.name)) is not None}  # contact is expected there
        checks["collision"] = _path_check(xyz, yaw, k, parts, gap(state, by_name.get(held)), held, near,
                                          by_name, poses, moved)
        if held is not None:  # the held object moves with the hand to the keyframe
            poses[held] = _to_world(k.xyz, k.yaw, local)
        for m in k.moves:
            if (t := true_name(m.name)) is not None and t != held:
                poses[t] = _shift(poses[t], m, believed[i].get(norm_name(m.name)))
        if closing and k.carrying is not None:
            checks["grasp"] = _grasp_check(k, target, by_name, poses, round_names)
            if target is not None and checks["grasp"] == "ok":
                held, local = target, _to_local(k.xyz, k.yaw, poses[target])
        if k.state == open_state and state != open_state and held is not None:
            bottom = poses[held][0][2] - by_name[held].size[2] / 2
            checks["release"] = "ok"
            info = f"{held}: {bottom * 100:.1f} cm above the table"
            poses[held] = drop(held, by_name, poses)
            held, local = None, None
        marks.append({"ok": all(v == "ok" for v in checks.values()), "checks": checks, "info": info, "holds": held})
        xyz, yaw, state = k.xyz, k.yaw, k.state
    return marks


def _believed_before(start: dict, keyframes: list[Keyframe], belief: list[Entity]) -> list[dict]:
    """The believed pose of each entity before each keyframe, by norm_name, with the rules of the player."""
    if not belief:
        return [{} for _ in keyframes]
    names = {norm_name(b.name): b.name for b in belief}
    resolved = [replace(k, carrying=names.get(norm_name(k.carrying)) if k.carrying else None,
                        moves=tuple(replace(m, name=names.get(norm_name(m.name), m.name)) for m in k.moves))
                for k in keyframes]
    track = build_track(start, resolved, move_s=lambda *_: 1.0, settle_s=0.0, change_s=0.0, gap=lambda _: 0.0)
    entities = [b.to_dict() for b in belief]
    after = poses_after(track, entities)
    first = {e["name"]: {"xyz": e["xyz"], "yaw": e["yaw"]} for e in entities}
    return [{norm_name(n): p for n, p in poses.items()} for poses in [first, *after[:-1]]] if keyframes else []


def _shift(pose, move, before: dict | None):
    """A true pose after a predicted move: shifted by the move against the believed pose before it."""
    if before is None:
        return (tuple(move.xyz), move.yaw)
    (x, y, z), yaw = pose
    d = [move.xyz[j] - before["xyz"][j] for j in range(3)]
    return ((x + d[0], y + d[1], z + d[2]), yaw + move.yaw - before["yaw"])


def drop(name: str, by_name: dict[str, Entity], poses: dict):
    """A released object falls onto the table or onto the highest object below it."""
    e, (centre, yaw) = by_name[name], poses[name]
    flat = _entity_box(e, ((centre[0], centre[1], 0.0), yaw))
    flat = (flat[0], (flat[1][0], flat[1][1], 1.0), flat[2])
    support = 0.0
    for other, o in by_name.items():
        if other == name:
            continue
        top = poses[other][0][2] + o.size[2] / 2
        box = _entity_box(o, poses[other])
        if top <= centre[2] and boxes_overlap(flat, ((box[0][0], box[0][1], 0.0), (box[1][0], box[1][1], 1.0), box[2])):
            support = max(support, top)
    return ((centre[0], centre[1], min(centre[2], support + e.size[2] / 2)), yaw)  # it never rises


def _to_local(xyz, yaw: float, pose):
    c, s = math.cos(math.radians(-yaw)), math.sin(math.radians(-yaw))
    dx, dy = pose[0][0] - xyz[0], pose[0][1] - xyz[1]
    return ((c * dx - s * dy, s * dx + c * dy, pose[0][2] - xyz[2]), pose[1] - yaw)


def _to_world(xyz, yaw: float, local):
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    lx, ly, lz = local[0]
    return ((xyz[0] + c * lx - s * ly, xyz[1] + s * lx + c * ly, xyz[2] + lz), local[1] + yaw)


def _path_check(xyz, yaw: float, k: Keyframe, parts, gap: float, held: str | None, target: str | None,
                by_name: dict[str, Entity], poses: dict, ignore: set[str] = frozenset()) -> str:
    # Sample every 1 cm of the move, and every 1 cm of the arc of the outer part corner for a turn.
    radius = max((math.hypot(abs(p["xyz"][0]) + abs(p.get("slide", 0)) * gap / 2 + p["size"][0] / 2,
                             abs(p["xyz"][1]) + p["size"][1] / 2) for p in parts), default=0.0)
    arc = math.radians(abs(k.yaw - yaw)) * radius
    n = max(1, math.ceil(max(math.dist(xyz, k.xyz), arc) / SAMPLE_M))
    for i in range(1, n + 1):
        a = i / n
        p = tuple(xyz[j] + a * (k.xyz[j] - xyz[j]) for j in range(3))
        y = yaw + a * (k.yaw - yaw)
        near_target = math.dist(p, k.xyz) <= IGNORE_TARGET_M
        for name, e in by_name.items():
            if name == held or name in ignore or (name == target and near_target):
                continue
            box = _entity_box(e, poses[name])
            for part, agent_box in agent_boxes(parts, p, y, gap):
                if boxes_overlap(agent_box, box, COLLIDE_M):
                    return (f"the {part.replace('_', ' ')} passes through {name} at "
                            f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
    return "ok"


def _grasp_check(k: Keyframe, target: str | None, by_name: dict[str, Entity], poses: dict,
                 round_names: set[str]) -> str:
    if target is None:
        return f"no true object is called {k.carrying!r}"
    e, (centre, obj_yaw) = by_name[target], poses[target]
    why = []
    period = symmetry(e, round_names)
    if period is not None:
        narrow = obj_yaw + (0.0 if e.size[0] <= e.size[1] else 90.0)  # the direction of the narrow side
        off = _fold(k.yaw - narrow, min(period, 180.0))
        if off > GRASP_ANGLE_DEG:
            why.append(f"the fingers close {off:.0f}° away from the narrow side of {target}")
    if _narrow_width(e) > MAX_GRASP_WIDTH:
        why.append(f"{target} is too wide: its narrow side is {_narrow_width(e) * 100:.1f} cm")
    miss = math.hypot(k.xyz[0] - centre[0], k.xyz[1] - centre[1])
    if miss > GRASP_CENTRE_M:
        why.append(f"the hand is {miss * 100:.1f} cm from the centre of {target}")
    return "; ".join(why) or "ok"


# ---- The prediction against the physics, after the plan ran ----
MOVED_M = 0.02  # an object that goes further than 2 cm moved


def prediction(track: dict, believed: list[dict], true_start: dict[str, dict], true_after: list[dict[str, dict]],
               match: dict[str, str]) -> dict:
    """Compare the predicted poses of the belief with the true poses after each keyframe that ran.

    - track and believed come from the plan view. The prediction follows the rules of the page player.
    - true_start and true_after give {"xyz", "yaw"} for each true name, before the plan and after each keyframe.
    - match maps a true name to a believed name.
    The displacement error compares the two moves from their own starts, so the perception error stays out."""
    kfs = track["keyframes"]
    ran = min(len(true_after), len(kfs))
    rows, lines = [], []
    if ran:
        predicted = poses_after(track, believed)[:ran]
        start = {e["name"]: e for e in believed}
        back = {b: t for t, b in match.items() if b in start and t in true_start}

        def moves_at(k: int, b: str, t: str) -> tuple[list[float], list[float]]:
            p, b0 = predicted[k][b]["xyz"], start[b]["xyz"]
            t1, t0 = true_after[k][t]["xyz"], true_start[t]["xyz"]
            return [p[j] - b0[j] for j in range(3)], [t1[j] - t0[j] for j in range(3)]

        def xy_cm(a, b) -> float:
            return _round(math.hypot(a[0] - b[0], a[1] - b[1]) * 100)

        for b, t in back.items():
            want, got = moves_at(ran - 1, b, t)
            p, t1 = predicted[ran - 1][b]["xyz"], true_after[ran - 1][t]["xyz"]
            said, did = math.dist(want, (0, 0, 0)) > MOVED_M, math.dist(got, (0, 0, 0)) > MOVED_M
            rows.append({"name": b, "true_name": t, "predicted_cm": [_round(v * 100) for v in want],
                         "real_cm": [_round(v * 100) for v in got], "disp_cm": xy_cm(want, got),
                         "final_xy_cm": xy_cm(p, t1), "final_z_cm": _round((p[2] - t1[2]) * 100),
                         "predicted_moved": said, "real_moved": did, "agree": said == did})
        for k in range(ran):
            for m in kfs[k].get("moves") or []:
                t = back.get(m["name"])
                lines.append({"keyframe": k + 1, "name": m["name"], "true_name": t,
                              "disp_cm": xy_cm(*moves_at(k, m["name"], t)) if t else None})
    disp = [r["disp_cm"] for r in rows]
    return {"rows": rows, "keyframes": lines,
            "summary": {"mean_disp_cm": _round(sum(disp) / len(disp)) if disp else 0.0,
                        "max_disp_cm": max(disp, default=0.0),
                        "mean_final_xy_cm": _round(sum(r["final_xy_cm"] for r in rows) / len(rows)) if rows else 0.0,
                        "agree": sum(r["agree"] for r in rows), "count": len(rows),
                        "ran": ran, "of": len(kfs), "partial": ran < len(kfs)}}
