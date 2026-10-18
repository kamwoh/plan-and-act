"""The poses of entities along a timed track, as the page player computes them. Plain Python only.

web/plan3d.js has the same rules in posesAt. tests/js/track_case.json checks that the two agree."""
from __future__ import annotations

import math
from collections.abc import Callable

Pose = dict  # {"xyz": [x, y, z], "yaw": degrees}


def wrap_deg(a: float) -> float:
    """The angle in (-180, 180]."""
    w = (a + 180.0) % 360.0 - 180.0
    return 180.0 if w == -180.0 else w


def _clamp01(a: float) -> float:
    return min(1.0, max(0.0, a))


def _smooth(a: float) -> float:
    return a * a * (3 - 2 * a)  # the same timing as the motion of the environment


def _lerp(a: float, b: float, s: float) -> float:
    return a + (b - a) * s


def _lerp3(a, b, s: float) -> list[float]:
    return [_lerp(a[i], b[i], s) for i in range(3)]


def agent_at(track: dict, t: float) -> Pose:
    """The agent pose at time t: the xyz and the yaw."""
    prev, t0 = track["start"], 0.0
    for kf in track["keyframes"]:
        if t < kf["t"]:
            s = _smooth(_clamp01((t - t0) / max(1e-6, kf["t"] - t0)))
            return {"xyz": _lerp3(prev["xyz"], kf["xyz"], s), "yaw": _lerp(prev["yaw"], kf["yaw"], s)}
        if t < kf["t_done"]:
            return {"xyz": list(kf["xyz"]), "yaw": kf["yaw"]}
        prev, t0 = kf, kf["t_done"]
    return {"xyz": list(prev["xyz"]), "yaw": prev["yaw"]}


def to_local(agent: Pose, pose: Pose) -> Pose:
    c, s = math.cos(math.radians(-agent["yaw"])), math.sin(math.radians(-agent["yaw"]))
    dx, dy = pose["xyz"][0] - agent["xyz"][0], pose["xyz"][1] - agent["xyz"][1]
    return {"xyz": [c * dx - s * dy, s * dx + c * dy, pose["xyz"][2] - agent["xyz"][2]],
            "yaw": pose["yaw"] - agent["yaw"]}


def to_world(agent: Pose, local: Pose) -> Pose:
    c, s = math.cos(math.radians(agent["yaw"])), math.sin(math.radians(agent["yaw"]))
    lx, ly, lz = local["xyz"]
    return {"xyz": [agent["xyz"][0] + c * lx - s * ly, agent["xyz"][1] + s * lx + c * ly, agent["xyz"][2] + lz],
            "yaw": agent["yaw"] + local["yaw"]}


def move_pose(pose: Pose, move: dict, s: float) -> Pose:
    """A pose part of the way to a move target. The yaw takes the short way round."""
    target = pose["yaw"] + wrap_deg(move["yaw"] - pose["yaw"])
    return {"xyz": _lerp3(pose["xyz"], move["xyz"], s), "yaw": _lerp(pose["yaw"], target, s)}


def keyframe_moves(k: int, kf: dict) -> list[dict]:
    return kf.get("moves") or []


def no_moves(k: int, kf: dict) -> list[dict]:
    return []


def carrying(k: int, kf: dict) -> str | None:
    return kf.get("carrying")


def poses_at(track: dict, entities: list[dict], t: float,
             carried_at: Callable[[int, dict], str | None] = carrying,
             moves_at: Callable[[int, dict], list[dict]] = keyframe_moves) -> dict[str, Pose]:
    """The pose of each entity at time t.

    - carried_at(k, kf) names the entity that the agent carries after keyframe k, or gives None. A carried
      entity follows the agent. A release leaves it where the agent lets go.
    - moves_at(k, kf) gives the moves of keyframe k: [{"name", "xyz", "yaw"}]. On the segment into keyframe k,
      each moved entity goes to its new pose with the timing of the agent. A carried entity does not move."""
    poses = {e["name"]: {"xyz": list(e["xyz"]), "yaw": e.get("yaw") or 0.0} for e in entities}
    carried = None  # (name, pose in the agent frame)
    t0 = 0.0
    for k, kf in enumerate(track["keyframes"]):
        moves = moves_at(k, kf) or []
        if t < kf["t"]:
            s = _smooth(_clamp01((t - t0) / max(1e-6, kf["t"] - t0)))
            for m in moves:
                if m["name"] in poses and not (carried and carried[0] == m["name"]):
                    poses[m["name"]] = move_pose(poses[m["name"]], m, s)
            break
        at = {"xyz": kf["xyz"], "yaw": kf["yaw"]}
        want = carried_at(k, kf) or None
        if carried and carried[0] != want:
            poses[carried[0]] = to_world(at, carried[1])
            carried = None
        for m in moves:
            if m["name"] in poses and not (carried and carried[0] == m["name"]):
                poses[m["name"]] = move_pose(poses[m["name"]], m, 1.0)
        if want and not carried and want in poses:
            carried = (want, to_local(at, poses[want]))
        t0 = kf["t_done"]
    if carried:
        poses[carried[0]] = to_world(agent_at(track, t), carried[1])
    return poses


def poses_after(track: dict, entities: list[dict]) -> list[dict[str, Pose]]:
    """The predicted poses of the entities at the end of each keyframe."""
    return [poses_at(track, entities, kf["t_done"]) for kf in track["keyframes"]]
