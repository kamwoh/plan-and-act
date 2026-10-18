"""The plan view of the MuJoCo arm environment: cameras, fixtures, the agent shape and the timed track."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from armmind.plan.formats import Entity, Keyframe, build_track, match_entities, norm_name
from armmind.plan.scores import perception, plan_checks
from armmind.scenes.base import rgba
from armmind.sim.arm import GRIP_S, SETTLE_S, Arm, clamp_target, move_duration
from armmind.sim.render import GRID_X, GRID_Y
from armmind.sim.world import World
from armmind.types import fmt

VIEW_VERSION = 1
TABLE = Entity("table", "box", (2.0, 2.0, 0.01), (0.0, 0.0, -0.005), 0.0, (0.55, 0.5, 0.42, 1.0))
OPEN_GAP = 0.08  # the finger gap of the open gripper
# The gripper as boxes in its own frame: the origin is the point between the fingertips, and the fingers
# close along x. Measured from the Panda model: the fingertips reach 1 cm below the origin.
HAND = {"name": "hand", "shape": "box", "size": [0.20, 0.058, 0.075], "xyz": [0.0, 0.0, 0.0825], "slide": 0}
FINGER_SIZE, FINGER_Z = [0.027, 0.021, 0.057], 0.0185
AGENT_COLOR = [0.86, 0.87, 0.9, 1.0]


def _vec(v) -> list[float]:
    return [round(float(x), 6) for x in v]


def cameras(world: World, width: int, height: int) -> list[dict]:
    """The pose and the field of view of each camera of the model."""
    m, d, scene = world.model, world.data, world.scene
    out = []
    for c in range(m.ncam):
        name = m.camera(c).name
        rot = d.cam_xmat[c].reshape(3, 3)  # the columns are the camera x, y and z axes
        out.append({"name": name, "label": scene.camera_labels.get(name, name), "pos": _vec(d.cam_xpos[c]),
                    "forward": _vec(-rot[:, 2]), "up": _vec(rot[:, 1]), "fovy": float(m.cam_fovy[c]),
                    "width": width, "height": height, "grid": name not in scene.camera_overrides})
    return out


def fixtures(world: World) -> list[Entity]:
    """The table and the fixtures of the scene. Each one is where its geom is now, so a drawer that starts
    open is drawn open."""
    out = [TABLE]
    for e in world.scene.fixtures:
        xyz = tuple(round(float(v), 6) for v in world.data.geom_xpos[world.model.geom(e.name).id])
        out.append(Entity(e.name, e.shape, e.size, xyz, e.yaw, e.color))
    return out


def true_entities(world: World) -> list[Entity]:
    """The true objects now, as boxes or cylinders with their measured sizes."""
    out = []
    for o in world.objects:
        info = world.object_info(o.name)
        shape = "cylinder" if o.shape == "cylinder" or o.round else "box"
        out.append(Entity(o.name, shape, (info.width, info.depth, info.height),
                          tuple(round(float(v), 6) for v in world.object_pos(o.name)),
                          round(world.object_yaw(o.name), 3), rgba(o.rgba) if o.rgba else None))
    return out


def aliases(world: World) -> dict[str, list[str]]:
    """The other names of each object: its label and the first part of the label, as `expect` accepts."""
    return {o.name: [o.label, o.label.split(",")[0]] for o in world.objects}


def resolve(name: str | None, believed: list[Entity]) -> Entity | None:
    if name is None:
        return None
    return next((b for b in believed if norm_name(b.name) == norm_name(name)), None)


def finger_gap(k: Keyframe, believed: list[Entity]) -> float:
    """Open: 0.08. Closed: the narrow width of the carried entity, at most 0.08, or 0 on nothing."""
    if k.state == "open":
        return OPEN_GAP
    held = resolve(k.carrying, believed)
    return round(min(OPEN_GAP, held.size[0], held.size[1]), 4) if held is not None else 0.0


def true_gap(state: str, held: Entity | None) -> float:
    """The finger gap with a true held object: as finger_gap, from the measured narrow width."""
    if state == "open":
        return OPEN_GAP
    return min(OPEN_GAP, held.size[0], held.size[1]) if held is not None else 0.0


def agent() -> dict:
    fingers = [{"name": f"finger_{side}", "shape": "box", "size": FINGER_SIZE,
                "xyz": [sign * FINGER_SIZE[0] / 2, 0.0, FINGER_Z], "slide": sign}
               for side, sign in (("left", 1), ("right", -1))]
    return {"parts": [HAND, *fingers], "slide_axis": [1.0, 0.0, 0.0], "color": AGENT_COLOR}


def timed_keyframes(belief, keyframes) -> list[Keyframe]:
    """The keyframes as the arm runs them: each target is moved into the workspace, and the carried name
    becomes the believed entity's name (None when no believed entity has it). A move keeps only a believed
    entity, under its own name."""
    out, believed = [], list(belief)
    for k in keyframes:
        held = resolve(k.carrying, believed)
        moves = tuple(replace(m, name=e.name) for m in k.moves if (e := resolve(m.name, believed)) is not None)
        out.append(Keyframe(tuple(round(float(v), 6) for v in clamp_target(k.xyz)), k.yaw, k.state,
                            held.name if held else None, k.note, moves))
    return out


def build_view(world: World, belief, keyframes, marks: list[dict] | None, width: int, height: int) -> dict:
    believed = list(belief)
    start = {"xyz": [round(float(v), 4) for v in world.tcp()], "yaw": round(world.gripper_yaw(), 1), "state": "open"}
    track = build_track(start, timed_keyframes(believed, keyframes),
                        move_s=lambda a, b, ya, yb: move_duration(np.asarray(a), np.asarray(b), ya, yb),
                        settle_s=SETTLE_S, change_s=GRIP_S, gap=lambda k: finger_gap(k, believed))
    return {
        "version": VIEW_VERSION,
        "cameras": cameras(world, width, height),
        "grid": {"x": list(GRID_X), "y": list(GRID_Y), "z": 0.0},
        "fixtures": [e.to_dict() for e in fixtures(world)],
        "agent": agent(),
        "believed": [e.to_dict() for e in believed],
        "true_start": [e.to_dict() for e in true_entities(world)],
        "true_match": match_entities([b.name for b in believed], aliases(world)),
        "track": track,
        "marks": list(marks or []),
    }


def check_plan(world: World, arm: Arm, belief, keyframes) -> dict:
    """Score the belief against the true scene now, and check each keyframe (call it before any motion)."""
    truth, names = true_entities(world), aliases(world)
    round_names = {o.name for o in world.objects if o.round}
    believed = list(belief)
    p = perception(believed, truth, names, round_names)

    def reach(xyz, yaw: float) -> str | None:
        target = clamp_target(xyz)
        if not np.allclose(target, xyz):
            return f"outside the workspace: the arm goes to {fmt(target)}"
        return None if arm.reachable(target, yaw) else "no joint solution for this point"

    start = {"xyz": tuple(float(v) for v in world.tcp()), "yaw": world.gripper_yaw(), "state": "open"}
    match = match_entities([b.name for b in believed], names)
    marks = plan_checks(start, list(keyframes), truth, names, round_names, agent()["parts"], reach,
                        open_state="open", gap=true_gap, believed_to_true={b: t for t, b in match.items()})
    return {"perception": p["rows"], "missing": p["missing"], "invented": p["invented"],
            "summary": {**p["summary"], "failed_keyframes": sum(not m["ok"] for m in marks)}, "marks": marks}
