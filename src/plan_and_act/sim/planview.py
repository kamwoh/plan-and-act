"""The plan view of the MuJoCo arm environment: cameras, fixtures, the agent shape and the timed track."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from plan_and_act.plan.formats import Entity, Keyframe, build_track, match_entities, norm_name
from plan_and_act.plan.scores import perception, plan_checks
from plan_and_act.scenes.base import rgba
from plan_and_act.sim.arm import GRIP_S, SETTLE_S, Arm, clamp_target, move_duration
from plan_and_act.sim.render import GRID_X, GRID_Y
from plan_and_act.sim.world import World
from plan_and_act.types import fmt

VIEW_VERSION = 1
TABLE = Entity("table", "box", (2.0, 2.0, 0.01), (0.0, 0.0, -0.005), 0.0, (0.55, 0.5, 0.42, 1.0))


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


def finger_gap(k: Keyframe, believed: list[Entity], open_gap: float) -> float:
    """Open: the open gap. Closed: the narrow width of the carried entity, at most the open gap, or 0 on
    nothing."""
    if k.state == "open":
        return open_gap
    held = resolve(k.carrying, believed)
    return round(min(open_gap, held.size[0], held.size[1]), 4) if held is not None else 0.0


def true_gap(state: str, held: Entity | None, open_gap: float) -> float:
    """The finger gap with a true held object: as finger_gap, from the measured narrow width."""
    if state == "open":
        return open_gap
    return min(open_gap, held.size[0], held.size[1]) if held is not None else 0.0


def agent(world: World) -> dict:
    """The gripper of the robot as boxes in its own frame: the origin is the tcp, and the fingers close along x."""
    return world.parts.agent


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


def _start_held(world: World, held: tuple[str, ...]) -> str | None:
    """The one true object that the gripper holds at the start, or None."""
    return held[0] if len(held) == 1 and held[0] in {o.name for o in world.objects} else None


def build_view(world: World, belief, keyframes, marks: list[dict] | None, width: int, height: int,
               state: str = "open", held: tuple[str, ...] = ()) -> dict:
    """The view of the page. `state` and `held` give the gripper at the start: a round that continues can
    start closed on an object. The start of the track then carries the believed match of that object."""
    believed = list(belief)
    match = match_entities([b.name for b in believed], aliases(world))
    start = {"xyz": [round(float(v), 4) for v in world.tcp()], "yaw": round(world.gripper_yaw(), 1), "state": state}
    if (true_held := _start_held(world, held)) is not None and true_held in match:
        start["carrying"] = match[true_held]
    track = build_track(start, timed_keyframes(believed, keyframes),
                        move_s=lambda a, b, ya, yb: move_duration(np.asarray(a), np.asarray(b), ya, yb),
                        settle_s=SETTLE_S, change_s=GRIP_S,
                        gap=lambda k: finger_gap(k, believed, world.parts.open_width))
    return {
        "version": VIEW_VERSION,
        "cameras": cameras(world, width, height),
        "grid": {"x": list(GRID_X), "y": list(GRID_Y), "z": 0.0},
        "fixtures": [e.to_dict() for e in fixtures(world)],
        "agent": agent(world),
        "believed": [e.to_dict() for e in believed],
        "true_match": match,
        "track": track,
        "marks": list(marks or []),
    }


def check_plan(world: World, arm: Arm, belief, keyframes, state: str = "open") -> dict:
    """Score the belief against the true scene now, and check each keyframe (call it before any motion).
    `state` is the gripper at the start. When it holds an object now, the checks start with it in the hand."""
    truth, names = true_entities(world), aliases(world)
    round_names = {o.name for o in world.objects if o.round}
    believed = list(belief)
    p = perception(believed, truth, names, round_names)

    def reach(xyz, yaw: float) -> str | None:
        target = clamp_target(xyz)
        if not np.allclose(target, xyz):
            return f"outside the workspace: the arm goes to {fmt(target)}"
        return None if arm.reachable(target, yaw) else "no joint solution for this point"

    start = {"xyz": tuple(float(v) for v in world.tcp()), "yaw": world.gripper_yaw(), "state": state}
    match = match_entities([b.name for b in believed], names)
    if state != "open" and (true_held := _start_held(world, arm.held())) is not None:
        start["holds"] = true_held
        if true_held in match:
            start["carrying"] = match[true_held]
    marks = plan_checks(start, list(keyframes), truth, names, round_names, agent(world)["parts"], reach,
                        open_state="open", gap=lambda state, held: true_gap(state, held, world.parts.open_width), believed_to_true={b: t for t, b in match.items()},
                        belief=believed)
    return {"perception": p["rows"], "missing": p["missing"], "invented": p["invented"],
            "summary": {**p["summary"], "failed_keyframes": sum(not m["ok"] for m in marks)}, "marks": marks}
