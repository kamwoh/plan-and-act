"""The robot arms. Each one is an MJCF arm, maybe a separate gripper, and the few facts that the code cannot
find in the model: the gripper bodies, the base position and the start point.

The code finds the rest. The arm joints are the hinges from the world to the gripper. The tcp site is the
point between the fingertips: its z axis points out of the gripper and its y axis points from the right finger
to the left finger. At yaw 0, every arm points the gripper down and closes the fingers along world x.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from plan_and_act.sim.assets import REPO_ROOT, AssetsMissing

TCP_SITE = "tcp"
GRIPPER_PREFIX = "gripper/"  # the names of a separate gripper start with this
# The tcp frame at yaw 0, in world axes (the columns): x along world y, y along world x, z down.
ROT0 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
TIP_MARGIN = 0.01  # the tcp is this far back from the fingertips
PAD_BAND = 0.03  # the width is the gap between the finger points from this far behind the tcp to the tips
PAD_HALF = (0.012, 0.003, 0.015)  # the largest half size of a pad: across, thickness, along the gripper axis
# Contact bits of a robot without self contact: its own bit, and it takes the bits of every scene geom.
# Two robot geoms never touch, because neither one takes the robot bit.
ROBOT_CONTYPE = 1 << 8
ROBOT_CONAFFINITY = 0x7FFFFFFF & ~ROBOT_CONTYPE
OPEN_S = 1.0  # the time that the gripper gets to open before the code measures it
# The physics of every robot: the arm code steps 2 ms at a time. These are the MuJoCo defaults.
TIMESTEP, ITERATIONS, LS_ITERATIONS = 0.002, 100, 50
_OPTIONS = ("timestep", "integrator", "cone", "impratio", "iterations", "ls_iterations", "noslip_iterations",
            "tolerance", "ls_tolerance", "jacobian", "solver", "gravity", "wind", "density", "viscosity", "o_margin")


def menagerie_dir() -> Path:
    return Path(os.environ.get("PLAN_AND_ACT_ASSETS", REPO_ROOT / "assets" / "mujoco_menagerie"))


def i2rt_dir() -> Path:
    return Path(os.environ.get("PLAN_AND_ACT_I2RT", REPO_ROOT / "assets" / "i2rt"))


def _file(ref: str) -> Path:
    """A model file from "menagerie:<path>" or "i2rt:<path>"."""
    source, _, rel = ref.partition(":")
    path = (menagerie_dir() if source == "menagerie" else i2rt_dir()) / rel
    if not path.is_file():
        raise AssetsMissing(f"robot model not found: {path}. Run scripts/fetch_assets.sh")
    return path


@dataclass(frozen=True)
class Robot:
    name: str
    title: str  # for the prompt and the page, e.g. "Franka Panda"
    arm: str  # the arm MJCF, as "menagerie:<path>" or "i2rt:<path>"
    root: str  # the top body of the gripper: the gripper is this body and all bodies below it
    left: str  # the finger bodies, with the bodies below them
    right: str
    left_geoms: str = ""  # a name start: only these geoms of the finger. For a fixed jaw on the wrist body.
    right_geoms: str = ""
    gripper: str = ""  # a separate gripper MJCF, attached at `mount`. Its names get GRIPPER_PREFIX.
    mount: str = ""  # a site of the arm, or else a body
    base: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_yaw: float = 0.0  # degrees about world z: an arm model that faces -y or -x turns to face +x
    ready: tuple[float, float, float] = (0.30, 0.0, 0.40)  # the start point of the gripper
    home: tuple[float, ...] = ()  # arm joint angles for the IK start; empty: keyframe 0, or all 0
    tcp_body: str = ""  # the body of the tcp site; empty: root
    tcp: tuple[float, float, float] | None = None  # the tcp in tcp_body; None: from the fingertips
    approach: tuple[float, float, float] | None = None  # the gripper axis in tcp_body; None: from the geometry
    kp: tuple[float, ...] = ()  # position servo gains for the arm joints, for an arm without actuators
    grip_kp: float = 0.0  # a position servo for the gripper joint `grip_joint`, for a gripper without one
    grip_joint: str = ""
    gravcomp: bool = True  # gravity compensation, as on most real arms: weak position servos then hold still
    width_joints: tuple[str, ...] = ()  # finger slide joints whose sum is the width; empty: measure the gap
    max_tilt: float = 0.0  # degrees: the gripper may tilt this far from straight down where it cannot point down
    pads: bool = False  # flat box pads on the inner finger faces near the tips, in place of the finger meshes.
    # MuJoCo collides a mesh as its convex hull, and the hull of an L-shaped finger is a wedge that pushes
    # a held object out.
    agent: dict | None = None  # the gripper shape of the plan view; None: the bounds of the collision geoms
    ik_from_home: bool = False  # the IK start is home, which points the gripper down. Else IK finds a start.
    self_contact: bool = False  # the robot parts touch each other. Off: an attached gripper does not push on
    # the last link of the arm, and the robot still touches the scene.
    note: str = ""  # a few words for the page

    @property
    def tcp_parent(self) -> str:
        return self.tcp_body or self.root


def copy_options(src: mujoco.MjSpec, dst: mujoco.MjSpec) -> None:
    """Give dst the physics options of src. Attach keeps the options of the parent and warns about each one
    that differs."""
    for name in _OPTIONS:
        setattr(dst.option, name, getattr(src.option, name))


def _gripper_spec(robot: Robot, arm: mujoco.MjSpec) -> None:
    """Attach the gripper. The arm takes the contact options of the gripper: a Robotiq grip needs its
    elliptic cone and high impratio."""
    g = mujoco.MjSpec.from_file(str(_file(robot.gripper)))
    arm.option.cone, arm.option.impratio = g.option.cone, g.option.impratio
    copy_options(arm, g)
    site = arm.site(robot.mount)
    if site is not None:
        arm.attach(g, site=site, prefix=GRIPPER_PREFIX)
    else:
        arm.attach(g, frame=arm.body(robot.mount).add_frame(), prefix=GRIPPER_PREFIX)


def _bodies_below(model: mujoco.MjModel, top: int) -> list[int]:
    out = []
    for b in range(model.nbody):
        p = b
        while p > 0 and p != top:
            p = model.body_parentid[p]
        if p == top:
            out.append(b)
    return out


def _chain_joints(model: mujoco.MjModel, body: int) -> list[int]:
    """The hinge joints from the world down to `body`, top first."""
    out = []
    b = body
    while b > 0:
        out += [j for j in range(model.njnt) if model.jnt_bodyid[j] == b
                and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE][::-1]
        b = model.body_parentid[b]
    return out[::-1]


def _spec(robot: Robot) -> mujoco.MjSpec:
    """The robot alone: the arm, the gripper, the servos that it lacks and the base position. No tcp site."""
    spec = mujoco.MjSpec.from_file(str(_file(robot.arm)))
    spec.option.timestep, spec.option.iterations, spec.option.ls_iterations = TIMESTEP, ITERATIONS, LS_ITERATIONS
    if robot.gripper:
        _gripper_spec(robot, spec)
    if not robot.self_contact:
        for g in spec.geoms:
            if g.contype or g.conaffinity:
                g.contype, g.conaffinity = ROBOT_CONTYPE, ROBOT_CONAFFINITY
    if robot.base != (0.0, 0.0, 0.0) or robot.base_yaw:
        half = np.radians(robot.base_yaw) / 2
        turn = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, turn)
        for b in spec.worldbody.bodies:
            b.pos = rot.reshape(3, 3) @ np.asarray(b.pos) + np.asarray(robot.base)
            quat = np.empty(4)
            mujoco.mju_mulQuat(quat, turn, np.asarray(b.quat, dtype=float))
            b.quat = quat
    if robot.kp or robot.grip_kp or robot.gravcomp:
        model = spec.compile()
        joints = [model.joint(j).name for j in _chain_joints(model, model.body(robot.root).id)]
        for name, kp in zip(joints, robot.kp, strict=False):
            _servo(spec, name, kp)
        if robot.grip_kp:
            _servo(spec, robot.grip_joint, robot.grip_kp)
        if robot.gravcomp:
            for b in spec.bodies[1:]:
                b.gravcomp = 1.0
    return spec


def _servo(spec: mujoco.MjSpec, joint: str, kp: float) -> None:
    j = spec.joint(joint)
    a = spec.add_actuator(name=joint, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
    a.set_to_position(kp=kp, kv=kp / 16)
    a.ctrlrange = list(j.range)
    a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    j.armature = max(float(np.max(j.armature)), 0.01)
    j.damping = np.maximum(np.asarray(j.damping, dtype=float), 0.1)


@dataclass(frozen=True)
class Parts:
    """What the code finds in the compiled robot."""
    arm_qpos: np.ndarray  # the qpos index of each arm joint
    arm_dof: np.ndarray  # the dof index of each arm joint
    arm_ctrl: np.ndarray  # the actuator of each arm joint
    grip_ctrl: int  # the gripper actuator
    grip_open: float  # its control when open and when closed
    grip_closed: float
    home: np.ndarray  # arm joint angles
    lo: np.ndarray  # arm joint limits
    hi: np.ndarray
    tcp_site: int
    gripper_bodies: frozenset[int]
    left_geoms: tuple[int, ...]
    right_geoms: tuple[int, ...]
    width_qpos: tuple[int, ...]
    left_points: tuple[tuple[int, np.ndarray], ...]  # (body, vertices in the body frame) of each finger geom
    right_points: tuple[tuple[int, np.ndarray], ...]
    open_width: float
    closed_width: float = 0.0  # the finger gap of the gripper closed on nothing
    agent: dict = field(default_factory=dict)  # the gripper shape of the plan view

    @property
    def dof(self) -> int:
        return len(self.arm_qpos)


def _collision(model: mujoco.MjModel, bodies) -> tuple[int, ...]:
    bodies = set(bodies)
    return tuple(g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies
                 and (model.geom_contype[g] or model.geom_conaffinity[g]))


def finger_geoms(robot: Robot, model: mujoco.MjModel, side: str) -> tuple[int, ...]:
    """The collision geoms of the left or the right finger."""
    body, start = (robot.left, robot.left_geoms) if side == "left" else (robot.right, robot.right_geoms)
    geoms = _collision(model, _bodies_below(model, model.body(body).id))
    return tuple(g for g in geoms if model.geom(g).name.startswith(start)) if start else geoms


def _world_points(model, data, geoms) -> np.ndarray:
    from plan_and_act.sim.meshes import geom_points
    out = []
    for g in geoms:
        b = model.geom_bodyid[g]
        out.append(geom_points(model, g) @ data.xmat[b].reshape(3, 3).T + data.xpos[b])
    return np.vstack(out)


def _gap(left: np.ndarray, right: np.ndarray, y: np.ndarray, z: np.ndarray, tip: float) -> tuple[float, float]:
    """The inner faces of the fingers along the closing axis y, from the points near the tips: the left one
    and the right one. The points are relative to one origin, and tip is the tip along z."""
    def near(pts: np.ndarray) -> np.ndarray:
        along = pts @ z
        band = pts[(along > tip - TIP_MARGIN - PAD_BAND) & (along <= tip + 1e-9)]
        return band if len(band) else pts
    return float((near(left) @ y).min()), float((near(right) @ y).max())


def _grip_axes(robot: Robot, model, data, left, right) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                                   float]:
    """The closing axis y (right finger to left finger), the gripper axis z, the finger points (world) and the
    tip along z. From the vertices, not from the convex hulls: an L-shaped finger fills its hull."""
    body = model.body(robot.tcp_parent).id
    rot, pos = data.xmat[body].reshape(3, 3), data.xpos[body]
    pl, pr = _world_points(model, data, left), _world_points(model, data, right)
    y = pl.mean(axis=0) - pr.mean(axis=0)
    z = rot @ np.asarray(robot.approach, dtype=float) if robot.approach is not None else \
        np.vstack([pl, pr]).mean(axis=0) - pos
    y /= np.linalg.norm(y)
    z -= y * (z @ y)
    z /= np.linalg.norm(z)
    tip = float((np.vstack([pl, pr]) @ z).max())
    return y, z, pl, pr, tip


def _open_gap(robot: Robot, model, data, left, right) -> float:
    y, z, pl, pr, tip = _grip_axes(robot, model, data, left, right)
    lf, rf = _gap(pl, pr, y, z, tip)
    return max(0.0, lf - rf)


def _grip_actuator(model: mujoco.MjModel, arm_joints: set[int]) -> int:
    rest = [a for a in range(model.nu) if not (model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                                               and model.actuator_trnid[a][0] in arm_joints)]
    if len(rest) != 1:
        raise ValueError(f"expected one gripper actuator, found {len(rest)}")
    return rest[0]


def _settle(model, data, ctrl: dict[int, float], qpos: dict[int, float], seconds: float) -> None:
    """Hold the arm still at qpos, with gravity off, and let the gripper go to its control."""
    g = model.opt.gravity.copy()
    model.opt.gravity[:] = 0.0
    try:
        for _ in range(round(seconds / model.opt.timestep)):
            for adr, q in qpos.items():
                data.qpos[adr] = q
            for a, c in ctrl.items():
                data.ctrl[a] = c
            mujoco.mj_step(model, data)
    finally:
        model.opt.gravity[:] = g


def _tcp_frame(robot: Robot, model, data, left, right) -> tuple[np.ndarray, np.ndarray]:
    """The tcp point and orientation (a quaternion) in tcp_body, from the open fingers now. The point is
    between the inner faces of the fingers, TIP_MARGIN back from the tips."""
    body = model.body(robot.tcp_parent).id
    rot, pos = data.xmat[body].reshape(3, 3), data.xpos[body]
    y, z, pl, pr, tip = _grip_axes(robot, model, data, left, right)
    x = np.cross(y, z)
    if robot.tcp is not None:
        point = np.asarray(robot.tcp, dtype=float)
    else:
        lf, rf = _gap(pl, pr, y, z, tip)
        across = float((np.vstack([pl, pr]) @ x).mean())
        world = x * across + y * (lf + rf) / 2 + z * (tip - TIP_MARGIN)
        point = rot.T @ (world - pos)
    frame = rot.T @ np.column_stack([x, y, z])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, frame.ravel())
    return point, quat


def _agent(model, data, site: int, gripper: tuple[int, ...], fingers: dict[int, tuple[int, ...]]) -> dict:
    """The gripper as boxes in its own frame: the origin at the tcp, x along the closing fingers, z up. Each
    box is the bound of collision geoms of the open gripper."""
    rot = data.site_xmat[site].reshape(3, 3)
    axes = np.column_stack([rot[:, 1], rot[:, 0], -rot[:, 2]])  # agent x, y, z in world
    origin = data.site_xpos[site]

    def bounds(geoms) -> tuple[np.ndarray, np.ndarray] | None:
        if not geoms:
            return None
        pts = (_world_points(model, data, geoms) - origin) @ axes
        return pts.min(axis=0), pts.max(axis=0)

    in_fingers = {g for gs in fingers.values() for g in gs}
    parts = []
    palm = bounds([g for g in gripper if g not in in_fingers])
    if palm is not None:
        lo, hi = palm
        lo[2] = max(lo[2], 0.0)
        parts.append({"name": "hand", "shape": "box", "size": _round(hi - lo), "xyz": _round((hi + lo) / 2),
                      "slide": 0})
    for side, sign in (("left", 1), ("right", -1)):
        b = bounds(fingers[sign])
        if b is None:
            continue
        lo, hi = b
        size, mid = hi - lo, (hi + lo) / 2
        # At a gap of 0 the inner face is at x 0. The plan view slides each finger out by half the gap.
        parts.append({"name": f"finger_{side}", "shape": "box", "size": _round(size),
                      "xyz": _round([sign * size[0] / 2, mid[1], mid[2]]), "slide": sign})
    return {"parts": parts, "slide_axis": [1.0, 0.0, 0.0], "color": [0.86, 0.87, 0.9, 1.0]}


def _round(v) -> list[float]:
    return [round(float(x), 4) for x in v]


def robot_spec(robot: Robot) -> mujoco.MjSpec:
    """The robot with its tcp site, ready to take a scene."""
    spec = _spec(robot)
    point, quat, grip_open, _, pads = _measured(robot)
    for body, pos, pquat, size in pads:  # the pads replace the finger collision meshes
        for b in [spec.body(body), *spec.body(body).find_all(mujoco.mjtObj.mjOBJ_BODY)]:
            for g in b.geoms:
                g.contype, g.conaffinity = 0, 0
        spec.body(body).add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, quat=pquat, size=size,
                                 contype=ROBOT_CONTYPE, conaffinity=ROBOT_CONAFFINITY, friction=[1.5, 0.01, 0.001],
                                 rgba=[0.2, 0.2, 0.2, 1.0], group=3)
    spec.body(robot.tcp_parent).add_site(name=TCP_SITE, pos=list(point), quat=list(quat), size=[0.005] * 3,
                                         group=5)
    if not spec.keys:  # the world resets to keyframe 0: the arm at home and the gripper open
        model = spec.compile()
        joints = _chain_joints(model, model.body(robot.root).id)
        qpos = model.qpos0.copy()
        qpos[model.jnt_qposadr[joints]] = _home(robot, model)
        ctrl = np.zeros(model.nu)
        for a in range(model.nu):
            if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and model.actuator_trnid[a][0] in joints:
                ctrl[a] = qpos[model.jnt_qposadr[model.actuator_trnid[a][0]]]
        ctrl[_grip_actuator(model, set(joints))] = grip_open
        spec.add_key(name="home", qpos=qpos.tolist(), ctrl=ctrl.tolist())
    return spec


_MEASURED: dict[str, tuple] = {}
_LOCK = threading.Lock()


def _measured(robot: Robot) -> tuple:
    """The tcp point and orientation, the gripper controls when open and closed, and the pads. Found once for
    each robot."""
    with _LOCK:
        if robot.name not in _MEASURED:
            _MEASURED[robot.name] = _measure(robot)
        return _MEASURED[robot.name]


def _measure(robot: Robot) -> tuple:
    model = _spec(robot).compile()
    data = mujoco.MjData(model)
    joints = _chain_joints(model, model.body(robot.root).id)
    grip = _grip_actuator(model, set(joints))
    home = _home(robot, model)
    qpos = {int(model.jnt_qposadr[j]): float(q) for j, q in zip(joints, home, strict=True)}
    left, right = finger_geoms(robot, model, "left"), finger_geoms(robot, model, "right")
    widths = {}
    lo, hi = (float(v) for v in model.actuator_ctrlrange[grip])
    for c in (lo, hi):
        mujoco.mj_resetData(model, data)
        _settle(model, data, {grip: c}, qpos, OPEN_S)
        mujoco.mj_forward(model, data)
        widths[c] = _open_gap(robot, model, data, left, right)
    grip_open, grip_closed = (hi, lo) if widths[hi] >= widths[lo] else (lo, hi)
    mujoco.mj_resetData(model, data)
    _settle(model, data, {grip: grip_open}, qpos, OPEN_S)
    mujoco.mj_forward(model, data)
    point, quat = _tcp_frame(robot, model, data, left, right)
    return point, quat, grip_open, grip_closed, _pads(robot, model, data, left, right) if robot.pads else ()


def _pads(robot: Robot, model, data, left, right) -> tuple[tuple[str, list, list, list], ...]:
    """A box pad on the inner face of each open finger near its tip: (body, pos, quat, half size)."""
    y, z, pl, pr, tip = _grip_axes(robot, model, data, left, right)
    x = np.cross(y, z)
    lf, rf = _gap(pl, pr, y, z, tip)
    out = []
    for body, face, sign, pts in ((robot.left, lf, 1.0, pl), (robot.right, rf, -1.0, pr)):
        along = pts @ z
        band = pts[along > tip - TIP_MARGIN - PAD_BAND]
        width = min(float(np.ptp(band @ x)) / 2, PAD_HALF[0])
        centre = (x * float((band @ x).mean()) + y * (face + sign * PAD_HALF[1])
                  + z * (tip - PAD_HALF[2]))
        b = model.body(body).id
        rot = data.xmat[b].reshape(3, 3)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, (rot.T @ np.column_stack([x, y, z])).ravel())
        out.append((body, list(rot.T @ (centre - data.xpos[b])), list(quat), [width, PAD_HALF[1], PAD_HALF[2]]))
    return tuple(out)


def _home(robot: Robot, model: mujoco.MjModel) -> np.ndarray:
    joints = _chain_joints(model, model.body(robot.root).id)
    if robot.home:
        return np.asarray(robot.home, dtype=float)
    if model.nkey:
        return np.array([model.key_qpos[0][model.jnt_qposadr[j]] for j in joints])
    return np.zeros(len(joints))


def parts(robot: Robot, model: mujoco.MjModel) -> Parts:
    """The parts of the robot in a compiled scene model."""
    joints = _chain_joints(model, model.body(robot.root).id)
    js = set(joints)
    arm_ctrl = []
    for j in joints:
        act = [a for a in range(model.nu) if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
               and model.actuator_trnid[a][0] == j]
        if len(act) != 1:
            raise ValueError(f"{robot.name}: joint {model.joint(j).name} has {len(act)} actuators")
        arm_ctrl.append(act[0])
    limited = model.jnt_limited[joints].astype(bool)
    lo = np.where(limited, model.jnt_range[joints, 0], -np.inf)
    hi = np.where(limited, model.jnt_range[joints, 1], np.inf)
    root = model.body(robot.root).id
    _, _, grip_open, grip_closed, _ = _measured(robot)
    width_qpos = tuple(int(model.jnt_qposadr[model.joint(n).id]) for n in robot.width_joints)
    left, right = finger_geoms(robot, model, "left"), finger_geoms(robot, model, "right")
    p = Parts(np.array([model.jnt_qposadr[j] for j in joints]), np.array([model.jnt_dofadr[j] for j in joints]),
              np.array(arm_ctrl), _grip_actuator(model, js), grip_open, grip_closed, _home(robot, model), lo, hi,
              model.site(TCP_SITE).id, frozenset(_bodies_below(model, root)), left, right, width_qpos,
              _local_points(model, left), _local_points(model, right), 0.0)
    data = mujoco.MjData(model)
    qpos = {int(a): float(q) for a, q in zip(p.arm_qpos, p.home, strict=True)}
    _settle(model, data, {p.grip_ctrl: grip_open, **dict(zip(p.arm_ctrl.tolist(), p.home, strict=True))}, qpos,
            OPEN_S)
    mujoco.mj_forward(model, data)
    width = width_of(p, model, data)
    shut = mujoco.MjData(model)
    _settle(model, shut, {p.grip_ctrl: grip_closed, **dict(zip(p.arm_ctrl.tolist(), p.home, strict=True))}, qpos,
            OPEN_S)
    mujoco.mj_forward(model, shut)
    closed = width_of(p, model, shut)
    agent = robot.agent or _agent(model, data, p.tcp_site, _collision(model, p.gripper_bodies),
                                  {1: p.left_geoms, -1: p.right_geoms})
    return Parts(**{**p.__dict__, "open_width": round(width, 4), "closed_width": round(closed, 4), "agent": agent})


def _local_points(model: mujoco.MjModel, geoms) -> tuple[tuple[int, np.ndarray], ...]:
    from plan_and_act.sim.meshes import geom_points
    return tuple((int(model.geom_bodyid[g]), geom_points(model, g)) for g in geoms)


def width_of(p: Parts, model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """The finger gap: the sum of the finger slides, or else the gap along the closing axis between the finger
    points near the tips. Not the distance of the convex hulls: an L-shaped finger fills its hull."""
    if p.width_qpos:
        return float(sum(data.qpos[a] for a in p.width_qpos))
    rot, origin = data.site_xmat[p.tcp_site].reshape(3, 3), data.site_xpos[p.tcp_site]

    def side(points) -> np.ndarray:
        pts = np.vstack([local @ data.xmat[b].reshape(3, 3).T + data.xpos[b] for b, local in points]) - origin
        along = pts @ rot[:, 2]
        near = pts[(along > -PAD_BAND) & (along < TIP_MARGIN + 0.005)]
        return (near if len(near) else pts) @ rot[:, 1]

    return max(0.0, float(side(p.left_points).min() - side(p.right_points).max()))


# The Panda gripper as boxes, measured from the model: the fingertips reach 1 cm below the tcp.
_PANDA_FINGER, _PANDA_FINGER_Z = [0.027, 0.021, 0.057], 0.0185
PANDA_AGENT = {"parts": [{"name": "hand", "shape": "box", "size": [0.20, 0.058, 0.075], "xyz": [0.0, 0.0, 0.0825],
                          "slide": 0},
                         *({"name": f"finger_{side}", "shape": "box", "size": _PANDA_FINGER,
                            "xyz": [sign * _PANDA_FINGER[0] / 2, 0.0, _PANDA_FINGER_Z], "slide": sign}
                           for side, sign in (("left", 1), ("right", -1)))],
               "slide_axis": [1.0, 0.0, 0.0], "color": [0.86, 0.87, 0.9, 1.0]}

# A Robotiq 2F-85 on an arm without a gripper.
_ROBOTIQ = {"gripper": "menagerie:robotiq_2f85/2f85.xml", "mount": "attachment_site", "root": "gripper/base_mount",
            "left": "gripper/left_follower", "right": "gripper/right_follower"}

# A short 6-joint arm with a small wrist range: the base closer to the table, a low start, and a tilt.
_LOW = {"base": (0.1, 0.0, 0.0), "ready": (0.30, 0.0, 0.25), "max_tilt": 45.0}

ROBOTS: dict[str, Robot] = {r.name: r for r in (
    Robot("panda", "Franka Panda", "menagerie:franka_emika_panda/panda.xml", "hand", "left_finger",
          "right_finger", tcp=(0.0, 0.0, 0.1034), approach=(0.0, 0.0, 1.0),
          width_joints=("finger_joint1", "finger_joint2"), agent=PANDA_AGENT, ik_from_home=True, gravcomp=False,
          self_contact=True, note="7 joints, parallel gripper"),
    Robot("fr3", "Franka FR3", "menagerie:franka_fr3/fr3.xml", **_ROBOTIQ, note="7 joints, Robotiq 2F-85"),
    Robot("ur5e", "Universal Robots UR5e", "menagerie:universal_robots_ur5e/ur5e.xml", **_ROBOTIQ,
          note="6 joints, Robotiq 2F-85"),
    Robot("ur10e", "Universal Robots UR10e", "menagerie:universal_robots_ur10e/ur10e.xml", **_ROBOTIQ,
          note="6 joints, long reach, Robotiq 2F-85"),
    Robot("gen3", "Kinova Gen3", "menagerie:kinova_gen3/gen3.xml", **{**_ROBOTIQ, "mount": "pinch_site"},
          note="7 joints, Robotiq 2F-85"),
    Robot("iiwa14", "KUKA LBR iiwa 14", "menagerie:kuka_iiwa_14/iiwa14.xml", **_ROBOTIQ,
          note="7 joints, Robotiq 2F-85"),
    Robot("sawyer", "Rethink Sawyer", "menagerie:rethink_robotics_sawyer/sawyer.xml", **_ROBOTIQ,
          note="7 joints, Robotiq 2F-85"),
    Robot("rizon4", "Flexiv Rizon 4", "menagerie:flexiv_rizon4/flexiv_rizon4.xml", **{**_ROBOTIQ, "mount": "link7"},
          note="7 joints, Robotiq 2F-85"),
    Robot("omy", "ROBOTIS OMY", "menagerie:robotis_omy_3m/omy_3m.xml", **_ROBOTIQ, base=(0.1, 0.0, 0.0),
          ready=(0.30, 0.0, 0.30), max_tilt=45.0, note="6 joints, Robotiq 2F-85, tilts"),
    Robot("xarm7", "UFACTORY xArm 7", "menagerie:ufactory_xarm7/xarm7.xml", "xarm_gripper_base_link",
          "left_finger", "right_finger", note="7 joints, linkage gripper"),
    Robot("lite6", "UFACTORY Lite 6", "menagerie:ufactory_lite6/lite6_gripper_narrow.xml", "gripper_body",
          "gripper_left_finger", "gripper_right_finger", **_LOW, note="6 joints, gripper opens 1.8 cm"),
    Robot("piper", "AgileX Piper", "menagerie:agilex_piper/piper.xml", "link6", "link7", "link8",
          ready=(0.30, 0.0, 0.30), max_tilt=45.0, note="6 joints, parallel gripper, tilts"),
    Robot("arx_l5", "ARX L5", "menagerie:arx_l5/arx_l5.xml", "link6", "link7", "link8", **_LOW,
          note="6 joints, parallel gripper, tilts"),
    Robot("yam", "I2RT YAM", "menagerie:i2rt_yam/yam.xml", "link_6", "link_left_finger", "link_right_finger",
          **_LOW, note="6 joints, parallel gripper, tilts"),
    Robot("yam_ultra", "I2RT YAM Ultra", "i2rt:i2rt/robot_models/arm/yam_ultra/v2/yam_ultra.xml", "gripper",
          "gripper/tip_left", "gripper/tip_right",
          gripper="i2rt:i2rt/robot_models/gripper/linear_4310/linear_4310.xml", mount="gripper",
          kp=(80.0, 80.0, 80.0, 40.0, 10.0, 10.0), grip_kp=300.0, grip_joint="gripper/joint7", pads=True,
          **_LOW, note="v2, 6 joints, linear_4310 gripper, tilts"),
    Robot("vx300s", "Trossen ViperX 300 S", "menagerie:trossen_vx300s/vx300s.xml", "gripper_link",
          "left_finger_link", "right_finger_link", ready=(0.30, 0.0, 0.35), max_tilt=45.0,
          note="6 joints, parallel gripper, tilts"),
    Robot("wx250s", "Trossen WidowX 250 S", "menagerie:trossen_wx250s/wx250s.xml", "wx250s/gripper_link",
          "wx250s/left_finger_link", "wx250s/right_finger_link", **_LOW, note="6 joints, parallel gripper, tilts"),
    Robot("wxai", "Trossen WidowX AI", "menagerie:trossen_wxai/wxai_follower.xml", "link_6", "carriage_left",
          "carriage_right", **_LOW, note="6 joints, parallel gripper, tilts"),
)}
