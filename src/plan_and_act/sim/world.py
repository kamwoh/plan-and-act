"""MuJoCo world: a robot arm, a table and the objects and fixed items of a scene."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from string import Template

import mujoco
import numpy as np

from plan_and_act.scenes.base import Layout, ObjSpec, Scene
from plan_and_act.sim import meshes
from plan_and_act.sim.robots import ROBOTS, ROT0, Robot, copy_options, parts, robot_spec, width_of
from plan_and_act.types import PANDA_FACTS, ArmFacts

TEMPLATE = Path(__file__).with_name("scene.xml")
CAMERAS = {  # name: (pos, xyaxes). A scene can replace one.
    "front": ("1.45 0 0.80", "0 1 0 -0.55 0 0.835"),
    "top": ("0.5 0 1.2", "0 1 0 -1 0 0"),
}
SHUT_GAP = 0.003  # m: a closed gripper this close to its closed-on-nothing gap holds nothing
SPAWN_GAP = 0.002  # a mesh object starts this far above the table, so that it does not start inside it


@dataclass(frozen=True)
class ObjInfo:
    """The measured size of an object, in its own frame."""
    width: float  # along x
    depth: float  # along y
    height: float
    bottom: float  # the lowest point, below the body origin
    narrow: str  # "x" or "y": the narrow horizontal axis


def _quat(yaw_deg: float) -> list[float]:
    half = np.radians(yaw_deg) / 2
    return [float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]


def _mesh_object_xml(o: ObjSpec, i: int) -> str:
    """A scanned object. Its geoms move so that the body origin is the centre of its box."""
    info = meshes.measure(o.mesh)
    density = 1000.0 * o.mass / info.mass
    return (f'    <body name="{o.name}" pos="0.5 {-0.3 + 0.1 * i:.2f} {info.dims[2] / 2 + SPAWN_GAP:g}">\n'
            f'      <freejoint name="{o.name}"/>\n'
            f'{meshes.geoms_xml(o.name, o.mesh, info.centre, density)}\n'
            f'    </body>')


def _object_xml(o: ObjSpec, i: int) -> str:
    if o.shape == "mesh":
        return _mesh_object_xml(o, i)
    size = " ".join(f"{v:g}" for v in o.size)
    return (f'    <body name="{o.name}" pos="0.5 {-0.3 + 0.1 * i:.2f} {o.half_height:g}">\n'
            f'      <freejoint name="{o.name}"/>\n'
            f'      <geom name="{o.name}" type="{o.shape}" size="{size}" rgba="{o.rgba}" mass="{o.mass:g}"\n'
            f'            friction="1.5 0.01 0.001"/>\n'
            f'    </body>')


def scene_xml(scene: Scene) -> str:
    objects = "\n".join(_object_xml(o, i) for i, o in enumerate(scene.objects))
    cameras = "\n".join(f'    <camera name="{name}" pos="{pos}" xyaxes="{axes}" fovy="45"/>'
                        for name, (pos, axes) in {**CAMERAS, **scene.camera_overrides}.items())
    mesh_assets = "\n".join(meshes.asset_xml(o.name, o.mesh) for o in scene.objects if o.shape == "mesh")
    assets = f"  <asset>\n{mesh_assets}\n  </asset>" if mesh_assets else ""
    return Template(TEMPLATE.read_text()).substitute(fixtures=scene.fixtures_xml, objects=objects, cameras=cameras,
                                                     assets=assets)


_MODELS: dict[tuple[str, str], tuple[mujoco.MjModel, object]] = {}  # (robot, scene XML) -> model, parts
_MODELS_LOCK = threading.Lock()


def robot_of(scene: Scene) -> Robot:
    return ROBOTS[scene.robot]


def arm_facts(scene: Scene) -> ArmFacts:
    """What the prompts say about the robot of the scene. The Panda needs no model."""
    if scene.robot == "panda":
        return PANDA_FACTS
    robot, (_, p) = robot_of(scene), _compiled(scene)
    return ArmFacts(robot.title, (robot.base[0], robot.base[1]), p.open_width, robot.ready, robot.max_tilt)


def _build(scene: Scene, xml: str) -> mujoco.MjModel:
    """The robot spec with the scene attached to it, so that the robot bodies come first."""
    robot = robot_spec(robot_of(scene))
    world = mujoco.MjSpec.from_string(xml)
    copy_options(robot, world)
    robot.attach(world, frame=robot.worldbody.add_frame(), prefix="")
    head, glob = robot.visual.headlight, robot.visual.global_
    head.diffuse, head.ambient, head.specular = (world.visual.headlight.diffuse, world.visual.headlight.ambient,
                                                world.visual.headlight.specular)
    glob.offwidth, glob.offheight = world.visual.global_.offwidth, world.visual.global_.offheight
    return robot.compile()


def _compiled(scene: Scene):
    """The compiled model of a scene and its robot parts. Each one compiles once. The worlds share the model
    and never change it: each world has its own MjData."""
    xml = scene_xml(scene)
    with _MODELS_LOCK:
        if (scene.robot, xml) not in _MODELS:
            model = _build(scene, xml)
            _MODELS[scene.robot, xml] = (model, parts(robot_of(scene), model))
        return _MODELS[scene.robot, xml]


class World:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.robot = robot_of(scene)
        self.model, self.parts = _compiled(scene)
        self.data = mujoco.MjData(self.model)
        self.tcp_id = self.parts.tcp_site
        self._gripper = set(self.parts.gripper_bodies)
        self.events: list[str] = []  # brief happenings, such as a button press, since the reset
        self.monitors: list = []
        self.rot0 = ROT0  # the tcp orientation at yaw 0: the gripper points down
        if self.robot.ik_from_home:  # the home pose points down: use it as measured, as the old Panda code did
            home = mujoco.MjData(self.model)
            mujoco.mj_resetDataKeyframe(self.model, home, 0)
            mujoco.mj_kinematics(self.model, home)
            self.rot0 = home.site_xmat[self.tcp_id].reshape(3, 3).copy()
        self._info = {o.name: self._measure(o) for o in scene.objects}
        # The collision points of each object in its body frame, to find its lowest point.
        self._points = {o.name: np.vstack([meshes.geom_points(self.model, g) for g in
                                           meshes.collision_geoms(self.model, self.model.body(o.name).id)])
                        for o in scene.objects if o.shape == "mesh"}

    def _measure(self, o: ObjSpec) -> ObjInfo:
        if o.shape == "mesh":
            lo, hi = meshes.body_bounds(self.model, self.model.body(o.name).id)
            w, d, h = (float(v) for v in hi - lo)
            bottom = float(lo[2])
        else:
            w, d, h = o.dims
            bottom = -o.half_height
        return ObjInfo(w, d, h, bottom, "x" if w <= d else "y")

    @property
    def objects(self) -> tuple[ObjSpec, ...]:
        return self.scene.objects

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def time(self) -> float:
        return float(self.data.time)

    def reset(self, layout: Layout, joint_qpos: dict[str, float] | None = None) -> None:
        """Objects at the layout, scene joints at the scene's start values, changed by joint_qpos."""
        missing = sorted({o.name for o in self.objects} - set(layout))
        if missing:
            raise ValueError(f"layout misses {missing}")
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        for o in self.objects:
            adr = self.model.jnt_qposadr[self.model.joint(o.name).id]
            x, y, *yaw = layout[o.name]  # (x, y) or (x, y, yaw)
            z = -self._info[o.name].bottom + (SPAWN_GAP if o.shape == "mesh" else 0.0)
            self.data.qpos[adr:adr + 7] = [x, y, z, *_quat(yaw[0] if yaw else 0.0)]
        for name, q in {**self.scene.initial_qpos, **(joint_qpos or {})}.items():
            self.data.qpos[self.model.jnt_qposadr[self.model.joint(name).id]] = q
        mujoco.mj_forward(self.model, self.data)
        self.events = []
        self.monitors = [make() for make in self.scene.monitors]  # fresh monitors are armed

    def resume(self, events: list[str]) -> None:
        """After a loaded state: keep its events, and let each fresh monitor see the state once. A button that
        is down now then does not give a second press event."""
        self.events = list(events)
        for m in self.monitors:
            m(self)

    def step(self, n: int = 1) -> None:
        if not self.monitors:
            mujoco.mj_step(self.model, self.data, nstep=n)
            return
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            self.events += [e for m in self.monitors if (e := m(self)) is not None]

    def joint_pos(self, name: str) -> float:
        return float(self.data.qpos[self.model.jnt_qposadr[self.model.joint(name).id]])

    def joints(self) -> dict[str, float]:
        """The scene joints that the scene reports, by name."""
        return {n: self.joint_pos(n) for n in self.scene.joints}

    def tcp(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp_id].copy()

    def gripper_yaw(self) -> float:
        """The turn of the gripper about world +z from the yaw 0 orientation, in degrees."""
        turn = self.data.site_xmat[self.tcp_id].reshape(3, 3) @ self.rot0.T
        return float(np.degrees(np.arctan2(turn[1, 0], turn[0, 0])))

    def gripper_width(self) -> float:
        """The finger gap. A gripper whose fingers cannot meet, closed on nothing, gives 0, as a gripper whose
        fingers meet does. Then "closed on nothing" means the same for every arm."""
        gap = width_of(self.parts, self.model, self.data)
        shut = self.parts.closed_width
        return 0.0 if shut > SHUT_GAP and gap <= shut + SHUT_GAP else gap

    def gripper_closed(self) -> bool:
        """True when the last gripper command was a close."""
        return bool(self.data.ctrl[self.parts.grip_ctrl] == self.parts.grip_closed)

    def object_pos(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    def object_info(self, name: str) -> ObjInfo:
        return self._info[name]

    def object_yaw(self, name: str) -> float:
        """The turn of the object about world +z, in degrees, in (-180, 180]."""
        rot = self.data.body(name).xmat.reshape(3, 3)
        yaw = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))
        return 180.0 if yaw == -180.0 else yaw

    def object_bottom(self, name: str) -> float:
        """The height of the lowest point of the object. For a box or a cylinder: its centre minus half its height."""
        body = self.data.body(name)
        if name not in self._points:
            return float(body.xpos[2] - self._info[name].height / 2)
        return float((self._points[name] @ body.xmat.reshape(3, 3).T)[:, 2].min() + body.xpos[2])

    def gripper_contacts(self) -> list[str]:
        body = self.model.geom_bodyid
        names: set[str] = set()
        for g1, g2 in zip(self.data.contact.geom1, self.data.contact.geom2, strict=True):
            for mine, other in ((body[g1], body[g2]), (body[g2], body[g1])):
                if mine in self._gripper and other not in self._gripper:
                    names.add("table" if other == 0 else self.model.body(other).name)
        return sorted(names)
