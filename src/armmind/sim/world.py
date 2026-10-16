"""MuJoCo world: the Panda arm, a table and the objects and fixed items of a scene."""
from __future__ import annotations

import os
from pathlib import Path
from string import Template

import mujoco
import numpy as np

from armmind.scenes.base import Layout, ObjSpec, Scene
from armmind.sim.assets import panda_dir

TEMPLATE = Path(__file__).with_name("scene.xml")
TCP_OFFSET = np.array([0.0, 0.0, 0.1034])  # from the hand frame to the point between the fingertips
GRIPPER_BODIES = ("hand", "left_finger", "right_finger")


def _object_xml(o: ObjSpec, i: int) -> str:
    size = " ".join(f"{v:g}" for v in o.size)
    return (f'    <body name="{o.name}" pos="0.5 {-0.3 + 0.1 * i:.2f} {o.half_height:g}">\n'
            f'      <freejoint name="{o.name}"/>\n'
            f'      <geom name="{o.name}" type="{o.shape}" size="{size}" rgba="{o.rgba}" mass="{o.mass:g}"\n'
            f'            friction="1.5 0.01 0.001"/>\n'
            f'    </body>')


def scene_xml(scene: Scene) -> str:
    objects = "\n".join(_object_xml(o, i) for i, o in enumerate(scene.objects))
    return Template(TEMPLATE.read_text()).substitute(fixtures=scene.fixtures_xml, objects=objects)


class World:
    def __init__(self, scene: Scene, model_dir: Path | None = None):
        self.scene = scene
        folder = model_dir or panda_dir()
        self.scene_file = folder / f"armmind_scene_{scene.name}.xml"  # MJCF resolves <include> next to it
        tmp = folder / f".armmind_scene_{scene.name}.{os.getpid()}.xml"
        tmp.write_text(scene_xml(scene))
        os.replace(tmp, self.scene_file)  # a whole file at all times, also for another process
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_file))
        self.data = mujoco.MjData(self.model)
        self.hand_id = self.model.body("hand").id
        self._gripper = {self.model.body(n).id for n in GRIPPER_BODIES}
        home = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, home, 0)
        mujoco.mj_kinematics(self.model, home)
        self.rot0 = home.xmat[self.hand_id].reshape(3, 3).copy()  # the home orientation: yaw 0, points down

    @property
    def objects(self) -> tuple[ObjSpec, ...]:
        return self.scene.objects

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def time(self) -> float:
        return float(self.data.time)

    def reset(self, layout: Layout) -> None:
        missing = sorted({o.name for o in self.objects} - set(layout))
        if missing:
            raise ValueError(f"layout misses {missing}")
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        for o in self.objects:
            adr = self.model.jnt_qposadr[self.model.joint(o.name).id]
            x, y = layout[o.name]
            self.data.qpos[adr:adr + 7] = [x, y, o.half_height, 1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1) -> None:
        mujoco.mj_step(self.model, self.data, nstep=n)

    def tcp(self) -> np.ndarray:
        rot = self.data.xmat[self.hand_id].reshape(3, 3)
        return self.data.xpos[self.hand_id] + rot @ TCP_OFFSET

    def gripper_yaw(self) -> float:
        """The turn of the hand about world +z from the home orientation, in degrees."""
        turn = self.data.xmat[self.hand_id].reshape(3, 3) @ self.rot0.T
        return float(np.degrees(np.arctan2(turn[1, 0], turn[0, 0])))

    def gripper_width(self) -> float:
        return float(self.data.qpos[7] + self.data.qpos[8])

    def object_pos(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    def gripper_contacts(self) -> list[str]:
        body = self.model.geom_bodyid
        names: set[str] = set()
        for g1, g2 in zip(self.data.contact.geom1, self.data.contact.geom2, strict=True):
            for mine, other in ((body[g1], body[g2]), (body[g2], body[g1])):
                if mine in self._gripper and other not in self._gripper:
                    names.add("table" if other == 0 else self.model.body(other).name)
        return sorted(names)
