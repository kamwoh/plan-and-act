"""MuJoCo world: the Panda arm, a table and three cubes."""
from __future__ import annotations

import shutil
from pathlib import Path

import mujoco
import numpy as np

from armmind.sim.assets import panda_dir

SCENE = Path(__file__).with_name("scene.xml")
CUBES = ("red_cube", "blue_cube", "green_cube")
CUBE_HALF = 0.02
TCP_OFFSET = np.array([0.0, 0.0, 0.1034])  # from the hand frame to the point between the fingertips
GRIPPER_BODIES = ("hand", "left_finger", "right_finger")

Layout = dict[str, tuple[float, float]]


class World:
    def __init__(self, model_dir: Path | None = None):
        folder = model_dir or panda_dir()
        scene = folder / "armmind_scene.xml"
        shutil.copyfile(SCENE, scene)  # MJCF resolves <include> next to the scene file
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.hand_id = self.model.body("hand").id
        self._gripper = {self.model.body(n).id for n in GRIPPER_BODIES}

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def time(self) -> float:
        return float(self.data.time)

    def reset(self, layout: Layout) -> None:
        missing = sorted(set(CUBES) - set(layout))
        if missing:
            raise ValueError(f"layout misses {missing}")
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        for name in CUBES:
            adr = self.model.jnt_qposadr[self.model.joint(name).id]
            x, y = layout[name]
            self.data.qpos[adr:adr + 7] = [x, y, CUBE_HALF, 1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1) -> None:
        mujoco.mj_step(self.model, self.data, nstep=n)

    def tcp(self) -> np.ndarray:
        rot = self.data.xmat[self.hand_id].reshape(3, 3)
        return self.data.xpos[self.hand_id] + rot @ TCP_OFFSET

    def gripper_width(self) -> float:
        return float(self.data.qpos[7] + self.data.qpos[8])

    def cube_pos(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    def gripper_contacts(self) -> list[str]:
        body = self.model.geom_bodyid
        names: set[str] = set()
        for g1, g2 in zip(self.data.contact.geom1, self.data.contact.geom2, strict=True):
            for mine, other in ((body[g1], body[g2]), (body[g2], body[g1])):
                if mine in self._gripper and other not in self._gripper:
                    names.add("table" if other == 0 else self.model.body(other).name)
        return sorted(names)
