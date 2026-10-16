"""Scanned objects: where their files are, their MJCF parts, and their measured size.

scripts/fetch_objects.sh downloads them from kevinzakka/mujoco_scanned_objects. Each folder has a visual mesh,
a texture and convex collision pieces.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import mujoco
import numpy as np

from armmind.sim.assets import REPO_ROOT, AssetsMissing

FRICTION = "2 0.05 0.001"  # slide, twist, roll. The twist and the condim below keep a held object from turning.
# Stiffer and harder contacts than the default. A soft contact lets a scanned object creep out of the weak
# Panda grip, because the grip touches a curved surface at few points.
CONTACT = 'condim="4" solref="0.005 1" solimp="0.95 0.99 0.001"'
_PIECE = re.compile(r"model_collision_(\d+)\.obj")


def objects_dir() -> Path:
    return Path(os.environ.get("ARMMIND_OBJECTS", REPO_ROOT / "assets" / "scanned_objects" / "models"))


def mesh_dir(mesh: str) -> Path:
    folder = objects_dir() / mesh
    if not (folder / "model.obj").is_file():
        raise AssetsMissing(f"scanned object {mesh} not found in {folder.parent}. Run scripts/fetch_objects.sh")
    return folder


def have_meshes(meshes) -> bool:
    try:
        for m in meshes:
            mesh_dir(m)
    except AssetsMissing:
        return False
    return True


def pieces(mesh: str) -> list[int]:
    """The numbers of the collision pieces, in order."""
    return sorted(int(m.group(1)) for f in mesh_dir(mesh).iterdir() if (m := _PIECE.fullmatch(f.name)))


def asset_xml(name: str, mesh: str) -> str:
    """The texture, material and meshes of one object. The names start with the object name. The paths are
    absolute, so the meshdir of panda.xml does not apply."""
    folder = mesh_dir(mesh)
    return "\n".join([
        f'    <texture name="{name}_tex" type="2d" file="{folder / "texture.png"}"/>',
        f'    <material name="{name}_mat" texture="{name}_tex" specular="0.3" shininess="0.3"/>',
        f'    <mesh name="{name}_vis" file="{folder / "model.obj"}"/>',
        *(f'    <mesh name="{name}_col{i}" file="{folder / f"model_collision_{i}.obj"}"/>' for i in pieces(mesh)),
    ])


def _vec(v) -> str:
    return " ".join(f"{float(x) + 0.0:g}" for x in v)  # + 0.0 turns -0.0 into 0.0


def geoms_xml(name: str, mesh: str, offset, density: float) -> str:
    """The visual geom and the collision geoms. Each geom moves by -offset, so the body origin is at offset.
    The visual geom has no mass and no contacts. The collision geoms are in group 3, which renders hide."""
    pos = _vec(-np.asarray(offset, dtype=float))
    return "\n".join([
        (f'      <geom name="{name}" type="mesh" mesh="{name}_vis" material="{name}_mat" pos="{pos}" '
         'contype="0" conaffinity="0" group="2" density="0"/>'),
        *(f'      <geom name="{name}_col{i}" type="mesh" mesh="{name}_col{i}" pos="{pos}" group="3" '
          f'density="{density:g}" friction="{FRICTION}" {CONTACT}/>' for i in pieces(mesh)),
    ])


def collision_geoms(model: mujoco.MjModel, body: int) -> list[int]:
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == body
            and (model.geom_contype[g] or model.geom_conaffinity[g])]


def geom_points(model: mujoco.MjModel, g: int) -> np.ndarray:
    """The vertices of a mesh geom, or the corners of a box geom, in the body frame."""
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        m = model.geom_dataid[g]
        local = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
    else:  # a box, or the bounding box of another primitive
        half = model.geom_aabb[g][3:]
        local = model.geom_aabb[g][:3] + half * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                                                           for sz in (-1, 1)])
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[g])
    return local @ rot.reshape(3, 3).T + model.geom_pos[g]


def body_bounds(model: mujoco.MjModel, body: int) -> tuple[np.ndarray, np.ndarray]:
    """The low and high corners of the collision geoms of a body, in the body frame."""
    pts = np.vstack([geom_points(model, g) for g in collision_geoms(model, body)])
    return pts.min(axis=0), pts.max(axis=0)


@dataclass(frozen=True)
class MeshInfo:
    lo: tuple[float, float, float]  # the box around the collision pieces, in the frame of the files
    hi: tuple[float, float, float]
    mass: float  # kg at density 1000

    @property
    def dims(self) -> tuple[float, float, float]:
        return tuple(float(h - lo) for lo, h in zip(self.lo, self.hi, strict=True))

    @property
    def centre(self) -> tuple[float, float, float]:
        return tuple(float((lo + h) / 2) for lo, h in zip(self.lo, self.hi, strict=True))


@cache
def measure(mesh: str) -> MeshInfo:
    """Compiles the collision pieces of one object alone and measures them."""
    folder = mesh_dir(mesh)
    assets = "".join(f'<mesh name="c{i}" file="{folder / f"model_collision_{i}.obj"}"/>' for i in pieces(mesh))
    geoms = "".join(f'<geom type="mesh" mesh="c{i}"/>' for i in pieces(mesh))
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><asset>{assets}</asset><worldbody><body name="o">{geoms}</body></worldbody></mujoco>')
    body = model.body("o").id
    lo, hi = body_bounds(model, body)
    return MeshInfo(tuple(map(float, lo)), tuple(map(float, hi)), float(model.body_mass[body]))
