"""Grasp and place for objects that can be turned: the gripper turns so that the fingers close across
the narrow side of the object."""
from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from armmind.scenes.base import ObjSpec, Scene
from armmind.types import Action

if TYPE_CHECKING:
    from armmind.sim.world import World

CARRY_Z = 0.22  # high enough to carry a tall item over a bowl rim
TALL = 0.06  # an item taller than this is held below its centre
LOW_GRIP = 0.4  # the grasp height of a tall item, as a part of its height
MIN_GRIP_Z = 0.02  # the lowest grasp height above the item bottom
FINGER_REACH = 0.035  # the item top must stay this close above the grasp point, or the hand hits it


def grasp_yaw(obj_yaw: float, narrow: str) -> float:
    """The gripper yaw in degrees, in [-90, 90), that closes the fingers along the narrow axis of the object.
    At yaw 0 the fingers close along world x. A positive yaw turns counter-clockwise, seen from above."""
    along = obj_yaw + (90.0 if narrow == "y" else 0.0)
    return (along + 90.0) % 180.0 - 90.0


def items(scene: Scene) -> tuple[ObjSpec, ...]:
    """The objects to pick up: the ones without a fixed place."""
    return tuple(o for o in scene.objects if o.place is None)


def grip_height(w: World, name: str) -> float:
    """The grasp height above the item bottom: its centre, or lower for a tall item, but high enough that
    the hand stays clear of the item top."""
    h = w.object_info(name).height
    return max(MIN_GRIP_Z, LOW_GRIP * h if h > TALL else h / 2, h - FINGER_REACH)


def _at(p, z: float, yaw: float | None = None) -> Action:
    return Action("move_to", (float(p[0]), float(p[1]), float(z)), yaw=yaw)


def grip(w: World, name: str) -> list[Action]:
    """Open, turn and move above the item, go down to the grasp height, close."""
    p = w.object_pos(name)
    yaw = grasp_yaw(w.object_yaw(name), w.object_info(name).narrow)
    z = w.object_bottom(name) + grip_height(w, name)
    return [Action("open"), _at(p, CARRY_Z, yaw), _at(p, z), Action("close")]


def place(w: World, name: str, xy, bottom_z: float, yaw: float | None = None) -> list[Action]:
    """Grasp the item, carry it above xy and lower it until its bottom is at bottom_z. Then let go and move up.
    With a yaw, the gripper turns to it on the way."""
    p = w.object_pos(name)
    release = bottom_z + grip_height(w, name)
    return [*grip(w, name), _at(p, CARRY_Z), _at(xy, CARRY_Z, yaw), _at(xy, release), Action("open"),
            _at(xy, CARRY_Z)]


def surface_z(w: World, xy, start_z: float = 0.5) -> float:
    """The height of the first surface below (x, y, start_z): the table or a collision geom.
    Visual geoms do not count. The arm must not be above the point."""
    geomid = np.zeros(1, dtype=np.int32)
    groups = np.array([1, 0, 0, 1, 0, 0], dtype=np.uint8)  # group 0: the table and fixtures; group 3: collision
    dist = mujoco.mj_ray(w.model, w.data, np.array([xy[0], xy[1], start_z], dtype=float),
                         np.array([0.0, 0.0, -1.0]), groups, 1, -1, geomid)
    return start_z - dist if dist >= 0 else 0.0
