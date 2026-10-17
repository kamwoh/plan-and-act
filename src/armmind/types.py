"""Plain data types shared by the simulation, the agent and the server."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ActionType = Literal["move_to", "open", "close", "wait"]
Status = Literal["reached", "blocked", "unreachable"]
Vec3 = tuple[float, float, float]
CLOSED_ON_NOTHING_M = 0.005  # a closed gripper below this width holds nothing


def fmt(p) -> str:
    return f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"


@dataclass(frozen=True)
class Action:
    type: ActionType
    xyz: Vec3 | None = None
    seconds: float = 0.5
    yaw: float | None = None  # degrees about world +z, 0 is the start orientation; None keeps the current yaw
    expect: str | None = None  # what the gripper should hold after the action: an object, or "nothing"

    def describe(self) -> str:
        if self.type == "move_to":
            if self.yaw is None:
                return f"move_to{fmt(self.xyz)}"
            return f"move_to{fmt(self.xyz)[:-1]}, yaw {self.yaw:g}°)"
        if self.type == "wait":
            return f"wait({self.seconds:.1f} s)"
        return self.type


@dataclass(frozen=True)
class ActionResult:
    action: Action
    status: Status
    tcp: Vec3
    gripper_width: float
    contacts: tuple[str, ...] = ()
    note: str = ""
    gripper_yaw: float = 0.0  # degrees
    # The scene objects that the gripper holds before and after the action: closed, not empty, touching.
    # None means unknown.
    held_before: tuple[str, ...] | None = None
    held_after: tuple[str, ...] | None = None

    def summary(self) -> str:
        s = f"{self.action.describe()}: {self.status}, gripper at {fmt(self.tcp)}"
        if round(self.gripper_yaw):
            s += f", yaw {self.gripper_yaw:.0f}°"
        s += f", width {self.gripper_width:.3f}"
        if self.contacts:
            s += f", touching {', '.join(self.contacts)}"
        if self.note:
            s += f" ({self.note})"
        if self.held_before is not None and self.held_after is not None:
            if gained := [n for n in self.held_after if n not in self.held_before]:
                s += f", now holding {', '.join(gained)}"
            if lost := [n for n in self.held_before if n not in self.held_after]:
                s += f", let go of {', '.join(lost)}"
        return s


@dataclass(frozen=True)
class Observation:
    front_png: bytes
    top_png: bytes
    tcp: Vec3
    gripper_width: float
    contacts: tuple[str, ...]
    objects: dict[str, Vec3] | None = None  # only with the debug switch on
    gripper_yaw: float = 0.0  # degrees
    events: tuple[str, ...] = ()  # what happened since the last observation, e.g. "red button pressed"
    joints: tuple[str, ...] | None = None  # moving parts with units, e.g. "drawer slide 0.120 m"; debug only
    object_yaws: dict[str, float] | None = None  # degrees about world +z, in (-180, 180]; debug only
