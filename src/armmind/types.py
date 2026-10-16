"""Plain data types shared by the simulation, the agent and the server."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ActionType = Literal["move_to", "open", "close", "wait"]
Status = Literal["reached", "blocked", "unreachable"]
Vec3 = tuple[float, float, float]


def fmt(p) -> str:
    return f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"


@dataclass(frozen=True)
class Action:
    type: ActionType
    xyz: Vec3 | None = None
    seconds: float = 0.5

    def describe(self) -> str:
        if self.type == "move_to":
            return f"move_to{fmt(self.xyz)}"
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

    def summary(self) -> str:
        s = (f"{self.action.describe()}: {self.status}, gripper at {fmt(self.tcp)}, "
             f"width {self.gripper_width:.3f}")
        if self.contacts:
            s += f", touching {', '.join(self.contacts)}"
        if self.note:
            s += f" ({self.note})"
        return s


@dataclass(frozen=True)
class Observation:
    front_png: bytes
    top_png: bytes
    tcp: Vec3
    gripper_width: float
    contacts: tuple[str, ...]
    objects: dict[str, Vec3] | None = None  # only with the debug switch on
