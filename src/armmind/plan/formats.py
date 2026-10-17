"""Generic plan formats: entities, keyframes and the view that the page plays. No simulator code here."""
from __future__ import annotations

from dataclasses import asdict, dataclass

Vec3 = tuple[float, float, float]
SHAPES = ("box", "cylinder", "sphere")


@dataclass(frozen=True)
class Entity:
    """A thing in the scene. The size is the full extent in metres along x, y and z of its own frame."""
    name: str
    shape: str  # "box", "cylinder" or "sphere"
    size: Vec3
    xyz: Vec3  # the centre
    yaw: float = 0.0  # degrees about +z
    color: tuple[float, float, float, float] | None = None  # r, g, b, a from 0 to 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["size"], d["xyz"] = list(self.size), list(self.xyz)
        d["color"] = list(self.color) if self.color is not None else None
        return d


@dataclass(frozen=True)
class Keyframe:
    """A pose of the agent. The agent goes there in a straight line, then changes to `state`."""
    xyz: Vec3
    yaw: float  # degrees
    state: str  # here "open" or "closed"
    carrying: str | None  # an entity name, or None
    note: str = ""

    def to_dict(self) -> dict:
        return {"xyz": list(self.xyz), "yaw": self.yaw, "state": self.state, "carrying": self.carrying,
                "note": self.note}

    @classmethod
    def from_dict(cls, d: dict) -> Keyframe:
        return cls(tuple(float(v) for v in d["xyz"]), float(d["yaw"]), str(d["state"]), d.get("carrying"),
                   str(d.get("note", "")))


def entity_from_dict(d: dict) -> Entity:
    color = d.get("color")
    return Entity(str(d["name"]), str(d["shape"]), tuple(float(v) for v in d["size"]),
                  tuple(float(v) for v in d["xyz"]), float(d.get("yaw", 0.0)),
                  tuple(float(v) for v in color) if color is not None else None)
