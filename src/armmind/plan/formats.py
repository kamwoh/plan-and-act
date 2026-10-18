"""Generic plan formats: entities, keyframes and the view that the page plays. No simulator code here."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Protocol

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
class Move:
    """A predicted pose of an entity at the end of a keyframe, for example after a push."""
    name: str
    xyz: Vec3
    yaw: float = 0.0  # degrees

    def to_dict(self) -> dict:
        return {"name": self.name, "xyz": list(self.xyz), "yaw": self.yaw}

    @classmethod
    def from_dict(cls, d: dict) -> Move:
        return cls(str(d["name"]), tuple(float(v) for v in d["xyz"]), float(d.get("yaw", 0.0)))


@dataclass(frozen=True)
class Keyframe:
    """A pose of the agent. The agent goes there in a straight line, then changes to `state`."""
    xyz: Vec3
    yaw: float  # degrees
    state: str  # here "open" or "closed"
    carrying: str | None  # an entity name, or None
    note: str = ""
    moves: tuple[Move, ...] = ()  # entities that the keyframe moves without carrying them

    def to_dict(self) -> dict:
        return {"xyz": list(self.xyz), "yaw": self.yaw, "state": self.state, "carrying": self.carrying,
                "note": self.note, "moves": [m.to_dict() for m in self.moves]}

    @classmethod
    def from_dict(cls, d: dict) -> Keyframe:
        return cls(tuple(float(v) for v in d["xyz"]), float(d["yaw"]), str(d["state"]), d.get("carrying"),
                   str(d.get("note", "")), tuple(Move.from_dict(m) for m in d.get("moves") or []))


def entity_from_dict(d: dict) -> Entity:
    color = d.get("color")
    return Entity(str(d["name"]), str(d["shape"]), tuple(float(v) for v in d["size"]),
                  tuple(float(v) for v in d["xyz"]), float(d.get("yaw", 0.0)),
                  tuple(float(v) for v in color) if color is not None else None)


def build_track(start: dict, keyframes, move_s: Callable[[Vec3, Vec3, float, float], float], settle_s: float,
                change_s: float, gap: Callable[[Keyframe], float]) -> dict:
    """The keyframes with the time of each one, from the motion rule of an environment.

    The agent moves from the last pose to the keyframe in `move_s` seconds and arrives at `t`. It holds still
    for `settle_s`, and for `change_s` more when its state changes. `t_done` is the end of all that.
    `start` holds the xyz, the yaw and the state before the first keyframe."""
    out, t = [], 0.0
    xyz, yaw, state = tuple(start["xyz"]), float(start["yaw"]), start["state"]
    for k in keyframes:
        t += move_s(xyz, k.xyz, yaw, k.yaw)
        changes = k.state != state
        done = t + settle_s + (change_s if changes else 0.0)
        out.append({**k.to_dict(), "gap": gap(k), "changes": changes, "t": round(t, 4), "t_done": round(done, 4)})
        xyz, yaw, state, t = k.xyz, k.yaw, k.state, done
    begin = Keyframe(tuple(start["xyz"]), float(start["yaw"]), start["state"], None)
    return {"start": {"xyz": list(begin.xyz), "yaw": begin.yaw, "state": begin.state, "gap": gap(begin)},
            "keyframes": out, "duration": round(t, 4)}


def norm_name(name: str) -> str:
    """A name for matching: lower case, with spaces in place of underscores."""
    return " ".join(name.strip().lower().replace("_", " ").split())


class PlanEnvironment(Protocol):
    """What an environment gives for a plan. The MuJoCo arm environment implements it."""

    def plan_view(self, belief: tuple[Entity, ...], keyframes: tuple[Keyframe, ...],
                  marks: list[dict] | None = None) -> dict: ...

    def check_plan(self, belief: tuple[Entity, ...], keyframes: tuple[Keyframe, ...]) -> dict: ...

    def execute_plan(self, keyframes: tuple[Keyframe, ...], should_stop: Callable[[], bool],
                     on_action: Callable[[int, int], None] | None = None) -> dict: ...


def match_entities(believed: list[str], truth: dict[str, list[str]]) -> dict[str, str]:
    """Match believed names to true names. `truth` gives each true name its other names, such as labels.
    Matching ignores case and treats "_" as a space. Each name matches once. Gives true name -> believed name."""
    out: dict[str, str] = {}
    taken: set[int] = set()
    for true_name, other in truth.items():
        keys = {norm_name(true_name), *(norm_name(o) for o in other if o)}
        for i, name in enumerate(believed):
            if i not in taken and norm_name(name) in keys:
                out[true_name] = name
                taken.add(i)
                break
    return out
