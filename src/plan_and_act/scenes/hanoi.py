"""Tower of Hanoi: a base with three upright pegs and three square rings of different sizes.

A ring is four boxes around a square hole. ObjSpec gives each object one geom, so a ring is a body in
fixtures_xml. It moves on three slide joints (x, y, z) and does not turn or tilt. A task can then set the start
place of each ring through initial_qpos, and a dropped ring slides down its peg without a jam.
"""
from __future__ import annotations

from plan_and_act.scenes.base import Scene, Task, box, cylinder
from plan_and_act.scenes.common import above, free
from plan_and_act.types import Action

BASE_X, BASE_HX, BASE_HY, BASE_T = 0.55, 0.05, 0.15, 0.01  # the base board: centre x, half sizes, thickness
PEGS = {"left": -0.10, "middle": 0.0, "right": 0.10}  # peg name -> y
PEG_R, PEG_H = 0.006, 0.042  # the pegs stand on the base board. The hand stays above a peg top.
PEG_TOP = BASE_T + PEG_H
HOLE = 0.014  # the half width of the square hole in each ring: 8 mm clear of a peg, for the arm error
RING_H = 0.015
# Ring name -> (half outer width, colour, label). The large ring is at the bottom of the start stack.
RINGS = {
    "large_ring": (0.032, "0.85 0.3 0.02 1", "large orange ring"),
    "medium_ring": (0.027, "0.1 0.2 0.9 1", "medium blue ring"),
    "small_ring": (0.022, "0.9 0.1 0.1 1", "small red ring"),
}
WOOD, METAL = "0.62 0.45 0.3 1", "0.55 0.57 0.6 1"
LIFT_Z = 0.16  # a carried ring clears the peg tops
GRIP_DZ = 0.006  # the tcp is this far above the ring centre: the fingertips stay above the ring below
DROP = 0.01  # let go of a ring when its bottom is this far below the peg top: it then slides down the peg
ON_PEG = HOLE - PEG_R + 0.001  # a ring around a peg has its centre this close to the peg axis, along x and y


def _level(i: int) -> float:
    """The centre height of a ring at stack level i: 0 is on the base board."""
    return BASE_T + RING_H / 2 + i * RING_H


def _geom(name: str, x: float, y: float, z: float, hx: float, hy: float, hz: float, rgba: str) -> str:
    return (f'      <geom name="{name}" type="box" pos="{x:g} {y:g} {z:g}" '
            f'size="{hx:g} {hy:g} {hz:g}" rgba="{rgba}" friction="1.5 0.01 0.001" density="500"/>')


def _walls(outer: float) -> list[tuple[str, float, float, float, float]]:
    """The four walls of a ring, in its body frame: (side, x, y, half x, half y)."""
    mid, half_t = (outer + HOLE) / 2, (outer - HOLE) / 2
    return [("near", -mid, 0, half_t, outer), ("far", mid, 0, half_t, outer),
            ("left", 0, -mid, HOLE, half_t), ("right", 0, mid, HOLE, half_t)]


# The scene start: all three rings on the left peg, the large ring at the bottom. Slide 0 is this place.
START = {ring: (BASE_X, PEGS["left"], _level(i)) for i, ring in enumerate(RINGS)}


def _ring_xml(ring: str) -> str:
    outer, rgba, _ = RINGS[ring]
    x, y, z = START[ring]
    return "\n".join([
        f'    <body name="{ring}" pos="{x:g} {y:g} {z:g}">',
        *(f'      <joint name="{ring}_{a}" type="slide" axis="{axis}"/>'
          for a, axis in (("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1"))),
        *(_geom(f"{ring}_{side}", dx, dy, 0, hx, hy, RING_H / 2, rgba) for side, dx, dy, hx, hy in _walls(outer)),
        "    </body>",
    ])


FIXTURES = "\n".join([
    f'    <body name="hanoi_base" pos="{BASE_X:g} 0 0">',
    _geom("hanoi_base", 0, 0, BASE_T / 2, BASE_HX, BASE_HY, BASE_T / 2, WOOD),
    *(f'      <geom name="peg_{p}" type="cylinder" pos="0 {y:g} {BASE_T + PEG_H / 2:g}" '
      f'size="{PEG_R:g} {PEG_H / 2:g}" rgba="{METAL}"/>' for p, y in PEGS.items()),
    "    </body>",
    *(_ring_xml(ring) for ring in RINGS),
])
SHAPES = (
    box("hanoi_base", (BASE_X, 0, BASE_T / 2), (2 * BASE_HX, 2 * BASE_HY, BASE_T), WOOD),
    *(cylinder(f"peg_{p}", (BASE_X, y, BASE_T + PEG_H / 2), PEG_R, PEG_H, METAL) for p, y in PEGS.items()),
    *(box(f"{ring}_{side}", (START[ring][0] + dx, START[ring][1] + dy, START[ring][2]), (2 * hx, 2 * hy, RING_H),
          RINGS[ring][1])
      for ring in RINGS for side, dx, dy, hx, hy in _walls(RINGS[ring][0])),
)
TEXT = (f"A wooden base board at x {BASE_X - BASE_HX:.2f} to {BASE_X + BASE_HX:.2f}, y {-BASE_HY:.2f} to "
        f"{BASE_HY:.2f}, {BASE_T * 100:.0f} cm thick. Three metal pegs stand on it at x {BASE_X:.2f}: "
        + ", ".join(f"the {p} peg at y {y:.2f}" for p, y in PEGS.items())
        + f". Each peg is {2 * PEG_R * 100:.1f} cm wide and its top is at z {PEG_TOP:.3f}. "
        f"Three square rings can move. Each one is {RING_H * 100:.1f} cm tall with a square hole "
        f"{2 * HOLE * 100:.1f} cm wide: "
        + ", ".join(f"the {label}, {2 * outer * 100:.1f} cm wide" for outer, _, label in RINGS.values())
        + ". At the start, the three rings are on the left peg, with the large ring at the bottom. "
        "Grasp a ring across its outer sides, at its centre height.")


def on_peg(w, ring: str, peg: str) -> bool:
    """The peg goes through the ring hole: the ring centre is near the peg axis and the ring bottom is below
    the peg top. The gripper does not touch the ring."""
    x, y, z = w.object_pos(ring)
    near = abs(x - BASE_X) < ON_PEG and abs(y - PEGS[peg]) < ON_PEG
    return bool(near and z - RING_H / 2 < PEG_TOP and free(w, ring))


def _level_of(w, ring: str) -> int:
    return round((w.object_pos(ring)[2] - _level(0)) / RING_H)


def _top_ring(w) -> bool:
    return on_peg(w, "small_ring", "right")


def _tower_on_right(w) -> bool:
    """All three rings are on the right peg, the large ring at the bottom and the small ring on top."""
    return all(on_peg(w, ring, "right") and _level_of(w, ring) == i for i, ring in enumerate(RINGS))


def _move(w, ring: str, peg: str) -> list[Action]:
    """Grasp the ring across its sides, carry it above the peg, lower it over the peg top and let it go."""
    p = w.object_pos(ring)
    grip = float(p[2]) + GRIP_DZ
    to = (BASE_X, PEGS[peg])
    release = PEG_TOP - DROP + RING_H / 2 + GRIP_DZ
    return [Action("open"), above(p, LIFT_Z), above(p, grip), Action("close"), above(p, LIFT_Z), above(to, LIFT_Z),
            above(to, release), Action("open"), above(to, LIFT_Z)]


def _top_ring_plan(w) -> list[Action]:
    return _move(w, "small_ring", "right")


def _last_two_plan(w) -> list[Action]:
    return [*_move(w, "medium_ring", "right"), *_move(w, "small_ring", "right")]


def _offsets(places: dict[str, tuple[str, int]]) -> dict[str, float]:
    """The slide joint values that put each ring on a peg at a stack level: ring -> (peg, level)."""
    out = {}
    for ring, (peg, i) in places.items():
        x, y, z = START[ring]
        out |= {f"{ring}_x": BASE_X - x, f"{ring}_y": PEGS[peg] - y, f"{ring}_z": _level(i) - z}
    return out


# The state before the last two moves of a three-ring tower: one ring on each peg.
LAST_TWO = _offsets({"large_ring": ("right", 0), "medium_ring": ("middle", 0), "small_ring": ("left", 0)})

BRIEF = 'A wooden base board with three metal pegs: left, middle and right. Three square rings with square holes can move: a large orange ring, a medium blue ring and a small red ring. At the start, the three rings are on the left peg, with the large ring at the bottom. Grasp a ring across its outer sides.'  # the fixed items for a prompt with names only: no positions or sizes

HANOI = Scene(
    name="hanoi", fixtures_brief=BRIEF, title="Tower of Hanoi", objects=(), fixtures_xml=FIXTURES, fixtures_text=TEXT, fixtures=SHAPES,
    region=(0.30, 0.34, -0.38, -0.34), min_gap=0.10,  # no free objects: a small region far from the base
    tasks={t.name: t for t in (
        Task("top_ring_to_right", "Move the small red ring from the top of the left peg to the right peg.",
             _top_ring, _top_ring_plan),
        Task("last_two_moves",
             "The large orange ring is on the right peg, the medium blue ring is on the middle peg and the small "
             "red ring is on the left peg. Finish the tower on the right peg: the large ring at the bottom, "
             "the medium ring in the middle and the small ring on top. Never put a ring on a smaller ring.",
             _tower_on_right, _last_two_plan, initial_qpos=LAST_TWO),
    )},
)
