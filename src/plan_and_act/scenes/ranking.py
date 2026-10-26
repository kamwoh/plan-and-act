"""Three cubes of different colours and sizes, and a flat strip with three slots: put the cubes in order."""
from __future__ import annotations

from plan_and_act.scenes.base import ObjSpec, Scene, Task, box
from plan_and_act.scenes.common import free, put

# The sizes do not follow the colours, so the two orders are different.
CUBES = (
    ObjSpec("red_cube", "box", (0.025, 0.025, 0.025), "0.9 0.1 0.1 1", "red cube, 5 cm", mass=0.06),
    ObjSpec("green_cube", "box", (0.015, 0.015, 0.015), "0.1 0.7 0.2 1", "green cube, 3 cm", mass=0.03),
    ObjSpec("blue_cube", "box", (0.02, 0.02, 0.02), "0.1 0.2 0.9 1", "blue cube, 4 cm"),
)
RGB = ("red_cube", "green_cube", "blue_cube")
BY_SIZE = ("green_cube", "blue_cube", "red_cube")  # smallest first

SLOT_X, SLOT_HALF = 0.66, 0.03
SLOTS = (-0.12, 0.0, 0.12)  # slot 1, 2 and 3: y centres, left to right in the front image
STRIP_HX, STRIP_HY = 0.045, 0.18
STRIP, SLOT = "0.85 0.85 0.8 1", "0.55 0.55 0.5 1"
ON_TABLE = 0.005  # a cube in a slot has its lowest point below this

# Visual only: contype and conaffinity 0, so nothing collides with the strip or a slot.
_FLAT = 'contype="0" conaffinity="0"'
FIXTURES = "\n".join([
    f'    <body name="ranking_strip" pos="{SLOT_X:g} 0 0.0006">',
    f'      <geom name="ranking_strip" type="box" size="{STRIP_HX:g} {STRIP_HY:g} 0.0005" rgba="{STRIP}" {_FLAT}/>',
    *(f'      <geom name="slot_{i + 1}" type="box" pos="0 {y:g} 0.0003" size="{SLOT_HALF:g} {SLOT_HALF:g} 0.0005" '
      f'rgba="{SLOT}" {_FLAT}/>' for i, y in enumerate(SLOTS)),
    "    </body>",
])
SHAPES = (
    box("ranking_strip", (SLOT_X, 0, 0.0006), (2 * STRIP_HX, 2 * STRIP_HY, 0.001), STRIP),
    *(box(f"slot_{i + 1}", (SLOT_X, y, 0.0009), (2 * SLOT_HALF, 2 * SLOT_HALF, 0.001), SLOT)
      for i, y in enumerate(SLOTS)),
)
TEXT = (f"A flat light strip on the table at x {SLOT_X:.2f}, from y {-STRIP_HY:.2f} to {STRIP_HY:.2f}. "
        f"It has three dark square slots, each {2 * SLOT_HALF * 100:.0f} cm wide, centred at x {SLOT_X:.2f}: "
        f"slot 1 at y {SLOTS[0]:.2f}, slot 2 at y {SLOTS[1]:.2f}, slot 3 at y {SLOTS[2]:.2f}. "
        f"Slot 1 is on the left in the front image.")


def in_slot(w, name: str, slot: int) -> bool:
    """The object's centre is over the slot (1, 2 or 3), it is on the table, and the gripper does not touch it."""
    x, y, _ = w.object_pos(name)
    return bool(abs(x - SLOT_X) <= SLOT_HALF and abs(y - SLOTS[slot - 1]) <= SLOT_HALF
                and w.object_bottom(name) < ON_TABLE and free(w, name))


def in_order(w, names) -> bool:
    return all(in_slot(w, n, i + 1) for i, n in enumerate(names))


def _plan(w, names):
    return [a for n, y in zip(names, SLOTS, strict=True)
            for a in put(w, n, (SLOT_X, y), w.object_info(n).height / 2 + 0.005)]


BRIEF = 'A flat light strip on the table with three dark square slots: slot 1, slot 2 and slot 3. Slot 1 is on the left in the front image.'  # the fixed items for a prompt with names only: no positions or sizes

RANKING = Scene(
    name="ranking", fixtures_brief=BRIEF, title="Ranking", objects=CUBES, fixtures_xml=FIXTURES, fixtures_text=TEXT, fixtures=SHAPES,
    region=(0.38, 0.56, -0.25, 0.25), min_gap=0.10,
    tasks={t.name: t for t in (
        Task("rank_rgb", "Put the red, green and blue cubes in slots 1, 2 and 3, in this order.",
             lambda w: in_order(w, RGB), lambda w: _plan(w, RGB)),
        Task("rank_size", "Put the cubes in the slots by size: the smallest in slot 1 and the largest in slot 3.",
             lambda w: in_order(w, BY_SIZE), lambda w: _plan(w, BY_SIZE)),
    )},
)
