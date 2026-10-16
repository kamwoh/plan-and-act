"""Three cubes and three flat coloured zones: put cubes on zones."""
from __future__ import annotations

from armmind.scenes.base import Scene, Task
from armmind.scenes.common import THREE_CUBES, free, put

ZONE_X, ZONE_HALF, RELEASE_Z = 0.68, 0.04, 0.025
ZONES = {  # colour: (y, rgba)
    "red": (-0.15, "1 0.6 0.6 1"),
    "blue": (0.0, "0.6 0.7 1 1"),
    "green": (0.15, "0.6 0.9 0.65 1"),
}

# Visual only: contype and conaffinity 0, so nothing collides with a zone.
FIXTURES = "\n".join(
    f'    <body name="{c}_zone" pos="{ZONE_X:g} {y:g} 0.0006">\n'
    f'      <geom name="{c}_zone" type="box" size="{ZONE_HALF:g} {ZONE_HALF:g} 0.0005" rgba="{rgba}" '
    f'contype="0" conaffinity="0"/>\n'
    f'    </body>'
    for c, (y, rgba) in ZONES.items())
TEXT = ("Three flat coloured squares (zones) on the table, each 8 cm wide, centred at x 0.68: "
        "red zone at y -0.15, blue zone at y 0.00, green zone at y 0.15.")


def on_zone(w, name: str, colour: str) -> bool:
    x, y, z = w.object_pos(name)
    zy = ZONES[colour][0]
    return bool(abs(x - ZONE_X) <= ZONE_HALF and abs(y - zy) <= ZONE_HALF and z < 0.03 and free(w, name))


def _to_zone(w, name: str, colour: str):
    return put(w, name, (ZONE_X, ZONES[colour][0]), RELEASE_Z)


def _sort_plan(w):
    return [a for c in ZONES for a in _to_zone(w, f"{c}_cube", c)]


ZONES_SCENE = Scene(
    name="zones", title="Zones", objects=THREE_CUBES, fixtures_xml=FIXTURES, fixtures_text=TEXT,
    region=(0.40, 0.58, -0.25, 0.25), min_gap=0.10,
    tasks={t.name: t for t in (
        Task("red_on_green", "Put the red cube on the green zone.", lambda w: on_zone(w, "red_cube", "green"),
             lambda w: _to_zone(w, "red_cube", "green")),
        Task("sort", "Put each cube on the zone of the same colour.",
             lambda w: all(on_zone(w, f"{c}_cube", c) for c in ZONES), _sort_plan),
    )},
)
