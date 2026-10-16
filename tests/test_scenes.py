import itertools

import numpy as np
import pytest

from armmind.scenes import SCENES
from armmind.sim.arm import Arm
from armmind.sim.world import World

CASES = [(s, t) for s in SCENES for t in SCENES[s].tasks]


def _radius(obj) -> float:
    """The radius of the object's footprint on the table."""
    return float(np.hypot(obj.size[0], obj.size[1])) if obj.shape == "box" else float(obj.size[0])


def test_registry_order_and_tasks():
    assert list(SCENES) == ["cubes", "tray", "zones", "shapes"]
    assert {n: list(s.tasks) for n, s in SCENES.items()} == {
        "cubes": ["reach", "pick", "stack"], "tray": ["red_in_tray", "all_in_tray"],
        "zones": ["red_on_green", "sort"], "shapes": ["pick_cylinder", "small_on_big"]}


def test_cubes_layout_is_unchanged():
    cubes = SCENES["cubes"]
    assert cubes.layout(0) == {"red_cube": (0.559, -0.115), "blue_cube": (0.41, -0.242),
                               "green_cube": (0.603, 0.206)}
    assert cubes.layout(4) == {"red_cube": (0.636, 0.006), "blue_cube": (0.644, -0.21),
                               "green_cube": (0.552, -0.062)}


@pytest.mark.parametrize("name", list(SCENES))
def test_layout_is_repeatable_and_spread_out(name):
    scene = SCENES[name]
    x0, x1, y0, y1 = scene.region
    for seed in range(20):
        a, b = scene.layout(seed), scene.layout(seed)
        assert a == b
        assert set(a) == {o.name for o in scene.objects}
        for p, q in itertools.combinations(a.values(), 2):
            assert np.hypot(p[0] - q[0], p[1] - q[1]) >= scene.min_gap
        for x, y in a.values():
            assert x0 <= x <= x1 and y0 <= y <= y1
    assert scene.layout(0) != scene.layout(1)


@pytest.mark.parametrize("name", list(SCENES))
def test_describe_lists_every_object(name):
    scene = SCENES[name]
    text = scene.describe()
    assert text.startswith("Objects: ")
    for obj in scene.objects:
        assert obj.label in text
    assert scene.fixtures_text in text


@pytest.mark.assets
@pytest.mark.parametrize("name", list(SCENES))
def test_model_loads_with_every_object(name):
    scene = SCENES[name]
    w = World(scene)
    w.reset(scene.layout(0))
    assert w.objects == scene.objects
    for obj in scene.objects:
        assert w.object_pos(obj.name)[2] == pytest.approx(obj.half_height, abs=1e-6)


@pytest.mark.assets
@pytest.mark.parametrize("name", list(SCENES))
def test_fixtures_are_outside_the_region(name):
    """Every fixture geom keeps clear of the spawn region, grown by the widest object."""
    scene = SCENES[name]
    w = World(scene)
    m = w.model
    grow = max(_radius(o) for o in scene.objects)
    x0, x1, y0, y1 = scene.region
    fixtures = 0
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        if b == 0 or m.body_parentid[b] != 0 or m.body_jntnum[b] or m.body(b).name == "link0":
            continue  # a fixture is a static body at the root, other than the arm base
        fixtures += 1
        cx, cy = m.body_pos[b][:2] + m.geom_pos[g][:2]
        hx, hy = m.geom_size[g][:2]
        apart = cx + hx < x0 - grow or cx - hx > x1 + grow or cy + hy < y0 - grow or cy - hy > y1 + grow
        assert apart, f"{m.body(b).name} geom {g} overlaps the region"
    assert fixtures == {"cubes": 0, "tray": 5, "zones": 3, "shapes": 0}[name]


@pytest.mark.assets
@pytest.mark.parametrize("name", list(SCENES))
def test_objects_rest_without_overlap(name):
    scene = SCENES[name]
    w = World(scene)
    for seed in range(5):
        w.reset(scene.layout(seed))
        w.step(200)
        for obj in scene.objects:
            np.testing.assert_allclose(w.object_pos(obj.name)[:2], scene.layout(seed)[obj.name], atol=0.002)
            assert w.object_pos(obj.name)[2] == pytest.approx(obj.half_height, abs=0.002)


@pytest.mark.assets
@pytest.mark.parametrize("scene_name,task_name", CASES)
def test_no_task_is_solved_at_start(scene_name, task_name):
    scene = SCENES[scene_name]
    w = World(scene)
    w.reset(scene.layout(0))
    Arm(w).reset_ready()
    assert not scene.tasks[task_name].success(w)


@pytest.mark.assets
@pytest.mark.parametrize("scene_name,task_name,seed", [(s, t, n) for s, t in CASES for n in range(3)])
def test_solution_solves_task(scene_name, task_name, seed):
    scene = SCENES[scene_name]
    task = scene.tasks[task_name]
    w = World(scene)
    w.reset(scene.layout(seed))
    arm = Arm(w)
    arm.reset_ready()
    for action in task.solution(w):
        arm.act(action)
    assert task.success(w)


@pytest.mark.assets
def test_zones_do_not_collide_and_the_tray_does():
    w = World(SCENES["zones"])
    m = w.model
    zone_geoms = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.endswith("_zone")]
    assert len(zone_geoms) == 3 and all(m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 for g in zone_geoms)
    t = World(SCENES["tray"]).model
    assert all(t.geom_contype[g] == 1 for g in range(t.ngeom) if t.body(t.geom_bodyid[g]).name == "tray")
