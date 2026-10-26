import re

import mujoco
import numpy as np
import pytest

from plan_and_act.scenes.laptop import LAPTOP
from plan_and_act.sim.arm import Arm
from plan_and_act.sim.world import World

MINE = [pytest.param(LAPTOP, id="laptop")]  # the generic checks run on each scene of this file


def _geom_aabb(model, data, g):
    c, half = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half + c
    world = corners @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
    return world.min(0), world.max(0)


def _world(lid: float | None = None) -> World:
    w = World(LAPTOP)
    w.reset(LAPTOP.layout(0), None if lid is None else {"laptop_lid": lid})
    return w


def test_tasks_and_text():
    assert list(LAPTOP.tasks) == ["open_laptop", "close_laptop"]
    assert LAPTOP.objects == () and LAPTOP.layout(0) == {}
    assert LAPTOP.describe() == f"Objects: none. {LAPTOP.fixtures_text}"


@pytest.mark.assets
@pytest.mark.parametrize("scene", MINE)
def test_model_loads(scene):
    w = World(scene)
    w.reset(scene.layout(0))
    assert w.objects == scene.objects
    assert set(w.joints()) == set(scene.joints)


@pytest.mark.assets
@pytest.mark.parametrize("scene", MINE)
def test_fixture_list_agrees_with_the_fixture_xml(scene):
    names = re.findall(r'<geom name="([^"]+)"', scene.fixtures_xml)
    assert sorted(e.name for e in scene.fixtures) == sorted(names)
    w = World(scene)
    mujoco.mj_forward(w.model, w.data)
    for e in scene.fixtures:
        lo, hi = _geom_aabb(w.model, w.data, w.model.geom(e.name).id)
        half = np.array(e.size) / 2
        assert np.allclose(lo, np.array(e.xyz) - half, atol=5e-4), e.name
        assert np.allclose(hi, np.array(e.xyz) + half, atol=5e-4), e.name


@pytest.mark.assets
@pytest.mark.parametrize("scene", MINE)
def test_fixtures_are_outside_the_region(scene):
    w = World(scene)
    m = w.model
    grow = max((o.footprint for o in scene.objects), default=0.0)
    x0, x1, y0, y1 = scene.region
    fixtures = 0
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        if b == 0 or m.body_parentid[b] != 0 or m.body_jntnum[b] or m.body(b).name == "link0":
            continue
        fixtures += 1
        cx, cy = m.body_pos[b][:2] + m.geom_pos[g][:2]
        hx, hy = m.geom_size[g][:2]
        assert cx + hx < x0 - grow or cx - hx > x1 + grow or cy + hy < y0 - grow or cy - hy > y1 + grow
    assert fixtures == 1


@pytest.mark.assets
@pytest.mark.parametrize("scene", MINE)
def test_objects_stay_still_at_their_start_poses(scene):
    w = World(scene)
    for seed in range(2):
        w.reset(scene.layout(seed))
        w.step(200)
        for obj in scene.objects:
            np.testing.assert_allclose(w.object_pos(obj.name)[:2], scene.layout(seed)[obj.name][:2], atol=0.002)


@pytest.mark.assets
@pytest.mark.parametrize("lid", [0.0, 0.3, 0.6, 1.2, 1.75])
def test_lid_stays_where_it_is_left(lid):
    w = _world(lid)
    w.step(1500)
    assert w.joint_pos("laptop_lid") == pytest.approx(lid, abs=0.01)


@pytest.mark.assets
@pytest.mark.parametrize("scene", MINE)
def test_no_task_is_solved_at_start(scene):
    for task in scene.tasks.values():
        w = World(scene)
        w.reset(scene.layout(0), task.initial_qpos)
        Arm(w).reset_ready()
        assert not task.success(w), task.name


@pytest.mark.assets
@pytest.mark.parametrize("scene,task_name,seed",
                         [pytest.param(p.values[0], t, n, id=f"{p.id}-{t}-{n}") for p in MINE
                          for t in p.values[0].tasks for n in range(5)])
def test_solution_solves_task(scene, task_name, seed):
    task = scene.tasks[task_name]
    w = World(scene)
    w.reset(scene.layout(seed), task.initial_qpos)
    arm = Arm(w)
    arm.reset_ready()
    for action in task.solution(w):
        arm.act(action)
    assert task.success(w)


@pytest.mark.assets
@pytest.mark.parametrize("task_name,lid,expected", [
    ("open_laptop", 1.0, False),
    ("open_laptop", 1.3, True),
    ("close_laptop", 0.04, True),
    ("close_laptop", 0.2, False),
])
def test_success_check_on_set_lid_angles(task_name, lid, expected):
    w = _world(lid)
    assert w.joint_pos("laptop_lid") == pytest.approx(lid)
    assert LAPTOP.tasks[task_name].success(w) is expected
