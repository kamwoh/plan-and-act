"""The ranking scene: three cubes of different sizes and a flat strip with three slots."""
import re

import mujoco
import numpy as np
import pytest

from plan_and_act.scenes.ranking import RANKING, SLOT_X, SLOTS
from plan_and_act.sim.arm import Arm
from plan_and_act.sim.world import World
from tests.test_plan_view import _entity_aabb, _geom_aabb

TASKS = list(RANKING.tasks)


def test_tasks_and_sizes():
    assert TASKS == ["rank_rgb", "rank_size"]
    sizes = {o.name: o.dims[0] for o in RANKING.objects}
    assert sizes == {"red_cube": 0.05, "green_cube": 0.03, "blue_cube": 0.04}


@pytest.mark.assets
def test_model_loads_with_every_object():
    w = World(RANKING)
    w.reset(RANKING.layout(0))
    for obj in RANKING.objects:
        assert w.object_pos(obj.name)[2] == pytest.approx(obj.half_height, abs=1e-6)


@pytest.mark.assets
def test_fixture_list_agrees_with_the_fixture_xml():
    names = re.findall(r'<geom name="([^"]+)"', RANKING.fixtures_xml)
    assert sorted(e.name for e in RANKING.fixtures) == sorted(names)
    w = World(RANKING)
    mujoco.mj_forward(w.model, w.data)
    for e in RANKING.fixtures:
        lo, hi = _geom_aabb(w.model, w.data, w.model.geom(e.name).id)
        elo, ehi = _entity_aabb(e)
        assert np.allclose(lo, elo, atol=5e-4) and np.allclose(hi, ehi, atol=5e-4), e.name


@pytest.mark.assets
def test_strip_does_not_collide_and_keeps_clear_of_the_region():
    m = World(RANKING).model
    grow = max(o.footprint for o in RANKING.objects)
    x0, x1, y0, y1 = RANKING.region
    geoms = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name == "ranking_strip"]
    assert len(geoms) == 4
    for g in geoms:
        assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0
        cx, cy = m.body_pos[m.geom_bodyid[g]][:2] + m.geom_pos[g][:2]
        hx, hy = m.geom_size[g][:2]
        assert cx + hx < x0 - grow or cx - hx > x1 + grow or cy + hy < y0 - grow or cy - hy > y1 + grow


@pytest.mark.assets
def test_objects_stay_still_at_their_start_poses():
    w = World(RANKING)
    for seed in range(2):
        w.reset(RANKING.layout(seed))
        w.step(200)
        for obj in RANKING.objects:
            np.testing.assert_allclose(w.object_pos(obj.name)[:2], RANKING.layout(seed)[obj.name][:2], atol=0.002)
            assert w.object_pos(obj.name)[2] == pytest.approx(obj.half_height, abs=0.002)


@pytest.mark.assets
@pytest.mark.parametrize("task_name", TASKS)
def test_no_task_is_solved_at_start(task_name):
    w = World(RANKING)
    w.reset(RANKING.layout(0))
    Arm(w).reset_ready()
    assert not RANKING.tasks[task_name].success(w)


@pytest.mark.assets
@pytest.mark.parametrize("task_name,seed", [(t, n) for t in TASKS for n in range(5)])
def test_solution_solves_task(task_name, seed):
    task = RANKING.tasks[task_name]
    w = World(RANKING)
    w.reset(RANKING.layout(seed))
    arm = Arm(w)
    arm.reset_ready()
    for action in task.solution(w):
        arm.act(action)
    assert task.success(w)


def _posed(order, lift: float = 0.0) -> World:
    """A world with the cubes in slots 1, 2 and 3 in the given order, the first one raised by lift."""
    w = World(RANKING)
    w.reset(RANKING.layout(0))
    half = {o.name: o.half_height for o in RANKING.objects}
    for i, (name, y) in enumerate(zip(order, SLOTS, strict=True)):
        adr = w.model.jnt_qposadr[w.model.joint(name).id]
        w.data.qpos[adr:adr + 7] = [SLOT_X, y, half[name] + (lift if i == 0 else 0.0), 1, 0, 0, 0]
    mujoco.mj_forward(w.model, w.data)
    assert w.gripper_contacts() == []
    return w


RGB, SIZE = ("red_cube", "green_cube", "blue_cube"), ("green_cube", "blue_cube", "red_cube")
# (task, order in slots 1 to 3, lift of the first cube, expected)
CHECKS = [
    ("rank_rgb", RGB, 0.0, True),
    ("rank_rgb", SIZE, 0.0, False),
    ("rank_rgb", RGB, 0.03, False),  # the red cube is above its slot, not in it
    ("rank_size", SIZE, 0.0, True),
    ("rank_size", RGB, 0.0, False),
]


@pytest.mark.assets
@pytest.mark.parametrize("task_name,order,lift,expected", CHECKS)
def test_success_check_on_set_poses(task_name, order, lift, expected):
    assert RANKING.tasks[task_name].success(_posed(order, lift)) is expected
