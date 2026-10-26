"""The shelf scene: three cubes and a fixed two-level shelf."""
import re

import mujoco
import numpy as np
import pytest

from plan_and_act.scenes.shelf import SHELF
from plan_and_act.sim.arm import Arm
from plan_and_act.sim.world import World
from tests.test_plan_view import _entity_aabb, _geom_aabb

TASKS = list(SHELF.tasks)


def test_tasks_and_text():
    assert TASKS == ["red_on_top_shelf", "all_on_shelf"]
    assert SHELF.fixtures_text in SHELF.describe()


@pytest.mark.assets
def test_model_loads_with_every_object():
    w = World(SHELF)
    w.reset(SHELF.layout(0))
    for obj in SHELF.objects:
        assert w.object_pos(obj.name)[2] == pytest.approx(obj.half_height, abs=1e-6)


@pytest.mark.assets
def test_fixture_list_agrees_with_the_fixture_xml():
    names = re.findall(r'<geom name="([^"]+)"', SHELF.fixtures_xml)
    assert sorted(e.name for e in SHELF.fixtures) == sorted(names)
    w = World(SHELF)
    mujoco.mj_forward(w.model, w.data)
    for e in SHELF.fixtures:
        lo, hi = _geom_aabb(w.model, w.data, w.model.geom(e.name).id)
        elo, ehi = _entity_aabb(e)
        assert np.allclose(lo, elo, atol=5e-4) and np.allclose(hi, ehi, atol=5e-4), e.name


@pytest.mark.assets
def test_fixtures_are_outside_the_region():
    m = World(SHELF).model
    grow = max(o.footprint for o in SHELF.objects)
    x0, x1, y0, y1 = SHELF.region
    geoms = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name == "shelf"]
    assert len(geoms) == 5
    for g in geoms:
        cx, cy = m.body_pos[m.geom_bodyid[g]][:2] + m.geom_pos[g][:2]
        hx, hy = m.geom_size[g][:2]
        assert cx + hx < x0 - grow or cx - hx > x1 + grow or cy + hy < y0 - grow or cy - hy > y1 + grow


@pytest.mark.assets
def test_objects_stay_still_at_their_start_poses():
    w = World(SHELF)
    for seed in range(2):
        w.reset(SHELF.layout(seed))
        w.step(200)
        for obj in SHELF.objects:
            np.testing.assert_allclose(w.object_pos(obj.name)[:2], SHELF.layout(seed)[obj.name][:2], atol=0.002)
            assert w.object_pos(obj.name)[2] == pytest.approx(obj.half_height, abs=0.002)


@pytest.mark.assets
@pytest.mark.parametrize("task_name", TASKS)
def test_no_task_is_solved_at_start(task_name):
    w = World(SHELF)
    w.reset(SHELF.layout(0))
    Arm(w).reset_ready()
    assert not SHELF.tasks[task_name].success(w)


@pytest.mark.assets
@pytest.mark.parametrize("task_name,seed", [(t, n) for t in TASKS for n in range(5)])
def test_solution_solves_task(task_name, seed):
    task = SHELF.tasks[task_name]
    w = World(SHELF)
    w.reset(SHELF.layout(seed))
    arm = Arm(w)
    arm.reset_ready()
    for action in task.solution(w):
        arm.act(action)
    assert task.success(w)


def _posed(poses: dict) -> World:
    """A world with each cube at the given centre (x, y, z). The arm stays in its home pose, high above."""
    w = World(SHELF)
    w.reset({n: p[:2] for n, p in poses.items()})
    for name, p in poses.items():
        adr = w.model.jnt_qposadr[w.model.joint(name).id]
        w.data.qpos[adr:adr + 7] = [*p, 1, 0, 0, 0]
    mujoco.mj_forward(w.model, w.data)
    assert w.gripper_contacts() == []
    return w


TOP, BOTTOM = (0.70, 0.0, 0.12), (0.60, 0.0, 0.06)
AWAY = {"blue_cube": (0.42, -0.22, 0.02), "green_cube": (0.42, 0.22, 0.02)}
# (task, poses, expected)
CHECKS = [
    ("red_on_top_shelf", {"red_cube": TOP, **AWAY}, True),
    ("red_on_top_shelf", {"red_cube": BOTTOM, **AWAY}, False),  # on the bottom level
    ("red_on_top_shelf", {"red_cube": (0.70, 0.0, 0.16), **AWAY}, False),  # above the top level
    ("red_on_top_shelf", {"red_cube": (0.70, 0.25, 0.02), **AWAY}, False),  # beside the shelf
    ("all_on_shelf", {"red_cube": TOP, "blue_cube": (0.60, -0.08, 0.06), "green_cube": (0.60, 0.08, 0.06)}, True),
    ("all_on_shelf", {"red_cube": TOP, "blue_cube": (0.60, -0.08, 0.06), "green_cube": (0.42, 0.22, 0.02)}, False),
]


@pytest.mark.assets
@pytest.mark.parametrize("task_name,poses,expected", CHECKS)
def test_success_check_on_set_poses(task_name, poses, expected):
    assert SHELF.tasks[task_name].success(_posed(poses)) is expected
