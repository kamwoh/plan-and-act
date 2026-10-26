"""The bucket scene: a free bucket with a handle, and a food box."""
import re

import mujoco
import numpy as np
import pytest

from plan_and_act.scenes.bucket import BUCKET, CX, CY
from plan_and_act.sim.arm import Arm
from plan_and_act.sim.world import World
from tests.test_plan_view import _entity_aabb, _geom_aabb

TASKS = list(BUCKET.tasks)


def test_tasks_and_layout():
    assert TASKS == ["food_in_bucket", "lift_bucket"]
    x0, x1, y0, y1 = BUCKET.region
    for seed in range(20):
        x, y, _ = BUCKET.layout(seed)["food_box"]
        assert x0 <= x <= x1 and y0 <= y <= y1
    assert "handle" in BUCKET.describe()


@pytest.mark.assets
def test_model_loads_and_the_bucket_is_free():
    w = World(BUCKET)
    w.reset(BUCKET.layout(0))
    m = w.model
    assert m.jnt_type[m.joint("bucket").id] == mujoco.mjtJoint.mjJNT_FREE
    assert w.object_pos("food_box")[2] == pytest.approx(BUCKET.objects[0].half_height, abs=1e-6)
    np.testing.assert_allclose(w.data.body("bucket").xpos, (CX, CY, 0.0), atol=1e-6)


@pytest.mark.assets
def test_fixture_list_agrees_with_the_fixture_xml():
    names = re.findall(r'<geom name="([^"]+)"', BUCKET.fixtures_xml)
    assert sorted(e.name for e in BUCKET.fixtures) == sorted(names)
    w = World(BUCKET)
    w.reset(BUCKET.layout(0))
    for e in BUCKET.fixtures:
        lo, hi = _geom_aabb(w.model, w.data, w.model.geom(e.name).id)
        elo, ehi = _entity_aabb(e)
        assert np.allclose(lo, elo, atol=5e-4) and np.allclose(hi, ehi, atol=5e-4), e.name


@pytest.mark.assets
def test_bucket_keeps_clear_of_the_region():
    w = World(BUCKET)
    w.reset(BUCKET.layout(0))
    grow = BUCKET.objects[0].footprint
    x0, x1, y0, y1 = BUCKET.region
    for e in BUCKET.fixtures:
        lo, hi = _entity_aabb(e)
        assert hi[0] < x0 - grow or lo[0] > x1 + grow or hi[1] < y0 - grow or lo[1] > y1 + grow, e.name


@pytest.mark.assets
def test_objects_and_bucket_stay_still_at_the_start():
    w = World(BUCKET)
    for seed in range(2):
        w.reset(BUCKET.layout(seed))
        w.step(200)
        np.testing.assert_allclose(w.object_pos("food_box")[:2], BUCKET.layout(seed)["food_box"][:2], atol=0.002)
        assert w.object_pos("food_box")[2] == pytest.approx(BUCKET.objects[0].half_height, abs=0.002)
        np.testing.assert_allclose(w.data.body("bucket").xpos, (CX, CY, 0.0), atol=0.002)


@pytest.mark.assets
@pytest.mark.parametrize("task_name", TASKS)
def test_no_task_is_solved_at_start(task_name):
    w = World(BUCKET)
    w.reset(BUCKET.layout(0))
    Arm(w).reset_ready()
    assert not BUCKET.tasks[task_name].success(w)


@pytest.mark.assets
@pytest.mark.parametrize("task_name,seed", [(t, n) for t in TASKS for n in range(5)])
def test_solution_solves_task(task_name, seed):
    task = BUCKET.tasks[task_name]
    w = World(BUCKET)
    w.reset(BUCKET.layout(seed))
    arm = Arm(w)
    arm.reset_ready()
    for action in task.solution(w):
        arm.act(action)
    assert task.success(w)


def _posed(food, lift: float = 0.0) -> World:
    """The food box at its centre (x, y, z) and the bucket raised by lift. The arm stays in its home pose."""
    w = World(BUCKET)
    w.reset(BUCKET.layout(0))
    for name, p in (("food_box", food), ("bucket", (CX, CY, lift))):
        adr = w.model.jnt_qposadr[w.model.joint(name).id]
        w.data.qpos[adr:adr + 7] = [*p, 1, 0, 0, 0]
    mujoco.mj_forward(w.model, w.data)
    assert w.gripper_contacts() == []
    return w


INSIDE = (CX, CY, 0.025)  # on the bucket floor
# (task, food centre, bucket lift, expected)
CHECKS = [
    ("food_in_bucket", INSIDE, 0.0, True),
    ("food_in_bucket", (CX, CY + 0.10, 0.02), 0.0, False),  # beside the bucket
    ("food_in_bucket", (CX, CY, 0.07), 0.0, False),  # above the rim
    ("lift_bucket", INSIDE, 0.0, False),  # the bucket is still on the table
    ("lift_bucket", (CX, CY, 0.025 + 0.08), 0.08, True),
    ("lift_bucket", (0.45, 0.0, 0.02), 0.08, False),  # the food is on the table
]


@pytest.mark.assets
@pytest.mark.parametrize("task_name,food,lift,expected", CHECKS)
def test_success_check_on_set_poses(task_name, food, lift, expected):
    assert BUCKET.tasks[task_name].success(_posed(food, lift)) is expected
