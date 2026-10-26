"""The peg-in-hole scene: the model, the fixed block, the pegs at the start, the checks and the plans."""
import re

import mujoco
import numpy as np
import pytest

from plan_and_act.scenes.peg import BX, HOLES, PEG, PEG_HALF_H
from plan_and_act.sim.arm import Arm
from plan_and_act.sim.world import World

TASKS = list(PEG.tasks)


def _geom_aabb(model, data, g):
    c, half = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half + c
    world = corners @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
    return world.min(0), world.max(0)


def _posed(poses: dict) -> World:
    """A world with each peg at a centre (x, y, z) and a quaternion. The arm stays in its home pose, high above."""
    w = World(PEG)
    w.reset({n: p[:2] for n, (p, _) in poses.items()})
    for name, (p, quat) in poses.items():
        adr = w.model.jnt_qposadr[w.model.joint(name).id]
        w.data.qpos[adr:adr + 7] = [*p, *quat]
    mujoco.mj_forward(w.model, w.data)
    assert w.gripper_contacts() == []
    return w


UP = (1, 0, 0, 0)
TILT = (np.cos(np.radians(15)), np.sin(np.radians(15)), 0, 0)  # 30 degrees about x
LEFT, RIGHT = (BX, HOLES["left"], PEG_HALF_H), (BX, HOLES["right"], PEG_HALF_H)
AWAY = (0.42, 0.20, PEG_HALF_H)


def test_tasks_and_objects():
    assert TASKS == ["insert_peg", "fill_both_holes"]
    assert [o.name for o in PEG.objects] == ["red_peg", "blue_peg"]
    assert all(max(o.dims[:2]) <= 0.07 for o in PEG.objects)  # the open gripper fits around each peg


@pytest.mark.assets
def test_model_loads_with_the_pegs_standing():
    w = World(PEG)
    w.reset(PEG.layout(0))
    for obj in PEG.objects:
        assert w.object_pos(obj.name)[2] == pytest.approx(obj.half_height, abs=1e-6)


@pytest.mark.assets
def test_fixture_list_agrees_with_the_fixture_xml():
    names = re.findall(r'<geom name="([^"]+)"', PEG.fixtures_xml)
    assert sorted(e.name for e in PEG.fixtures) == sorted(names)
    w = World(PEG)
    mujoco.mj_forward(w.model, w.data)
    for e in PEG.fixtures:
        lo, hi = _geom_aabb(w.model, w.data, w.model.geom(e.name).id)
        half = np.array(e.size) / 2
        assert np.allclose(lo, np.array(e.xyz) - half, atol=5e-4), e.name
        assert np.allclose(hi, np.array(e.xyz) + half, atol=5e-4), e.name


@pytest.mark.assets
def test_fixtures_are_outside_the_region():
    w = World(PEG)
    m = w.model
    grow = max(o.footprint for o in PEG.objects)
    x0, x1, y0, y1 = PEG.region
    static = [g for g in range(m.ngeom) if (b := m.geom_bodyid[g]) and m.body_parentid[b] == 0
              and not m.body_jntnum[b] and m.body(b).name != "link0"]
    assert len(static) == 5
    for g in static:
        cx, cy = m.body_pos[m.geom_bodyid[g]][:2] + m.geom_pos[g][:2]
        hx, hy = m.geom_size[g][:2]
        assert cx + hx < x0 - grow or cx - hx > x1 + grow or cy + hy < y0 - grow or cy - hy > y1 + grow


@pytest.mark.assets
def test_objects_stay_still_at_their_start_poses():
    w = World(PEG)
    for seed in range(2):
        w.reset(PEG.layout(seed))
        w.step(200)
        for obj in PEG.objects:
            np.testing.assert_allclose(w.object_pos(obj.name)[:2], PEG.layout(seed)[obj.name][:2], atol=0.002)
            assert w.object_pos(obj.name)[2] == pytest.approx(obj.half_height, abs=0.002)


@pytest.mark.assets
@pytest.mark.parametrize("task_name", TASKS)
def test_no_task_is_solved_at_start(task_name):
    task = PEG.tasks[task_name]
    w = World(PEG)
    w.reset(PEG.layout(0), task.initial_qpos)
    Arm(w).reset_ready()
    assert not task.success(w)


@pytest.mark.assets
@pytest.mark.parametrize("task_name,seed", [(t, n) for t in TASKS for n in range(5)])
def test_solution_solves_task(task_name, seed):
    task = PEG.tasks[task_name]
    w = World(PEG)
    w.reset(PEG.layout(seed), task.initial_qpos)
    arm = Arm(w)
    arm.reset_ready()
    for action in task.solution(w):
        arm.act(action)
    assert task.success(w)


# (task, poses, expected): each task has one set that passes, to show that the pose code works.
CHECK_CASES = [
    ("insert_peg", {"red_peg": (LEFT, UP), "blue_peg": (AWAY, UP)}, True),
    ("insert_peg", {"red_peg": (RIGHT, UP), "blue_peg": (AWAY, UP)}, True),
    ("insert_peg", {"red_peg": ((BX - 0.08, 0.0, PEG_HALF_H), UP), "blue_peg": (AWAY, UP)}, False),  # beside it
    ("insert_peg", {"red_peg": ((BX, HOLES["left"], PEG_HALF_H + 0.03), UP), "blue_peg": (AWAY, UP)},
     False),  # not deep enough
    ("insert_peg", {"red_peg": ((BX, HOLES["left"], PEG_HALF_H), TILT), "blue_peg": (AWAY, UP)}, False),
    ("fill_both_holes", {"red_peg": (LEFT, UP), "blue_peg": (RIGHT, UP)}, True),
    ("fill_both_holes", {"red_peg": (LEFT, UP), "blue_peg": (AWAY, UP)}, False),
]


@pytest.mark.assets
@pytest.mark.parametrize("task_name,poses,expected", CHECK_CASES)
def test_success_check_on_set_poses(task_name, poses, expected):
    w = _posed(poses)
    assert PEG.tasks[task_name].success(w) is expected
