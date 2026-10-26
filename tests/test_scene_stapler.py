import re

import mujoco
import numpy as np
import pytest

from plan_and_act.scenes.stapler import FIRE, PRESS_X, PRESS_Z, STAPLER, press
from plan_and_act.sim.arm import Arm
from plan_and_act.sim.world import World
from plan_and_act.types import Action
from tests.test_plan_view import _entity_aabb, _geom_aabb

TASKS = list(STAPLER.tasks)


def _world(qpos=None) -> World:
    w = World(STAPLER)
    w.reset(STAPLER.layout(0), qpos)
    return w


def _run(actions, qpos=None) -> World:
    w = _world(qpos)
    arm = Arm(w)
    arm.reset_ready()
    for a in actions:
        arm.act(a)
    return w


def _set(w: World, q: float) -> None:
    """Puts the arm at angle q at rest, then takes one step, so that the monitor sees it."""
    j = w.model.joint("stapler_arm")
    w.data.qpos[w.model.jnt_qposadr[j.id]] = q
    w.data.qvel[w.model.jnt_dofadr[j.id]] = 0.0
    mujoco.mj_forward(w.model, w.data)
    w.step()


@pytest.mark.assets
def test_model_loads():
    w = _world()
    assert set(w.joints()) == {"stapler_arm"}
    assert w.joint_pos("stapler_arm") == pytest.approx(0.0, abs=1e-3)


@pytest.mark.assets
def test_fixture_list_agrees_with_the_fixture_xml():
    names = re.findall(r'<geom name="([^"]+)"', STAPLER.fixtures_xml)
    assert sorted(e.name for e in STAPLER.fixtures) == sorted(names)
    w = World(STAPLER)
    mujoco.mj_forward(w.model, w.data)
    for e in STAPLER.fixtures:
        lo, hi = _geom_aabb(w.model, w.data, w.model.geom(e.name).id)
        elo, ehi = _entity_aabb(e)
        assert np.allclose(lo, elo, atol=5e-4) and np.allclose(hi, ehi, atol=5e-4), e.name


@pytest.mark.assets
def test_fixtures_are_outside_the_region():
    m = World(STAPLER).model
    x0, x1, y0, y1 = STAPLER.region
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        if b == 0 or m.body_parentid[b] != 0 or m.body_jntnum[b] or m.body(b).name == "link0":
            continue
        cx, cy = m.body_pos[b][:2] + m.geom_pos[g][:2]
        hx, hy = m.geom_size[g][:2]
        assert cx + hx < x0 or cx - hx > x1 or cy + hy < y0 or cy - hy > y1, m.geom(g).name


@pytest.mark.assets
def test_the_arm_stays_up_when_nothing_touches_it():
    w = _run([Action("wait", seconds=2.0)])
    assert w.joint_pos("stapler_arm") == pytest.approx(0.0, abs=2e-3)
    assert w.events == []


@pytest.mark.assets
@pytest.mark.parametrize("task_name", TASKS)
def test_no_task_is_solved_at_start(task_name):
    task = STAPLER.tasks[task_name]
    w = _run([], task.initial_qpos)
    assert not task.success(w)


@pytest.mark.assets
@pytest.mark.parametrize("task_name,seed", [(t, n) for t in TASKS for n in range(5)])
def test_solution_solves_task(task_name, seed):
    task = STAPLER.tasks[task_name]
    w = World(STAPLER)
    w.reset(STAPLER.layout(seed), task.initial_qpos)
    arm = Arm(w)
    arm.reset_ready()
    for a in task.solution(w):
        arm.act(a)
    assert task.success(w)


@pytest.mark.assets
@pytest.mark.parametrize("q,expected", [(FIRE + 0.02, True), (FIRE - 0.02, False)])
def test_press_check_on_set_angles(q, expected):
    w = _world()
    _set(w, q)
    assert (w.events == ["stapler pressed"]) is expected
    assert STAPLER.tasks["press_stapler"].success(w) is expected


@pytest.mark.assets
def test_the_spring_lifts_the_arm_after_a_press():
    w = _run(press())
    assert w.events == ["stapler pressed"]
    w.step(500)
    assert w.joint_pos("stapler_arm") == pytest.approx(0.0, abs=2e-3)


@pytest.mark.assets
def test_a_light_push_does_not_press_the_stapler():
    w = _run([Action("close"), Action("move_to", (PRESS_X, 0.0, 0.15), yaw=0.0),
              Action("move_to", (PRESS_X, 0.0, PRESS_Z + 0.022))])
    assert "stapler_arm" in w.gripper_contacts()
    assert 0.0 < w.joint_pos("stapler_arm") < FIRE
    assert not STAPLER.tasks["press_stapler"].success(w)
