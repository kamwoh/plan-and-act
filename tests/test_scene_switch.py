import re

import mujoco
import numpy as np
import pytest

from plan_and_act.scenes.switch import BOX_X, BOX_Y, ON, PRESS_DX, PRESS_Z, SWITCH, TILT, flip
from plan_and_act.sim.arm import Arm
from plan_and_act.sim.world import World
from plan_and_act.types import Action
from tests.test_plan_view import _entity_aabb, _geom_aabb

TASKS = list(SWITCH.tasks)


def _run(actions, qpos=None) -> World:
    w = World(SWITCH)
    w.reset(SWITCH.layout(0), qpos)
    arm = Arm(w)
    arm.reset_ready()
    for a in actions:
        arm.act(a)
    return w


def _set(w: World, q: float) -> None:
    """Puts the rocker at angle q at rest."""
    j = w.model.joint("switch")
    w.data.qpos[w.model.jnt_qposadr[j.id]] = q
    w.data.qvel[w.model.jnt_dofadr[j.id]] = 0.0
    mujoco.mj_forward(w.model, w.data)


@pytest.mark.assets
def test_model_loads():
    w = _run([])
    assert set(w.joints()) == {"switch"}
    assert w.joint_pos("switch") == pytest.approx(0.0, abs=1e-3)


@pytest.mark.assets
def test_fixture_list_agrees_with_the_fixture_xml():
    names = re.findall(r'<geom name="([^"]+)"', SWITCH.fixtures_xml)
    assert sorted(e.name for e in SWITCH.fixtures) == sorted(names)
    w = World(SWITCH)
    mujoco.mj_forward(w.model, w.data)
    for e in SWITCH.fixtures:
        lo, hi = _geom_aabb(w.model, w.data, w.model.geom(e.name).id)
        elo, ehi = _entity_aabb(e)
        assert np.allclose(lo, elo, atol=5e-4) and np.allclose(hi, ehi, atol=5e-4), e.name


@pytest.mark.assets
def test_fixtures_are_outside_the_region():
    m = World(SWITCH).model
    x0, x1, y0, y1 = SWITCH.region
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        if b == 0 or m.body_parentid[b] != 0 or m.body_jntnum[b] or m.body(b).name == "link0":
            continue
        cx, cy = m.body_pos[b][:2] + m.geom_pos[g][:2]
        hx, hy = m.geom_size[g][:2]
        assert cx + hx < x0 or cx - hx > x1 or cy + hy < y0 or cy - hy > y1, m.geom(g).name


@pytest.mark.assets
@pytest.mark.parametrize("q", [0.0, ON])
def test_the_switch_stays_at_each_end(q):
    w = _run([Action("wait", seconds=2.0)], {"switch": q})
    assert w.joint_pos("switch") == pytest.approx(q, abs=3e-3)


@pytest.mark.assets
@pytest.mark.parametrize("q,end", [(0.8 * TILT, 0.0), (1.2 * TILT, ON)])
def test_a_rocker_left_between_the_ends_falls_to_the_nearer_end(q, end):
    w = _run([])
    _set(w, q)
    w.step(1500)
    assert w.joint_pos("switch") == pytest.approx(end, abs=3e-3)


@pytest.mark.assets
@pytest.mark.parametrize("task_name", TASKS)
def test_no_task_is_solved_at_start(task_name):
    task = SWITCH.tasks[task_name]
    w = _run([], task.initial_qpos)
    assert not task.success(w)


@pytest.mark.assets
@pytest.mark.parametrize("task_name,seed", [(t, n) for t in TASKS for n in range(5)])
def test_solution_solves_task(task_name, seed):
    task = SWITCH.tasks[task_name]
    w = World(SWITCH)
    w.reset(SWITCH.layout(seed), task.initial_qpos)
    arm = Arm(w)
    arm.reset_ready()
    for a in task.solution(w):
        arm.act(a)
    assert task.success(w)


@pytest.mark.assets
@pytest.mark.parametrize("q,on,off", [(ON, True, False), (0.0, False, True), (TILT, False, False)])
def test_success_checks_on_set_angles(q, on, off):
    w = _run([])
    _set(w, q)
    assert SWITCH.tasks["switch_on"].success(w) is on
    assert SWITCH.tasks["switch_off"].success(w) is off


@pytest.mark.assets
def test_switch_on_check_rejects_a_gripper_on_the_rocker():
    """The plan without its last move holds the rocker down: the check passes only after the gripper goes up."""
    plan = flip(1.0)
    w = _run(plan[:-1])
    assert "switch_rocker" in w.gripper_contacts()
    assert not SWITCH.tasks["switch_on"].success(w)


@pytest.mark.assets
def test_a_push_on_the_low_end_does_not_turn_the_switch_on():
    x = BOX_X - PRESS_DX  # the near end, which is already down when the switch is off
    w = _run([Action("close"), Action("move_to", (x, BOX_Y, 0.15), yaw=90.0), Action("move_to", (x, BOX_Y, PRESS_Z)),
              Action("move_to", (x, BOX_Y, 0.15))])
    assert w.joint_pos("switch") == pytest.approx(0.0, abs=3e-3)
    assert not SWITCH.tasks["switch_on"].success(w)
