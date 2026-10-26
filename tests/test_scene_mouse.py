import re

import mujoco
import numpy as np
import pytest

from plan_and_act.scenes.mouse import (
    CLICK,
    CLICK_ON_PAD,
    MOUSE,
    MOUSE_X,
    MOUSE_Y,
    PAD_HX,
    PAD_X,
    PAD_Y,
    SHELL_HZ,
    TRAVEL,
    click_at,
)
from plan_and_act.sim.arm import Arm
from plan_and_act.sim.world import World
from plan_and_act.types import Action
from tests.test_plan_view import _entity_aabb, _geom_aabb

TASKS = list(MOUSE.tasks)
ON_PAD = (PAD_X, PAD_Y, SHELL_HZ)
OFF_PAD = (PAD_X + PAD_HX + 0.01, PAD_Y, SHELL_HZ)


def _run(actions) -> World:
    w = World(MOUSE)
    w.reset(MOUSE.layout(0))
    arm = Arm(w)
    arm.reset_ready()
    for a in actions:
        arm.act(a)
    return w


def _put(w: World, xyz, quat=(1, 0, 0, 0)) -> None:
    """Puts the mouse at rest at the given centre and orientation."""
    j = w.model.joint("mouse")
    adr, dof = w.model.jnt_qposadr[j.id], w.model.jnt_dofadr[j.id]
    w.data.qpos[adr:adr + 7] = [*xyz, *quat]
    w.data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(w.model, w.data)


def _click_now(w: World) -> None:
    """Pushes the left button to the bottom by hand and takes one step, so that the monitor sees a click."""
    w.data.qpos[w.model.jnt_qposadr[w.model.joint("mouse_button").id]] = TRAVEL
    mujoco.mj_forward(w.model, w.data)
    w.step()


@pytest.mark.assets
def test_model_loads():
    w = World(MOUSE)
    w.reset(MOUSE.layout(0))
    assert set(w.joints()) == {"mouse_button"}
    np.testing.assert_allclose(w.object_pos("mouse"), (MOUSE_X, MOUSE_Y, SHELL_HZ), atol=1e-6)


@pytest.mark.assets
def test_fixture_list_agrees_with_the_fixture_xml():
    names = re.findall(r'<geom name="([^"]+)"', MOUSE.fixtures_xml)
    assert sorted(e.name for e in MOUSE.fixtures) == sorted(names)
    w = World(MOUSE)
    mujoco.mj_forward(w.model, w.data)
    for e in MOUSE.fixtures:
        lo, hi = _geom_aabb(w.model, w.data, w.model.geom(e.name).id)
        elo, ehi = _entity_aabb(e)
        assert np.allclose(lo, elo, atol=5e-4) and np.allclose(hi, ehi, atol=5e-4), e.name


@pytest.mark.assets
def test_fixtures_are_outside_the_region():
    m = World(MOUSE).model
    x0, x1, y0, y1 = MOUSE.region
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        if b == 0 or m.body_parentid[b] != 0 or m.body_jntnum[b] or m.body(b).name == "link0":
            continue
        cx, cy = m.body_pos[b][:2] + m.geom_pos[g][:2]
        hx, hy = m.geom_size[g][:2]
        assert cx + hx < x0 or cx - hx > x1 or cy + hy < y0 or cy - hy > y1, m.geom(g).name


@pytest.mark.assets
def test_the_pad_does_not_collide():
    m = World(MOUSE).model
    g = m.geom("mouse_pad").id
    assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0


@pytest.mark.assets
def test_the_mouse_and_its_button_stay_still_at_the_start():
    w = _run([Action("wait", seconds=2.0)])
    np.testing.assert_allclose(w.object_pos("mouse"), (MOUSE_X, MOUSE_Y, SHELL_HZ), atol=0.002)
    assert abs(w.joint_pos("mouse_button")) < 5e-4
    assert w.events == []


@pytest.mark.assets
@pytest.mark.parametrize("task_name", TASKS)
def test_no_task_is_solved_at_start(task_name):
    task = MOUSE.tasks[task_name]
    w = _run([])
    assert not task.success(w)


@pytest.mark.assets
@pytest.mark.parametrize("task_name,seed", [(t, n) for t in TASKS for n in range(5)])
def test_solution_solves_task(task_name, seed):
    task = MOUSE.tasks[task_name]
    w = World(MOUSE)
    w.reset(MOUSE.layout(seed), task.initial_qpos)
    arm = Arm(w)
    arm.reset_ready()
    for a in task.solution(w):
        arm.act(a)
    assert task.success(w)


# A quarter turn about x: the mouse lies on its side.
ON_SIDE = (np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0)


@pytest.mark.assets
@pytest.mark.parametrize("xyz,quat,expected", [(ON_PAD, (1, 0, 0, 0), True), (OFF_PAD, (1, 0, 0, 0), False),
                                               ((PAD_X, PAD_Y, 0.03), ON_SIDE, False)])
def test_mouse_on_pad_check_on_set_poses(xyz, quat, expected):
    w = _run([])
    _put(w, xyz, quat)
    assert MOUSE.tasks["mouse_on_pad"].success(w) is expected


@pytest.mark.assets
@pytest.mark.parametrize("xyz,event,expected", [(ON_PAD, CLICK_ON_PAD, True), (OFF_PAD, CLICK, False)])
def test_click_event_says_where_the_mouse_is(xyz, event, expected):
    w = _run([])
    _put(w, xyz)
    _click_now(w)
    assert w.events == [event]
    assert MOUSE.tasks["click_on_pad"].success(w) is expected


@pytest.mark.assets
def test_a_click_off_the_pad_does_not_count_after_a_move_onto_it():
    w = _run([])
    _click_now(w)
    assert w.events == [CLICK]
    _put(w, ON_PAD)
    assert MOUSE.tasks["mouse_on_pad"].success(w)
    assert not MOUSE.tasks["click_on_pad"].success(w)


@pytest.mark.assets
def test_the_scripted_click_gives_one_event_and_the_button_comes_back_up():
    w = _run(click_at(MOUSE_X, MOUSE_Y))
    assert w.events == [CLICK]
    w.step(200)
    assert abs(w.joint_pos("mouse_button")) < 5e-4


@pytest.mark.assets
def test_carrying_the_mouse_does_not_click_it():
    w = _run(MOUSE.tasks["mouse_on_pad"].solution(World(MOUSE)))
    assert w.events == []
