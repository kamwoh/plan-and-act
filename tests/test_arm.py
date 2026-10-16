import numpy as np
import pytest

from armmind.sim.arm import REACH_TOL, READY, Arm
from armmind.sim.world import World
from armmind.types import Action

LAYOUT = {"red_cube": (0.5, -0.1), "blue_cube": (0.45, 0.15), "green_cube": (0.62, 0.05)}

pytestmark = pytest.mark.assets


@pytest.fixture
def arm():
    w = World()
    w.reset(LAYOUT)
    a = Arm(w)
    a.reset_ready()
    return a


def test_reset_ready_pose(arm):
    np.testing.assert_allclose(arm.w.tcp(), READY, atol=0.01)
    assert arm.w.gripper_width() == pytest.approx(0.08, abs=0.005)


def test_move_to_reaches_target(arm):
    res = arm.act(Action("move_to", (0.5, 0.2, 0.2)))
    assert res.status == "reached"
    assert np.linalg.norm(np.array(res.tcp) - [0.5, 0.2, 0.2]) <= REACH_TOL


def test_move_to_clamps_to_workspace(arm):
    res = arm.act(Action("move_to", (0.5, 0.0, 0.9)))
    assert res.status == "reached"
    assert "clamped" in res.note
    assert res.tcp[2] == pytest.approx(0.5, abs=REACH_TOL)


def test_unreachable_target_does_not_move(arm):
    before = arm.w.tcp().copy()
    t0 = arm.w.time
    res = arm.act(Action("move_to", (0.75, 0.4, 0.01)))
    assert res.status == "unreachable"
    assert arm.w.time == t0
    np.testing.assert_allclose(arm.w.tcp(), before)


def test_close_on_nothing_and_open(arm):
    assert arm.act(Action("close")).gripper_width < 0.01
    assert arm.act(Action("open")).gripper_width > 0.07


def test_grasp_and_lift_red_cube(arm):
    x, y, _ = arm.w.cube_pos("red_cube")
    for a in [Action("move_to", (x, y, 0.15)), Action("move_to", (x, y, 0.02)), Action("close")]:
        arm.act(a)
    res = arm.act(Action("move_to", (x, y, 0.2)))
    assert res.gripper_width == pytest.approx(0.04, abs=0.008)
    assert "red_cube" in res.contacts
    assert arm.w.cube_pos("red_cube")[2] > 0.15


def test_wait_is_capped_and_frames_are_paced():
    w = World()
    w.reset(LAYOUT)
    frames = []
    a = Arm(w, on_frame=lambda: frames.append(w.time))
    a.reset_ready()
    frames.clear()
    t0 = w.time
    a.act(Action("wait", seconds=10.0))
    assert w.time - t0 == pytest.approx(3.0, abs=0.01)
    assert 88 <= len(frames) <= 92


def test_clamped_and_unreachable_note_says_both(arm):
    res = arm.act(Action("move_to", (0.9, 0.5, 0.0)))
    assert res.status == "unreachable"
    assert res.note == "target clamped to the workspace, and no joint solution for this point"
