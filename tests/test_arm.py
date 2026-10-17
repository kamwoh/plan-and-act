import numpy as np
import pytest

from armmind.scenes import SCENES
from armmind.sim.arm import REACH_TOL, READY, Arm
from armmind.sim.world import World
from armmind.types import Action

LAYOUT = {"red_cube": (0.5, -0.1), "blue_cube": (0.45, 0.15), "green_cube": (0.62, 0.05)}

pytestmark = pytest.mark.assets


@pytest.fixture
def arm():
    w = World(SCENES["cubes"])
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
    x, y, _ = arm.w.object_pos("red_cube")
    for a in [Action("move_to", (x, y, 0.15)), Action("move_to", (x, y, 0.02)), Action("close")]:
        arm.act(a)
    res = arm.act(Action("move_to", (x, y, 0.2)))
    assert res.gripper_width == pytest.approx(0.04, abs=0.008)
    assert "red_cube" in res.contacts
    assert arm.w.object_pos("red_cube")[2] > 0.15


def test_wait_is_capped_and_frames_are_paced():
    w = World(SCENES["cubes"])
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


def _dist_to_segment(p, a, b) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def test_move_to_follows_a_straight_line():
    w = World(SCENES["cubes"])
    w.reset(LAYOUT)
    points = []
    a = Arm(w, on_frame=lambda: points.append(w.tcp().copy()))
    a.reset_ready()
    start, goal = w.tcp().copy(), np.array([0.6, 0.25, 0.12])
    points.clear()
    assert a.act(Action("move_to", tuple(goal))).status == "reached"
    assert len(points) > 10
    assert max(_dist_to_segment(p, start, goal) for p in points) < 0.01


def test_yaw_turns_the_gripper_and_back(arm):
    assert arm.w.gripper_yaw() == pytest.approx(0.0, abs=1.0)
    res = arm.act(Action("move_to", (0.5, 0.0, 0.2), yaw=45.0))
    assert res.status == "reached"
    assert arm.w.gripper_yaw() == pytest.approx(45.0, abs=3.0)
    assert res.gripper_yaw == pytest.approx(45.0, abs=3.0)
    assert "yaw 45" in res.summary()
    res = arm.act(Action("move_to", (0.5, 0.1, 0.2)))  # no yaw: keep 45
    assert arm.w.gripper_yaw() == pytest.approx(45.0, abs=3.0)
    res = arm.act(Action("move_to", (0.5, 0.0, 0.2), yaw=0.0))
    assert arm.w.gripper_yaw() == pytest.approx(0.0, abs=3.0)
    assert "yaw" not in res.summary().split(": ", 1)[1]  # the result part leaves out a yaw of 0


def test_pure_rotation_takes_time(arm):
    arm.act(Action("move_to", (0.5, 0.0, 0.2)))
    t0 = arm.w.time
    arm.act(Action("move_to", (0.5, 0.0, 0.2), yaw=90.0))
    assert arm.w.time - t0 >= 0.9


def test_yaw_outside_the_range_is_clamped(arm):
    res = arm.act(Action("move_to", (0.5, 0.0, 0.2), yaw=150.0))
    assert res.status == "reached"
    assert "yaw clamped to 90" in res.note
    assert arm.w.gripper_yaw() == pytest.approx(90.0, abs=3.0)


def test_path_that_leaves_the_reach_stops_as_unreachable(arm):
    # At yaw 90 the wrist cannot follow the whole sweep around the base: joint 7 runs out of range.
    assert arm.act(Action("move_to", (0.25, 0.4, 0.01), yaw=90.0)).status == "reached"
    res = arm.act(Action("move_to", (0.25, -0.3, 0.3), yaw=150.0))
    assert res.status == "unreachable"
    assert res.note == "yaw clamped to 90, path leaves the reachable area"
    assert -0.25 < res.tcp[1] < 0.35  # it stopped on the way
    assert arm._yaw == pytest.approx(90.0)  # the yaw of the last good point
    held = arm.w.tcp().copy()
    arm.act(Action("wait", seconds=0.5))
    np.testing.assert_allclose(arm.w.tcp(), held, atol=0.005)


FLIP_POINTS = [(0.6, 0.0, 0.02), (0.4, 0.2, 0.3), (0.35, -0.3, 0.1), (0.65, 0.25, 0.05), (0.5, -0.2, 0.4)]


def test_yaw_flip_with_a_long_move_is_reached():
    """+-90 to -+90 together with a long move: the IK must not give up near a 180 degree turn."""
    import itertools
    w = World(SCENES["cubes"])
    failed = []
    for a, b in itertools.permutations(FLIP_POINTS, 2):
        for yaw in (90.0, -90.0):
            w.reset(LAYOUT)
            arm = Arm(w)
            arm.reset_ready()
            first = arm.act(Action("move_to", a, yaw=yaw)).status
            second = arm.act(Action("move_to", b, yaw=-yaw))
            if (first, second.status) != ("reached", "reached") or abs(w.gripper_yaw() + yaw) > 3:
                failed.append((a, yaw, b, first, second.status, second.note))
    assert failed == []


def test_reviewer_flip_example(arm):
    assert arm.act(Action("move_to", (0.6, 0.0, 0.02), yaw=90.0)).status == "reached"
    res = arm.act(Action("move_to", (0.4, 0.2, 0.3), yaw=-90.0))
    assert res.status == "reached"
    assert res.gripper_yaw == pytest.approx(-90.0, abs=3.0)


def test_a_stopped_turn_keeps_the_yaw_it_holds(arm):
    assert arm.act(Action("move_to", (0.25, 0.3, 0.3), yaw=-90.0)).status == "reached"
    res = arm.act(Action("move_to", (0.25, -0.4, 0.01), yaw=90.0))
    assert res.status == "unreachable" and res.note == "path leaves the reachable area"
    assert -60 < arm.w.gripper_yaw() < 60  # it stopped part way through the turn
    assert arm.act(Action("wait", seconds=0.3)).gripper_yaw == pytest.approx(arm._yaw, abs=3.0)


def test_results_carry_the_held_objects_before_and_after(arm):
    for a in (Action("open"), Action("move_to", (0.5, -0.1, 0.12)), Action("move_to", (0.5, -0.1, 0.02))):
        arm.act(a)
    grab = arm.act(Action("close"))
    assert grab.held_before == () and grab.held_after == ("red_cube",)
    assert "now holding red_cube" in grab.summary()
    up = arm.act(Action("move_to", (0.5, -0.1, 0.12)))
    assert up.held_before == up.held_after == ("red_cube",)
    let_go = arm.act(Action("open"))
    assert let_go.held_before == ("red_cube",) and "table" not in let_go.held_after
