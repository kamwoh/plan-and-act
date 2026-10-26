"""The Tower of Hanoi scene: the model, the fixed items, the rings at the start, the checks and the plans."""
import re

import mujoco
import numpy as np
import pytest

from plan_and_act.scenes.hanoi import BASE_X, HANOI, LAST_TWO, PEG_TOP, PEGS, RINGS, START, _level
from plan_and_act.sim.arm import Arm
from plan_and_act.sim.world import World
from plan_and_act.types import Action

TASKS = list(HANOI.tasks)


def _geom_aabb(model, data, g):
    c, half = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half + c
    world = corners @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
    return world.min(0), world.max(0)


def _set_rings(w: World, centres: dict) -> None:
    """Put each ring at a centre (x, y, z) through its slide joints. The arm stays high above."""
    for ring, xyz in centres.items():
        for axis, v, v0 in zip("xyz", xyz, START[ring], strict=True):
            w.data.qpos[w.model.jnt_qposadr[w.model.joint(f"{ring}_{axis}").id]] = v - v0
    mujoco.mj_forward(w.model, w.data)
    assert w.gripper_contacts() == []


def _on(peg: str, level: int, dx: float = 0.0, dz: float = 0.0) -> tuple[float, float, float]:
    return (BASE_X + dx, PEGS[peg], _level(level) + dz)


def test_tasks_and_start_offsets():
    assert TASKS == ["top_ring_to_right", "last_two_moves"]
    assert HANOI.objects == () and HANOI.layout(0) == {}
    assert set(LAST_TWO) == {f"{r}_{a}" for r in RINGS for a in "xyz"}
    assert "left peg" in HANOI.fixtures_text and "right peg" in HANOI.fixtures_text


@pytest.mark.assets
def test_model_loads_with_the_rings_stacked_on_the_left_peg():
    w = World(HANOI)
    w.reset({})
    for i, ring in enumerate(RINGS):
        np.testing.assert_allclose(w.object_pos(ring), (BASE_X, PEGS["left"], _level(i)), atol=1e-6)


@pytest.mark.assets
def test_fixture_list_agrees_with_the_fixture_xml():
    names = re.findall(r'<geom name="([^"]+)"', HANOI.fixtures_xml)
    assert sorted(e.name for e in HANOI.fixtures) == sorted(names)
    w = World(HANOI)
    mujoco.mj_forward(w.model, w.data)
    for e in HANOI.fixtures:
        lo, hi = _geom_aabb(w.model, w.data, w.model.geom(e.name).id)
        half = np.array(e.size) / 2
        assert np.allclose(lo, np.array(e.xyz) - half, atol=5e-4), e.name
        assert np.allclose(hi, np.array(e.xyz) + half, atol=5e-4), e.name


@pytest.mark.assets
def test_static_fixtures_are_outside_the_region():
    w = World(HANOI)
    m = w.model
    x0, x1, y0, y1 = HANOI.region
    static = [g for g in range(m.ngeom) if (b := m.geom_bodyid[g]) and m.body_parentid[b] == 0
              and not m.body_jntnum[b] and m.body(b).name != "link0"]
    assert len(static) == 4  # the base board and three pegs
    for g in static:
        cx, cy = m.body_pos[m.geom_bodyid[g]][:2] + m.geom_pos[g][:2]
        hx, hy = m.geom_size[g][:2]
        assert cx + hx < x0 or cx - hx > x1 or cy + hy < y0 or cy - hy > y1


@pytest.mark.assets
@pytest.mark.parametrize("task_name", TASKS)
def test_rings_stay_still_at_the_start(task_name):
    task = HANOI.tasks[task_name]
    w = World(HANOI)
    w.reset({}, task.initial_qpos)
    start = {r: w.object_pos(r) for r in RINGS}
    w.step(200)
    for ring in RINGS:
        np.testing.assert_allclose(w.object_pos(ring), start[ring], atol=0.002)


@pytest.mark.assets
@pytest.mark.parametrize("task_name", TASKS)
def test_no_task_is_solved_at_start(task_name):
    task = HANOI.tasks[task_name]
    w = World(HANOI)
    w.reset({}, task.initial_qpos)
    Arm(w).reset_ready()
    assert not task.success(w)


@pytest.mark.assets
@pytest.mark.parametrize("task_name,seed", [(t, n) for t in TASKS for n in range(5)])
def test_solution_solves_task(task_name, seed):
    task = HANOI.tasks[task_name]
    w = World(HANOI)
    w.reset(HANOI.layout(seed), task.initial_qpos)
    arm = Arm(w)
    arm.reset_ready()
    for action in task.solution(w):
        arm.act(action)
    assert task.success(w)


STACK = {"large_ring": _on("left", 0), "medium_ring": _on("left", 1)}
TOWER = {"large_ring": _on("right", 0), "medium_ring": _on("right", 1), "small_ring": _on("right", 2)}
# (task, ring centres, expected): each task has one set that passes, to show that the pose code works.
CHECK_CASES = [
    ("top_ring_to_right", {**STACK, "small_ring": _on("right", 0)}, True),
    ("top_ring_to_right", {**STACK, "small_ring": _on("right", 0, dx=0.03)}, False),  # beside the peg
    ("top_ring_to_right", {**STACK, "small_ring": _on("right", 0, dz=PEG_TOP)}, False),  # above the peg top
    ("top_ring_to_right", {**STACK, "small_ring": _on("middle", 0)}, False),
    ("last_two_moves", TOWER, True),
    ("last_two_moves", {"small_ring": _on("right", 0), "medium_ring": _on("right", 1),
                        "large_ring": _on("right", 2)}, False),  # the wrong order
    ("last_two_moves", {**TOWER, "small_ring": _on("middle", 0)}, False),
]


@pytest.mark.assets
@pytest.mark.parametrize("task_name,centres,expected", CHECK_CASES)
def test_success_check_on_set_poses(task_name, centres, expected):
    w = World(HANOI)
    w.reset({})
    _set_rings(w, centres)
    assert HANOI.tasks[task_name].success(w) is expected


@pytest.mark.assets
def test_check_rejects_a_ring_the_gripper_holds():
    """The plan without its last two actions holds the small ring over the right peg. The check passes only
    after the gripper lets go and the ring slides down."""
    task = HANOI.tasks["top_ring_to_right"]
    w = World(HANOI)
    w.reset({})
    arm = Arm(w)
    arm.reset_ready()
    plan = task.solution(w)
    assert plan[-2] == Action("open")
    for a in plan[:-2]:
        arm.act(a)
    assert "small_ring" in w.gripper_contacts()
    assert not task.success(w)
    for a in plan[-2:]:
        arm.act(a)
    assert task.success(w)
    assert w.object_pos("small_ring")[2] == pytest.approx(_level(0), abs=0.002)  # down on the base board
