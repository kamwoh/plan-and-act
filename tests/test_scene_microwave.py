import re

import mujoco
import numpy as np
import pytest

from plan_and_act.scenes.microwave import FLOOR_Z, FRONT, MICROWAVE, STARTED, start_monitor
from plan_and_act.sim.arm import Arm
from plan_and_act.sim.world import World
from plan_and_act.types import Action

SCENE = MICROWAVE
TASKS = list(SCENE.tasks)
MINE = [pytest.param(MICROWAVE, id="microwave")]  # the generic checks run on each scene of this file


def _geom_aabb(model, data, g):
    c, half = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half + c
    world = corners @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
    return world.min(0), world.max(0)


def _world(joint_qpos=None, seed: int = 0) -> World:
    w = World(SCENE)
    w.reset(SCENE.layout(seed), joint_qpos)
    return w


def _put(w: World, name: str, xyz) -> None:
    """Moves the object to the centre xyz, square to the axes."""
    adr = w.model.jnt_qposadr[w.model.joint(name).id]
    w.data.qpos[adr:adr + 7] = [*xyz, 1, 0, 0, 0]
    mujoco.mj_forward(w.model, w.data)


def test_tasks_and_layout():
    assert TASKS == ["open_door", "food_in_microwave", "cook"]
    for seed in range(5):
        assert SCENE.layout(seed) == SCENE.layout(seed)
    assert SCENE.layout(0) != SCENE.layout(1)
    assert SCENE.fixtures_text in SCENE.describe()


@pytest.mark.assets
def test_model_loads():
    w = _world()
    assert w.objects == SCENE.objects
    assert w.object_pos("noodles")[2] == pytest.approx(SCENE.objects[0].half_height, abs=1e-6)
    assert set(w.joints()) == {"microwave_door", "start_button"}


@pytest.mark.assets
@pytest.mark.parametrize("scene", MINE)
def test_fixture_list_agrees_with_the_fixture_xml(scene):
    names = re.findall(r'<geom name="([^"]+)"', scene.fixtures_xml)
    assert sorted(e.name for e in scene.fixtures) == sorted(names)
    w = World(scene)
    mujoco.mj_forward(w.model, w.data)
    for e in scene.fixtures:
        lo, hi = _geom_aabb(w.model, w.data, w.model.geom(e.name).id)
        half = np.array(e.size) / 2
        assert np.allclose(lo, np.array(e.xyz) - half, atol=5e-4), e.name
        assert np.allclose(hi, np.array(e.xyz) + half, atol=5e-4), e.name


@pytest.mark.assets
@pytest.mark.parametrize("scene", MINE)
def test_fixtures_are_outside_the_region(scene):
    """Every static fixture geom keeps clear of the spawn region, grown by the widest object."""
    w = World(scene)
    m = w.model
    grow = max(o.footprint for o in scene.objects)
    x0, x1, y0, y1 = scene.region
    fixtures = 0
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        if b == 0 or m.body_parentid[b] != 0 or m.body_jntnum[b] or m.body(b).name == "link0":
            continue
        fixtures += 1
        cx, cy = m.body_pos[b][:2] + m.geom_pos[g][:2]
        hx, hy = m.geom_size[g][:2]
        assert cx + hx < x0 - grow or cx - hx > x1 + grow or cy + hy < y0 - grow or cy - hy > y1 + grow
    assert fixtures == 6


@pytest.mark.assets
def test_open_door_keeps_clear_of_the_region():
    """At full open, no door geom reaches the spawn region, grown by the widest object."""
    w = _world({"microwave_door": np.radians(110.0)})
    m, d = w.model, w.data
    grow = max(o.footprint for o in SCENE.objects)
    x0, x1, y0, y1 = SCENE.region
    door = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name == "microwave_door"]
    assert len(door) == 4
    for g in door:
        lo, hi = _geom_aabb(m, d, g)
        assert hi[0] < x0 - grow or lo[0] > x1 + grow or hi[1] < y0 - grow or lo[1] > y1 + grow, m.geom(g).name


@pytest.mark.assets
@pytest.mark.parametrize("scene", MINE)
def test_objects_stay_still_at_their_start_poses(scene):
    w = World(scene)
    for seed in range(2):
        w.reset(scene.layout(seed))
        w.step(200)
        for obj in scene.objects:
            np.testing.assert_allclose(w.object_pos(obj.name)[:2], scene.layout(seed)[obj.name][:2], atol=0.002)
            assert w.object_pos(obj.name)[2] == pytest.approx(obj.half_height, abs=0.002)


@pytest.mark.assets
@pytest.mark.parametrize("angle", [0.0, 0.8])
def test_door_stays_where_it_is_left(angle):
    w = _world({"microwave_door": angle})
    w.step(1500)
    assert w.joint_pos("microwave_door") == pytest.approx(angle, abs=0.01)
    assert w.events == []


@pytest.mark.assets
@pytest.mark.parametrize("scene", MINE)
def test_no_task_is_solved_at_start(scene):
    for task in scene.tasks.values():
        w = World(scene)
        w.reset(scene.layout(0), task.initial_qpos)
        Arm(w).reset_ready()
        assert not task.success(w), task.name


@pytest.mark.assets
@pytest.mark.parametrize("scene,task_name,seed",
                         [pytest.param(p.values[0], t, n, id=f"{p.id}-{t}-{n}") for p in MINE
                          for t in p.values[0].tasks for n in range(5)])
def test_solution_solves_task(scene, task_name, seed):
    task = scene.tasks[task_name]
    w = World(scene)
    w.reset(scene.layout(seed), task.initial_qpos)
    arm = Arm(w)
    arm.reset_ready()
    for action in task.solution(w):
        arm.act(action)
    assert task.success(w)


@pytest.mark.assets
@pytest.mark.parametrize("angle,expected", [(0.5, False), (1.3, True)])
def test_open_door_needs_70_degrees(angle, expected):
    w = _world({"microwave_door": angle})
    assert SCENE.tasks["open_door"].success(w) is expected


INSIDE = (FRONT + 0.08, 0.0, FLOOR_Z + 0.025)


@pytest.mark.assets
@pytest.mark.parametrize("xyz,expected", [
    (INSIDE, True),
    ((FRONT + 0.02, 0.0, FLOOR_Z + 0.025), False),  # the back of the box sticks out of the cavity
    ((FRONT + 0.08, 0.11, FLOOR_Z + 0.025), False),  # it goes into the side wall
    ((0.43, -0.3, 0.025), False),  # on the table
])
def test_food_check_on_set_poses(xyz, expected):
    w = _world({"microwave_door": 1.7})
    _put(w, "noodles", xyz)
    assert SCENE.tasks["food_in_microwave"].success(w) is expected


@pytest.mark.assets
@pytest.mark.parametrize("door,events,expected", [
    (0.0, [STARTED], True),
    (0.0, [], False),  # the button was not pressed
    (0.0, ["microwave started empty"], False),
    (0.3, [STARTED], False),  # the door is open now
])
def test_cook_check(door, events, expected):
    w = _world({"microwave_door": door})
    _put(w, "noodles", INSIDE)
    w.events = list(events)
    assert SCENE.tasks["cook"].success(w) is expected


@pytest.mark.assets
@pytest.mark.parametrize("door,food_in,event", [
    (0.0, True, STARTED),
    (0.0, False, "microwave started empty"),
    (0.5, True, "start button pressed with the door open"),
])
def test_start_monitor_says_what_the_press_did(door, food_in, event):
    w = _world({"microwave_door": door})
    if food_in:
        _put(w, "noodles", INSIDE)
    check = start_monitor()
    assert check(w) is None  # the button is up
    w.data.qpos[w.model.jnt_qposadr[w.model.joint("start_button").id]] = 0.007
    assert check(w) == event
    assert check(w) is None  # once for each press


@pytest.mark.assets
def test_button_press_gives_one_event():
    w = _world()
    arm = Arm(w)
    arm.reset_ready()
    for a in [Action("close"), Action("move_to", (0.56, -0.175, 0.2)), Action("move_to", (0.56, -0.175, 0.134)),
              Action("move_to", (0.56, -0.175, 0.2))]:
        arm.act(a)
    assert w.events == ["microwave started empty"]
