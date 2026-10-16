import itertools

import mujoco
import numpy as np
import pytest

from armmind.scenes import SCENES
from armmind.sim.arm import Arm
from armmind.sim.world import World
from armmind.types import Action

CASES = [(s, t) for s in SCENES for t in SCENES[s].tasks]


def _radius(obj) -> float:
    """The radius of the object's footprint on the table."""
    return float(np.hypot(obj.size[0], obj.size[1])) if obj.shape == "box" else float(obj.size[0])


def test_registry_order_and_tasks():
    assert list(SCENES) == ["cubes", "tray", "zones", "shapes", "drawer", "buttons"]
    assert {n: list(s.tasks) for n, s in SCENES.items()} == {
        "cubes": ["reach", "pick", "stack"], "tray": ["red_in_tray", "all_in_tray"],
        "zones": ["red_on_green", "sort"], "shapes": ["pick_cylinder", "small_on_big"],
        "drawer": ["open_drawer", "close_drawer", "cube_in_drawer"], "buttons": ["press_green", "press_sequence"]}


def test_cubes_layout_is_unchanged():
    cubes = SCENES["cubes"]
    assert cubes.layout(0) == {"red_cube": (0.559, -0.115), "blue_cube": (0.41, -0.242),
                               "green_cube": (0.603, 0.206)}
    assert cubes.layout(4) == {"red_cube": (0.636, 0.006), "blue_cube": (0.644, -0.21),
                               "green_cube": (0.552, -0.062)}


@pytest.mark.parametrize("name", list(SCENES))
def test_layout_is_repeatable_and_spread_out(name):
    scene = SCENES[name]
    x0, x1, y0, y1 = scene.region
    for seed in range(20):
        a, b = scene.layout(seed), scene.layout(seed)
        assert a == b
        assert set(a) == {o.name for o in scene.objects}
        for p, q in itertools.combinations(a.values(), 2):
            assert np.hypot(p[0] - q[0], p[1] - q[1]) >= scene.min_gap
        for x, y in a.values():
            assert x0 <= x <= x1 and y0 <= y <= y1
    if len(scene.objects) > 1:
        assert scene.layout(0) != scene.layout(1)


@pytest.mark.parametrize("name", list(SCENES))
def test_describe_lists_every_object(name):
    scene = SCENES[name]
    text = scene.describe()
    assert text.startswith("Objects: ")
    for obj in scene.objects:
        assert obj.label in text
    assert scene.fixtures_text in text


@pytest.mark.assets
@pytest.mark.parametrize("name", list(SCENES))
def test_model_loads_with_every_object(name):
    scene = SCENES[name]
    w = World(scene)
    w.reset(scene.layout(0))
    assert w.objects == scene.objects
    for obj in scene.objects:
        assert w.object_pos(obj.name)[2] == pytest.approx(obj.half_height, abs=1e-6)


@pytest.mark.assets
@pytest.mark.parametrize("name", list(SCENES))
def test_fixtures_are_outside_the_region(name):
    """Every fixture geom keeps clear of the spawn region, grown by the widest object."""
    scene = SCENES[name]
    w = World(scene)
    m = w.model
    grow = max((_radius(o) for o in scene.objects), default=0.0)
    x0, x1, y0, y1 = scene.region
    fixtures = 0
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        if b == 0 or m.body_parentid[b] != 0 or m.body_jntnum[b] or m.body(b).name == "link0":
            continue  # a fixture is a static body at the root, other than the arm base
        fixtures += 1
        cx, cy = m.body_pos[b][:2] + m.geom_pos[g][:2]
        hx, hy = m.geom_size[g][:2]
        apart = cx + hx < x0 - grow or cx - hx > x1 + grow or cy + hy < y0 - grow or cy - hy > y1 + grow
        assert apart, f"{m.body(b).name} geom {g} overlaps the region"
    assert fixtures == {"cubes": 0, "tray": 5, "zones": 3, "shapes": 0, "drawer": 5, "buttons": 1}[name]


@pytest.mark.assets
@pytest.mark.parametrize("name", list(SCENES))
def test_objects_stay_still_at_their_start_poses(name):
    scene = SCENES[name]
    w = World(scene)
    for seed in range(5):
        w.reset(scene.layout(seed))
        w.step(200)
        for obj in scene.objects:
            np.testing.assert_allclose(w.object_pos(obj.name)[:2], scene.layout(seed)[obj.name], atol=0.002)
            assert w.object_pos(obj.name)[2] == pytest.approx(obj.half_height, abs=0.002)


@pytest.mark.assets
@pytest.mark.parametrize("scene_name,task_name", CASES)
def test_no_task_is_solved_at_start(scene_name, task_name):
    scene = SCENES[scene_name]
    task = scene.tasks[task_name]
    w = World(scene)
    w.reset(scene.layout(0), task.initial_qpos)
    Arm(w).reset_ready()
    assert not task.success(w)


@pytest.mark.assets
@pytest.mark.parametrize("scene_name,task_name,seed", [(s, t, n) for s, t in CASES for n in range(3)])
def test_solution_solves_task(scene_name, task_name, seed):
    scene = SCENES[scene_name]
    task = scene.tasks[task_name]
    w = World(scene)
    w.reset(scene.layout(seed), task.initial_qpos)
    arm = Arm(w)
    arm.reset_ready()
    for action in task.solution(w):
        arm.act(action)
    assert task.success(w)


@pytest.mark.assets
def test_zones_do_not_collide_and_the_tray_does():
    w = World(SCENES["zones"])
    m = w.model
    zone_geoms = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.endswith("_zone")]
    assert len(zone_geoms) == 3 and all(m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 for g in zone_geoms)
    t = World(SCENES["tray"]).model
    assert all(t.geom_contype[g] == 1 for g in range(t.ngeom) if t.body(t.geom_bodyid[g]).name == "tray")


def test_layout_gives_up_when_the_objects_cannot_fit():
    from dataclasses import replace
    crowded = replace(SCENES["cubes"], region=(0.5, 0.51, 0.0, 0.01))
    with pytest.raises(ValueError, match="cubes"):
        crowded.layout(0)


def _posed(scene_name: str, poses: dict) -> World:
    """A world with each object at the given centre (x, y, z). The arm stays in its home pose, high above."""
    scene = SCENES[scene_name]
    w = World(scene)
    w.reset({n: p[:2] for n, p in poses.items()})
    for name, p in poses.items():
        adr = w.model.jnt_qposadr[w.model.joint(name).id]
        w.data.qpos[adr:adr + 7] = [*p, 1, 0, 0, 0]
    mujoco.mj_forward(w.model, w.data)
    assert w.gripper_contacts() == []
    return w


AWAY = {"blue_cube": (0.42, -0.22, 0.02), "green_cube": (0.62, -0.22, 0.02)}
TRAY_IN = (0.50, 0.28, 0.025)
# (scene, task, poses, expected): each set has one pose that passes, to show that the pose code works.
CHECK_CASES = [
    ("cubes", "stack", {"red_cube": (0.5, 0.0, 0.06), "blue_cube": (0.5, 0.0, 0.02), "green_cube": (0.62, -0.22, 0.02)},
     True),
    ("cubes", "stack", {"red_cube": (0.5, 0.05, 0.02), "blue_cube": (0.5, 0.0, 0.02), "green_cube": (0.62, -0.22, 0.02)},
     False),
    ("tray", "red_in_tray", {"red_cube": TRAY_IN, **AWAY}, True),
    ("tray", "red_in_tray", {"red_cube": (0.50 + 0.056, 0.28, 0.025), **AWAY}, False),
    ("tray", "red_in_tray", {"red_cube": (0.50, 0.28, 0.07), **AWAY}, False),
    ("zones", "red_on_green", {"red_cube": (0.68, 0.15, 0.02), **AWAY}, True),
    ("zones", "red_on_green", {"red_cube": (0.68 + 0.045, 0.15, 0.02), **AWAY}, False),
    ("zones", "red_on_green", {"red_cube": (0.68, 0.0, 0.02), **AWAY}, False),
    ("shapes", "small_on_big", {"big_cube": (0.5, 0.0, 0.03), "small_cube": (0.5, 0.0, 0.075),
                                "cylinder": (0.42, -0.22, 0.03)}, True),
    ("shapes", "small_on_big", {"big_cube": (0.5, 0.0, 0.03), "small_cube": (0.5, 0.06, 0.015),
                                "cylinder": (0.42, -0.22, 0.03)}, False),
    ("shapes", "small_on_big", {"big_cube": (0.5, 0.0, 0.03), "small_cube": (0.53, 0.0, 0.075),
                                "cylinder": (0.42, -0.22, 0.03)}, False),
]


@pytest.mark.assets
@pytest.mark.parametrize("scene_name,task_name,poses,expected", CHECK_CASES)
def test_success_check_on_set_poses(scene_name, task_name, poses, expected):
    w = _posed(scene_name, poses)
    assert SCENES[scene_name].tasks[task_name].success(w) is expected


@pytest.mark.assets
@pytest.mark.parametrize("touch", [True, False])
def test_tray_check_rejects_a_cube_the_gripper_holds(touch):
    """The gripper holds the red cube inside the tray. The check passes only after the gripper lets go."""
    from armmind.types import Action
    scene = SCENES["tray"]
    w = _posed("tray", {"red_cube": (0.5, 0.0, 0.02), **AWAY})
    arm = Arm(w)
    arm.reset_ready()
    for a in [Action("open"), Action("move_to", (0.5, 0.0, 0.15)), Action("move_to", (0.5, 0.0, 0.02)),
              Action("close"), Action("move_to", (0.5, 0.0, 0.15)), Action("move_to", (0.5, 0.28, 0.15)),
              Action("move_to", (0.5, 0.28, 0.03))]:
        arm.act(a)
    if not touch:
        arm.act(Action("open"))
        arm.act(Action("move_to", (0.5, 0.28, 0.15)))
    x, y, z = w.object_pos("red_cube")
    assert abs(x - 0.5) < 0.01 and abs(y - 0.28) < 0.01 and z < 0.06  # inside the tray either way
    assert ("red_cube" in w.gripper_contacts()) is touch
    assert scene.tasks["red_in_tray"].success(w) is (not touch)


def test_a_scene_without_objects_has_an_empty_layout():
    from dataclasses import replace
    empty = replace(SCENES["cubes"], objects=())
    assert empty.layout(0) == {}
    assert empty.describe().startswith("Objects: none.")


def _run(scene_name: str, actions, joint_qpos=None) -> World:
    scene = SCENES[scene_name]
    w = World(scene)
    w.reset(scene.layout(0), joint_qpos)
    arm = Arm(w)
    arm.reset_ready()
    for a in actions:
        arm.act(a)
    return w


@pytest.mark.assets
@pytest.mark.parametrize("slide,expected", [(0.05, False), (0.11, True)])
def test_open_drawer_needs_10_cm(slide, expected):
    w = _run("drawer", [], {"drawer": slide})
    assert w.joint_pos("drawer") == pytest.approx(slide, abs=1e-3)
    assert SCENES["drawer"].tasks["open_drawer"].success(w) is expected


@pytest.mark.assets
def test_drawer_stays_where_it_is_left():
    w = _run("drawer", [], {"drawer": 0.08})
    w.step(500)
    assert w.joint_pos("drawer") == pytest.approx(0.08, abs=1e-3)


@pytest.mark.assets
def test_close_drawer_check_rejects_a_gripper_on_the_handle():
    from armmind.scenes.drawer import handle_pos
    scene = SCENES["drawer"]
    w = World(scene)
    w.reset(scene.layout(0), {"drawer": 0.0})
    arm = Arm(w)
    arm.reset_ready()
    h = handle_pos(w)
    for a in [Action("close"), Action("move_to", (h[0] - 0.05, 0.0, h[2]), yaw=90.0),
              Action("move_to", (h[0] - 0.005, 0.0, h[2]))]:
        arm.act(a)
    assert "drawer" in w.gripper_contacts()
    assert not scene.tasks["close_drawer"].success(w)


@pytest.mark.assets
def test_cube_in_drawer_check_needs_the_cube_inside():
    scene = SCENES["drawer"]
    w = World(scene)
    w.reset(scene.layout(0), {"drawer": 0.12})
    assert not scene.tasks["cube_in_drawer"].success(w)  # the cube is on the table
    floor = w.data.body("drawer").xpos
    adr = w.model.jnt_qposadr[w.model.joint("red_cube").id]
    w.data.qpos[adr:adr + 3] = [floor[0], floor[1], floor[2] + 0.02]
    mujoco.mj_forward(w.model, w.data)
    assert scene.tasks["cube_in_drawer"].success(w)


def _presses(*colours: str) -> list:
    from armmind.scenes.buttons import press
    return [Action("close"), *(a for c in colours for a in press(c))]


@pytest.mark.assets
@pytest.mark.parametrize("colours,expected", [(("red", "green", "blue"), True), (("green", "red", "blue"), False),
                                              (("green",), False)])
def test_press_sequence_needs_the_right_order(colours, expected):
    w = _run("buttons", _presses(*colours))
    assert w.events == [f"{c} button pressed" for c in colours]  # once per press
    assert SCENES["buttons"].tasks["press_sequence"].success(w) is expected
    assert SCENES["buttons"].tasks["press_green"].success(w) is True


@pytest.mark.assets
def test_buttons_rest_up_and_no_event_at_start():
    w = _run("buttons", [Action("wait", seconds=1.0)])
    assert w.events == []
    assert all(q < 0.002 for q in w.joints().values())
    assert set(w.joints()) == {"red_button", "green_button", "blue_button"}
