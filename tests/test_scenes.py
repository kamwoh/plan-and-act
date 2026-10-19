import itertools

import mujoco
import numpy as np
import pytest

from armmind.scenes import SCENES
from armmind.sim.arm import Arm
from armmind.sim.world import World
from armmind.types import Action

CASES = [(s, t) for s in SCENES for t in SCENES[s].tasks]


def test_registry_order_and_tasks():
    assert list(SCENES) == ["cubes", "tray", "zones", "shapes", "drawer", "buttons", "kitchen", "toys"]
    assert {n: list(s.tasks) for n, s in SCENES.items()} == {
        "cubes": ["reach", "pick", "stack"], "tray": ["red_in_tray", "all_in_tray"],
        "zones": ["red_on_green", "sort"], "shapes": ["pick_cylinder", "small_on_big"],
        "drawer": ["open_drawer", "close_drawer", "cube_in_drawer"], "buttons": ["press_green", "press_sequence"],
        "kitchen": ["bottle_in_bowl", "purple_box_on_plate", "clear_table"],
        "toys": ["panda_in_tray", "all_toys_in_tray"]}


def test_cubes_layout_is_unchanged():
    cubes = SCENES["cubes"]
    assert cubes.layout(0) == {"red_cube": (0.559, -0.115, 0.0), "blue_cube": (0.41, -0.242, 0.0),
                               "green_cube": (0.603, 0.206, 0.0)}
    assert cubes.layout(4) == {"red_cube": (0.636, 0.006, 0.0), "blue_cube": (0.644, -0.21, 0.0),
                               "green_cube": (0.552, -0.062, 0.0)}


@pytest.mark.parametrize("name", ["cubes", "tray", "zones", "shapes", "drawer", "buttons"])
def test_old_scenes_have_no_turned_or_placed_objects(name):
    scene = SCENES[name]
    assert all(not o.random_yaw and o.place is None and o.shape != "mesh" for o in scene.objects)
    assert all(yaw == 0.0 for seed in range(5) for _, _, yaw in scene.layout(seed).values())


@pytest.mark.parametrize("name", list(SCENES))
def test_layout_is_repeatable_and_spread_out(name):
    scene = SCENES[name]
    x0, x1, y0, y1 = scene.region
    for seed in range(20):
        a, b = scene.layout(seed), scene.layout(seed)
        assert a == b
        assert set(a) == {o.name for o in scene.objects}
        for (m, p), (n, q) in itertools.combinations(a.items(), 2):
            radii = {o.name: o.footprint for o in scene.objects}
            assert np.hypot(p[0] - q[0], p[1] - q[1]) >= max(scene.min_gap, radii[m] + radii[n])
        for o in scene.objects:
            x, y, _ = a[o.name]
            if o.place is None:
                assert x0 <= x <= x1 and y0 <= y <= y1
            else:
                assert (x, y) == o.place
    if len(scene.objects) > 0:
        assert scene.layout(0) != scene.layout(1)


@pytest.mark.parametrize("name", list(SCENES))
def test_describe_lists_every_object(name):
    scene = SCENES[name]
    text = scene.describe()
    assert text.startswith("Objects: ")
    for obj in scene.objects:
        assert obj.label in text and obj.text() in text
    assert scene.fixtures_text in text


@pytest.mark.assets
@pytest.mark.parametrize("name", list(SCENES))
def test_model_loads_with_every_object(name):
    scene = SCENES[name]
    w = World(scene)
    w.reset(scene.layout(0))
    assert w.objects == scene.objects
    for obj in scene.objects:
        gap = 0.002 if obj.shape == "mesh" else 0.0
        assert w.object_pos(obj.name)[2] == pytest.approx(obj.half_height + gap, abs=1e-6)


@pytest.mark.assets
@pytest.mark.parametrize("name", list(SCENES))
def test_fixtures_are_outside_the_region(name):
    """Every fixture geom keeps clear of the spawn region, grown by the widest object."""
    scene = SCENES[name]
    w = World(scene)
    m = w.model
    grow = max((o.footprint for o in scene.objects), default=0.0)
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
    assert fixtures == {"cubes": 0, "tray": 5, "zones": 3, "shapes": 0, "drawer": 5, "buttons": 1, "kitchen": 0,
                        "toys": 5}[name]


@pytest.mark.assets
@pytest.mark.parametrize("name", list(SCENES))
def test_objects_stay_still_at_their_start_poses(name):
    scene = SCENES[name]
    w = World(scene)
    for seed in range(2):
        w.reset(scene.layout(seed))
        w.step(200)
        for obj in scene.objects:
            np.testing.assert_allclose(w.object_pos(obj.name)[:2], scene.layout(seed)[obj.name][:2], atol=0.002)
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


@pytest.mark.assets
def test_drawer_front_camera_is_the_side_view():
    m = World(SCENES["drawer"]).model
    assert m.cam_pos[m.camera("front").id][1] < -0.8  # far to the -y side
    cubes = World(SCENES["cubes"]).model
    assert cubes.cam_pos[cubes.camera("front").id][0] == pytest.approx(1.45)


def test_image_text_and_camera_labels():
    drawer = SCENES["drawer"]
    assert drawer.image_text.startswith("Image 1: side camera, looking along +y")
    assert drawer.camera_labels["front"] == "Side"
    for name in ("cubes", "tray", "zones", "shapes", "buttons"):
        assert SCENES[name].image_text is None and SCENES[name].camera_overrides == {}
        assert SCENES[name].camera_labels == {"front": "Front", "top": "Top · grid"}


@pytest.mark.assets
def test_joint_text_has_units_and_meaning():
    from armmind.env import MujocoEnv
    for name, want in [("drawer", ("drawer slide 0.120 m (0 = closed)",)),
                       ("buttons", ("red button depth 0.000 m", "green button depth 0.000 m",
                                    "blue button depth 0.000 m"))]:
        scene = SCENES[name]
        env = MujocoEnv(scene, None, 0, "x", show_positions=True)
        try:
            if name == "drawer":
                env.world.reset(scene.layout(0), {"drawer": 0.12})
            assert env.observe().joints == want
        finally:
            env.close()


@pytest.mark.assets
def test_open_drawer_keeps_clear_of_the_cube_region():
    """At full open (0.15 m), no drawer geom reaches the cube start region. The cube starts square to the axes,
    so the region grows by its half width, plus 3 mm."""
    scene = SCENES["drawer"]
    w = World(scene)
    w.reset(scene.layout(0), {"drawer": 0.15})
    m, d = w.model, w.data
    grow = scene.objects[0].size[0] + 0.003
    x0, x1, y0, y1 = scene.region
    drawer = m.body("drawer").id
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == drawer]
    assert len(geoms) == 8
    for g in geoms:
        rot = d.geom_xmat[g].reshape(3, 3)
        centre = d.geom_xpos[g] + rot @ m.geom_aabb[g][:3]
        half = np.abs(rot) @ m.geom_aabb[g][3:]
        apart = (centre[0] + half[0] < x0 - grow or centre[0] - half[0] > x1 + grow
                 or centre[1] + half[1] < y0 - grow or centre[1] - half[1] > y1 + grow)
        assert apart, f"{m.geom(g).name} reaches the cube region"


def test_buttons_text_says_to_close_the_gripper_first():
    assert "Close the gripper first" in SCENES["buttons"].fixtures_text


def _dropped(scene_name: str, spots: dict, seed: int = 0, settle: bool = True) -> World:
    """A world from the layout of the seed, with some items moved: name -> (x, y, bottom height). The items
    fall the last millimetres and settle for 1 s, unless settle is False. The arm stays in its home pose."""
    scene = SCENES[scene_name]
    w = World(scene)
    w.reset(scene.layout(seed))
    for name, (x, y, bottom) in spots.items():
        adr = w.model.jnt_qposadr[w.model.joint(name).id]
        w.data.qpos[adr:adr + 3] = [x, y, bottom - w.object_info(name).bottom + 0.002]
    mujoco.mj_forward(w.model, w.data)
    if settle:
        w.step(500)
    assert w.gripper_contacts() == []
    return w


AWAY_K = {"purple_box": (0.40, -0.10, 0.0), "yellow_box": (0.40, 0.10, 0.0)}
# (task, item spots, expected). The bowl inner floor is about 1 cm up, the plate surface about 0.8 cm up.
KITCHEN_CHECKS = [
    ("bottle_in_bowl", {"bottle": (0.62, 0.22, 0.012), **AWAY_K}, True),
    ("bottle_in_bowl", {"bottle": (0.62, 0.22 - 0.12, 0.0), **AWAY_K}, False),  # beside the bowl
    ("bottle_in_bowl", {"bottle": (0.62, -0.22, 0.01), **AWAY_K}, False),  # on the plate
    ("purple_box_on_plate", {"purple_box": (0.62, -0.22, 0.01), "bottle": (0.40, 0.0, 0.0),
                             "yellow_box": (0.40, 0.12, 0.0)}, True),
    ("purple_box_on_plate", {"purple_box": (0.62, -0.22 + 0.14, 0.0), "bottle": (0.40, 0.0, 0.0),
                             "yellow_box": (0.40, 0.12, 0.0)}, False),  # beside the plate
    ("purple_box_on_plate", {"purple_box": (0.62, 0.22, 0.012), "bottle": (0.40, 0.0, 0.0),
                             "yellow_box": (0.40, 0.12, 0.0)}, False),  # in the bowl
    ("clear_table", {"bottle": (0.62, 0.22, 0.012), "purple_box": (0.62, -0.175, 0.01),
                     "yellow_box": (0.62, -0.265, 0.01)}, True),
    ("clear_table", {"bottle": (0.62, 0.22, 0.012), "purple_box": (0.62, -0.175, 0.01),
                     "yellow_box": (0.45, -0.1, 0.0)}, False),  # one item is still on the table
]


@pytest.mark.assets
@pytest.mark.objects
@pytest.mark.parametrize("task_name,spots,expected", KITCHEN_CHECKS)
def test_kitchen_checks_on_set_spots(task_name, spots, expected):
    w = _dropped("kitchen", spots)
    assert SCENES["kitchen"].tasks[task_name].success(w) is expected


AWAY_T = {"orange_robot": (0.40, -0.20, 0.0), "green_robot": (0.60, -0.20, 0.0), "pink_toy": (0.40, 0.0, 0.0)}
# (task, toy spots, settle, expected)
TOY_CHECKS = [
    ("panda_in_tray", {"panda": (0.50, 0.28, 0.005), **AWAY_T}, True, True),
    ("panda_in_tray", {"panda": (0.50, 0.28 - 0.12, 0.0), **AWAY_T}, True, False),  # beside the tray
    # The height limit only: the panda is set above the pink toy and checked at once, before it can fall.
    # A low point at 7.4 cm is too high for "in the tray".
    pytest.param("panda_in_tray", {**AWAY_T, "pink_toy": (0.50, 0.28, 0.005), "panda": (0.50, 0.28, 0.074)},
                 False, False, id="height-limit-only-not-settled"),
    ("all_toys_in_tray", {"green_robot": (0.46, 0.25, 0.005), "pink_toy": (0.46, 0.315, 0.005),
                          "orange_robot": (0.54, 0.25, 0.005), "panda": (0.54, 0.313, 0.005)}, True, True),
    ("all_toys_in_tray", {"green_robot": (0.46, 0.25, 0.005), "pink_toy": (0.46, 0.315, 0.005),
                          "orange_robot": (0.54, 0.25, 0.005), "panda": (0.40, 0.0, 0.0)}, True, False),
]


@pytest.mark.assets
@pytest.mark.objects
@pytest.mark.parametrize("task_name,spots,settle,expected", TOY_CHECKS)
def test_toy_checks_on_set_spots(task_name, spots, settle, expected):
    w = _dropped("toys", spots, settle=settle)
    assert SCENES["toys"].tasks[task_name].success(w) is expected


@pytest.mark.assets
@pytest.mark.objects
@pytest.mark.parametrize("scene_name,task_name,item", [("kitchen", "bottle_in_bowl", "bottle"),
                                                        ("kitchen", "purple_box_on_plate", "purple_box"),
                                                        ("toys", "panda_in_tray", "panda")])
def test_mesh_checks_reject_an_item_the_gripper_holds(scene_name, task_name, item):
    """The scripted plan without its last two actions holds the item at the target. The check passes only
    after the gripper lets go and moves up."""
    scene = SCENES[scene_name]
    task = scene.tasks[task_name]
    w = World(scene)
    w.reset(scene.layout(0))
    arm = Arm(w)
    arm.reset_ready()
    plan = task.solution(w)
    assert plan[-2] == Action("open")
    for a in plan[:-2]:
        arm.act(a)
    assert item in w.gripper_contacts()
    assert not task.success(w)
    for a in plan[-2:]:
        arm.act(a)
    assert task.success(w)


def test_objects_txt_lists_every_scanned_object_once():
    from pathlib import Path

    import armmind.scenes
    listed = Path(armmind.scenes.__file__).with_name("objects.txt").read_text().split()
    used = [m for s in SCENES.values() for m in s.meshes]
    assert sorted(listed) == sorted(set(listed)) == sorted(set(used))


@pytest.mark.assets
@pytest.mark.objects
def test_on_plate_rejects_an_item_on_the_rim():
    """The purple box rests with its bottom at the rim height, beside the plate centre. It must fail."""
    w = _dropped("kitchen", {"bottle": (0.40, 0.0, 0.0), "yellow_box": (0.40, 0.12, 0.0)})
    plate = w.object_pos("plate")
    info = w.object_info("plate")
    rim = plate[2] + info.bottom + info.height
    adr = w.model.jnt_qposadr[w.model.joint("purple_box").id]
    w.data.qpos[adr:adr + 7] = [plate[0], plate[1] - 0.05, rim - w.object_info("purple_box").bottom, 1, 0, 0, 0]
    mujoco.mj_forward(w.model, w.data)
    assert w.object_bottom("purple_box") == pytest.approx(rim, abs=1e-4)
    assert not SCENES["kitchen"].tasks["purple_box_on_plate"].success(w)
    w.data.qpos[adr + 2] -= rim - 0.011  # down to about the plate floor: then it passes
    mujoco.mj_forward(w.model, w.data)
    assert SCENES["kitchen"].tasks["purple_box_on_plate"].success(w)
