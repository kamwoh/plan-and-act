import math
import re

import numpy as np
import pytest

from armmind.scenes.base import ObjSpec, Scene
from armmind.sim.assets import AssetsMissing
from armmind.sim.meshes import CONTACT, FRICTION, asset_xml, geoms_xml, mesh_dir

FAKE = "Fake_Object"


@pytest.fixture
def fake_objects(tmp_path, monkeypatch):
    """A models folder with one object of three collision pieces. The files are empty: only names matter here."""
    folder = tmp_path / FAKE
    folder.mkdir()
    for f in ("model.obj", "texture.png", "model.xml", *(f"model_collision_{i}.obj" for i in (0, 1, 2, 10))):
        (folder / f).write_text("")
    monkeypatch.setenv("ARMMIND_OBJECTS", str(tmp_path))
    return folder


def test_asset_xml_has_prefixed_names_and_absolute_paths(fake_objects):
    xml = asset_xml("cup", FAKE)
    assert f'<texture name="cup_tex" type="2d" file="{fake_objects / "texture.png"}"/>' in xml
    assert '<material name="cup_mat" texture="cup_tex"' in xml
    assert f'<mesh name="cup_vis" file="{fake_objects / "model.obj"}"/>' in xml
    for i in (0, 1, 2, 10):
        assert f'<mesh name="cup_col{i}" file="{fake_objects / f"model_collision_{i}.obj"}"/>' in xml
    assert xml.count("<mesh ") == 5
    assert all(p.startswith("/") for p in re.findall(r'file="([^"]+)"', xml))


def test_geoms_xml_hides_collision_and_shifts_to_the_centre(fake_objects):
    xml = geoms_xml("cup", FAKE, offset=(0.01, -0.02, -0.03), density=500.0)
    assert '<geom name="cup" type="mesh" mesh="cup_vis" material="cup_mat" pos="-0.01 0.02 0.03" ' \
           'contype="0" conaffinity="0" group="2" density="0"/>' in xml
    assert xml.count('group="3"') == 4 and xml.count(f'friction="{FRICTION}" {CONTACT}') == 4
    assert '<geom name="cup_col10" type="mesh" mesh="cup_col10" pos="-0.01 0.02 0.03" group="3" density="500"' in xml
    assert 'condim="4"' in CONTACT


def test_missing_object_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ARMMIND_OBJECTS", str(tmp_path))
    with pytest.raises(AssetsMissing, match="fetch_objects.sh"):
        mesh_dir("Nothing_Here")


def test_grasp_yaw_picks_the_narrow_axis():
    from armmind.scenes.handling import grasp_yaw
    # The fingers close along world x at yaw 0. A positive yaw turns them counter-clockwise.
    assert grasp_yaw(0.0, "x") == 0.0
    assert abs(grasp_yaw(0.0, "y")) == 90.0
    assert grasp_yaw(30.0, "x") == pytest.approx(30.0)
    assert grasp_yaw(30.0, "y") == pytest.approx(-60.0)
    assert grasp_yaw(120.0, "x") == pytest.approx(-60.0)
    assert grasp_yaw(-170.0, "y") == pytest.approx(-80.0)
    assert grasp_yaw(100.0, "y") == pytest.approx(10.0)
    for yaw in np.linspace(-180, 180, 37):
        for axis in "xy":
            assert -90.0 <= grasp_yaw(float(yaw), axis) <= 90.0


def _boxes() -> Scene:
    """Two cube objects, one with a random yaw, in a small scene."""
    return Scene(name="yawtest", title="Yaw", objects=(
        ObjSpec("a", "box", (0.02, 0.02, 0.02), "1 0 0 1", "a"),
        ObjSpec("b", "box", (0.03, 0.01, 0.02), "0 0 1 1", "b", random_yaw=True),
    ), fixtures_xml="", fixtures_text="", region=(0.4, 0.6, -0.2, 0.2), min_gap=0.08, tasks={})


def test_layout_yaw_is_repeatable_and_changes_with_the_seed():
    s = _boxes()
    a = s.layout(0)
    assert a == s.layout(0)
    assert a["a"][2] == 0.0  # no random yaw: always 0
    yaws = {s.layout(seed)["b"][2] for seed in range(10)}
    assert len(yaws) == 10 and all(-180.0 < y <= 180.0 for y in yaws)


def test_layout_keeps_big_objects_apart_by_their_footprint():
    big = ObjSpec("big", "box", (0.1, 0.1, 0.02), "1 0 0 1", "big")
    s = Scene(name="fp", title="Fp", objects=(big, ObjSpec("small", "box", (0.02, 0.02, 0.02), "0 0 1 1", "s")),
              fixtures_xml="", fixtures_text="", region=(0.3, 0.7, -0.3, 0.3), min_gap=0.05, tasks={})
    need = math.hypot(0.1, 0.1) + math.hypot(0.02, 0.02)
    for seed in range(30):
        p, q = s.layout(seed).values()
        assert math.hypot(p[0] - q[0], p[1] - q[1]) >= need


def test_layout_puts_a_placed_object_at_its_place_and_keeps_others_clear():
    plate = ObjSpec("plate", "cylinder", (0.1, 0.01), "1 1 1 1", "plate", place=(0.5, 0.0))
    s = Scene(name="pl", title="Pl", objects=(plate, ObjSpec("c", "box", (0.02, 0.02, 0.02), "0 0 1 1", "c")),
              fixtures_xml="", fixtures_text="", region=(0.3, 0.7, -0.3, 0.3), min_gap=0.05, tasks={})
    for seed in range(20):
        lay = s.layout(seed)
        assert lay["plate"] == (0.5, 0.0, 0.0)
        assert math.hypot(lay["c"][0] - 0.5, lay["c"][1]) >= 0.1 + math.hypot(0.02, 0.02)


@pytest.mark.assets
def test_quaternion_matches_the_yaw():
    from armmind.sim.world import World
    w = World(_boxes())
    w.reset({"a": (0.45, -0.1, 0.0), "b": (0.55, 0.1, 30.0)})
    adr = w.model.jnt_qposadr[w.model.joint("b").id]
    np.testing.assert_allclose(w.data.qpos[adr + 3:adr + 7], [math.cos(math.radians(15)), 0, 0,
                                                               math.sin(math.radians(15))], atol=1e-9)
    assert w.object_yaw("b") == pytest.approx(30.0, abs=1e-6)
    assert w.object_yaw("a") == pytest.approx(0.0, abs=1e-6)
    w.reset({"a": (0.45, -0.1), "b": (0.55, 0.1)})  # an (x, y) layout still works: yaw 0
    assert w.object_yaw("b") == pytest.approx(0.0, abs=1e-6)


@pytest.mark.assets
def test_object_info_of_a_box():
    from armmind.sim.world import World
    w = World(_boxes())
    info = w.object_info("b")
    assert (info.width, info.depth, info.height) == pytest.approx((0.06, 0.02, 0.04))
    assert info.bottom == pytest.approx(-0.02) and info.narrow == "y"
    w.reset({"a": (0.45, -0.1, 0.0), "b": (0.55, 0.1, 0.0)})
    assert w.object_bottom("b") == pytest.approx(0.0, abs=1e-6)


def test_mesh_spec_needs_a_mesh_name():
    with pytest.raises(ValueError, match="mesh"):
        ObjSpec("x", "mesh", label="thing")
    assert ObjSpec("x", "mesh", label="thing", mesh="Some_Folder").size == ()


@pytest.mark.objects
def test_scanned_mesh_compiles_and_measures():
    from armmind.scenes import SCENES
    from armmind.sim.meshes import measure
    for scene in ("kitchen", "toys"):
        for o in SCENES[scene].objects:
            if o.shape != "mesh":
                continue
            info = measure(o.mesh)
            w, d, h = info.dims
            assert 0.01 < w < 0.3 and 0.01 < d < 0.3 and 0.01 < h < 0.2
            assert info.mass > 0


@pytest.mark.assets
@pytest.mark.objects
@pytest.mark.parametrize("scene_name", ["kitchen", "toys"])
def test_world_measures_mesh_objects_centred(scene_name):
    from armmind.scenes import SCENES
    from armmind.sim.meshes import measure
    from armmind.sim.world import World
    scene = SCENES[scene_name]
    w = World(scene)
    for o in scene.objects:
        info = w.object_info(o.name)
        dims = measure(o.mesh).dims
        assert (info.width, info.depth, info.height) == pytest.approx(dims, abs=1e-4)
        assert info.bottom == pytest.approx(-info.height / 2, abs=1e-4)  # the body origin is the box centre
        assert info.narrow == ("x" if info.width <= info.depth else "y")
        mass = w.model.body_subtreemass[w.model.body(o.name).id]
        assert mass == pytest.approx(o.mass, rel=0.02)


@pytest.mark.assets
@pytest.mark.objects
@pytest.mark.parametrize("scene_name", ["kitchen", "toys"])
def test_mesh_objects_rest_on_the_table(scene_name):
    """Each object starts 2 mm above the table, at its layout yaw, and stands still for 2 s."""
    from armmind.scenes import SCENES
    from armmind.sim.world import World
    scene = SCENES[scene_name]
    w = World(scene)
    for seed in range(3):
        lay = scene.layout(seed)
        w.reset(lay)
        for name, (x, y, yaw) in lay.items():
            assert w.object_bottom(name) == pytest.approx(0.002, abs=1e-4)
            assert w.object_yaw(name) == pytest.approx(yaw, abs=1e-6)
        assert all(d >= -1e-4 for d in w.data.contact.dist)  # nothing starts inside anything
        w.step(round(2.0 / w.dt))
        for name, (x, y, yaw) in lay.items():
            np.testing.assert_allclose(w.object_pos(name)[:2], (x, y), atol=0.003)
            assert w.object_bottom(name) == pytest.approx(0.0, abs=0.002)
            turn = (w.object_yaw(name) - yaw + 180) % 360 - 180
            assert abs(turn) < 2.0
            z_axis = w.data.body(name).xmat.reshape(3, 3)[:, 2]
            assert z_axis[2] > math.cos(math.radians(3))  # not tipped


@pytest.mark.assets
@pytest.mark.objects
@pytest.mark.parametrize("scene_name", ["kitchen", "toys"])
@pytest.mark.parametrize("seed", range(3))
def test_scripted_grasp_lifts_every_item(scene_name, seed):
    from armmind.scenes import SCENES
    from armmind.scenes.handling import grip, items
    from armmind.sim.arm import Arm
    from armmind.sim.world import World
    from armmind.types import Action
    scene = SCENES[scene_name]
    w = World(scene)
    arm = Arm(w)
    for o in items(scene):
        w.reset(scene.layout(seed))
        arm.reset_ready()
        z0 = w.object_pos(o.name)[2]
        p = w.object_pos(o.name)
        for a in [*grip(w, o.name), Action("move_to", (float(p[0]), float(p[1]), 0.25))]:
            arm.act(a)
        assert w.object_pos(o.name)[2] >= z0 + 0.10, o.name
        assert o.name in w.gripper_contacts()


def test_mesh_label_without_objects_is_the_plain_label(tmp_path, monkeypatch):
    monkeypatch.setenv("ARMMIND_OBJECTS", str(tmp_path))
    assert ObjSpec("x", "mesh", label="red can", mesh="Missing_Folder").text() == "red can"


@pytest.mark.objects
def test_mesh_label_has_the_measured_size():
    from armmind.scenes import SCENES
    bottle = next(o for o in SCENES["kitchen"].objects if o.name == "bottle")
    w, d, h = (round(v * 100, 1) for v in bottle.dims)
    assert bottle.text() == f"white bottle, {w:.1f} cm wide, {d:.1f} cm deep, {h:.1f} cm tall"
    assert bottle.text() in SCENES["kitchen"].describe()
