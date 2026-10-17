"""The generic plan formats and the PlanView of the MuJoCo arm environment."""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from armmind.plan.formats import Entity, Keyframe, build_track
from armmind.scenes import SCENES
from armmind.sim.world import CAMERAS

ROOT = Path(__file__).resolve().parents[1]


def test_generic_modules_import_no_mujoco():
    code = ("import sys, armmind.plan.formats, armmind.plan.scores; "
            "bad = [m for m in sys.modules if m.split('.')[0] in ('mujoco', 'numpy') or m.startswith('armmind.sim')]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr


def test_build_track_follows_the_given_motion_rule():
    kfs = [Keyframe((0.5, 0.0, 0.4), 0.0, "open", None), Keyframe((0.5, 0.0, 0.1), 0.0, "closed", "cube"),
           Keyframe((0.5, 0.0, 0.1), 0.0, "closed", "cube")]
    track = build_track({"xyz": (0.3, 0.0, 0.4), "yaw": 0.0, "state": "open"}, kfs,
                        move_s=lambda a, b, ya, yb: abs(a[2] - b[2]) * 10 + 1, settle_s=0.5, change_s=2.0,
                        gap=lambda k: 0.08 if k.state == "open" else 0.04)
    ts = [(k["t"], k["t_done"]) for k in track["keyframes"]]
    assert ts == pytest.approx([(1.0, 1.5), (5.5, 8.0), (9.0, 9.5)])
    assert track["duration"] == pytest.approx(9.5) and track["start"]["gap"] == 0.08
    assert [k["gap"] for k in track["keyframes"]] == [0.08, 0.04, 0.04]
    assert [k["changes"] for k in track["keyframes"]] == [False, True, False]


# ---- The MuJoCo arm environment ----
def _env(scene="cubes", seed=0, task=None):
    from armmind.env import MujocoEnv
    sc = SCENES[scene]
    return MujocoEnv(sc, sc.tasks.get(task) if task else None, seed, "")


def _axes(pos, xyaxes):
    p = np.array([float(v) for v in pos.split()])
    a = np.array([float(v) for v in xyaxes.split()])
    x, y = a[:3] / np.linalg.norm(a[:3]), a[3:]
    y = y - x * (x @ y)
    y /= np.linalg.norm(y)
    return p, -np.cross(x, y), y


@pytest.mark.assets
@pytest.mark.parametrize("scene", ["cubes", "drawer"])
def test_cameras_have_the_mjcf_poses_and_fields_of_view(scene):
    env = _env(scene)
    try:
        view = env.plan_view((), (Keyframe((0.5, 0.0, 0.2), 0.0, "open", None),))
    finally:
        env.close()
    cams = {c["name"]: c for c in view["cameras"]}
    assert set(cams) == {"front", "top"}
    for name, spec in {**CAMERAS, **SCENES[scene].camera_overrides}.items():
        pos, forward, up = _axes(*spec)
        c = cams[name]
        assert np.allclose(c["pos"], pos, atol=1e-6)
        assert np.allclose(c["forward"], forward, atol=1e-6) and np.allclose(c["up"], up, atol=1e-6)
        assert c["fovy"] == 45 and (c["width"], c["height"]) == (640, 480)
        assert c["label"] == SCENES[scene].camera_labels[name]
        assert c["grid"] == (name not in SCENES[scene].camera_overrides)
    assert view["grid"]["x"][0] == 0.2 and view["grid"]["y"][-1] == 0.4


def _geom_aabb(model, data, g):
    c, half = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half + c
    world = corners @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
    return world.min(0), world.max(0)


def _entity_aabb(e: Entity):
    hx, hy, hz = (v / 2 for v in e.size)
    if e.shape == "box":
        c, s = np.cos(np.radians(e.yaw)), np.sin(np.radians(e.yaw))
        ex, ey = abs(c) * hx + abs(s) * hy, abs(s) * hx + abs(c) * hy
    else:
        ex, ey = hx, hy
    half = np.array([ex, ey, hz])
    return np.array(e.xyz) - half, np.array(e.xyz) + half


@pytest.mark.assets
@pytest.mark.parametrize("scene", list(SCENES))
def test_fixture_list_agrees_with_the_fixture_xml(scene):
    import mujoco

    from armmind.sim.world import World
    sc = SCENES[scene]
    names = re.findall(r'<geom name="([^"]+)"', sc.fixtures_xml)
    assert sorted(e.name for e in sc.fixtures) == sorted(names)
    w = World(sc)
    mujoco.mj_forward(w.model, w.data)
    for e in sc.fixtures:
        lo, hi = _geom_aabb(w.model, w.data, w.model.geom(e.name).id)
        elo, ehi = _entity_aabb(e)
        assert np.allclose(lo, elo, atol=5e-4) and np.allclose(hi, ehi, atol=5e-4), e.name
        assert e.color is not None and len(e.color) == 4


@pytest.mark.assets
def test_view_has_the_table_fixtures_and_the_true_start():
    env = _env("tray", 1)
    try:
        belief = (Entity("red cube", "box", (0.04, 0.04, 0.04), (0.5, 0.0, 0.02), 0.0),)
        view = env.plan_view(belief, (Keyframe((0.5, 0.0, 0.2), 0.0, "open", None),))
        true = {e["name"]: e for e in view["true_start"]}
        assert set(true) == {"red_cube", "blue_cube", "green_cube"}
        red = true["red_cube"]
        assert np.allclose(red["xyz"], env.world.object_pos("red_cube"), atol=1e-6)
        assert red["size"] == pytest.approx([0.04, 0.04, 0.04]) and red["shape"] == "box"
        assert red["color"] == pytest.approx([0.9, 0.1, 0.1, 1.0])
        assert view["true_match"] == {"red_cube": "red cube"}
        fixtures = {e["name"] for e in view["fixtures"]}
        assert "table" in fixtures and "tray_floor" in fixtures
        assert view["believed"][0]["name"] == "red cube"
    finally:
        env.close()


@pytest.mark.assets
def test_drawer_fixtures_follow_the_start_slide():
    env = _env("drawer", 0, task="close_drawer")  # the drawer starts 12 cm out
    try:
        view = env.plan_view((), (Keyframe((0.5, 0.0, 0.2), 0.0, "open", None),))
    finally:
        env.close()
    fx = {e["name"]: e for e in view["fixtures"]}
    plain = {e.name: e for e in SCENES["drawer"].fixtures}
    assert fx["drawer_handle"]["xyz"][0] == pytest.approx(plain["drawer_handle"].xyz[0] - 0.12, abs=1e-3)
    assert fx["cabinet_top"]["xyz"] == pytest.approx(list(plain["cabinet_top"].xyz))


@pytest.mark.assets
def test_round_mesh_objects_are_cylinders_in_the_true_start():
    if not SCENES["kitchen"].meshes:
        pytest.skip("no meshes")
    from armmind.sim.meshes import have_meshes
    if not have_meshes(set(SCENES["kitchen"].meshes)):
        pytest.skip("scanned objects missing")
    env = _env("kitchen", 0)
    try:
        view = env.plan_view((), (Keyframe((0.5, 0.0, 0.2), 0.0, "open", None),))
    finally:
        env.close()
    shapes = {e["name"]: e["shape"] for e in view["true_start"]}
    assert shapes["bowl"] == "cylinder" and shapes["plate"] == "cylinder" and shapes["bottle"] == "box"


@pytest.mark.assets
def test_agent_parts_and_the_finger_gap():
    env = _env("cubes", 0)
    try:
        belief = (Entity("red cube", "box", (0.04, 0.05, 0.04), (0.5, -0.1, 0.02), 0.0),
                  Entity("wide box", "box", (0.12, 0.10, 0.04), (0.5, 0.1, 0.02), 0.0))
        kfs = (Keyframe((0.5, -0.1, 0.15), 0.0, "open", None),
               Keyframe((0.5, -0.1, 0.02), 0.0, "closed", "Red_Cube"),
               Keyframe((0.5, 0.1, 0.02), 0.0, "closed", "wide box"),
               Keyframe((0.5, 0.1, 0.02), 0.0, "closed", None),
               Keyframe((0.5, 0.1, 0.02), 0.0, "closed", "ghost"))
        view = env.plan_view(belief, kfs)
    finally:
        env.close()
    agent = view["agent"]
    parts = {p["name"]: p for p in agent["parts"]}
    assert set(parts) == {"hand", "finger_left", "finger_right"}
    assert parts["hand"]["slide"] == 0 and {parts["finger_left"]["slide"], parts["finger_right"]["slide"]} == {1, -1}
    assert agent["slide_axis"] == [1.0, 0.0, 0.0]
    # The fingers reach 1 cm below the agent point, and the hand sits above the fingers.
    f = parts["finger_left"]
    assert f["xyz"][2] - f["size"][2] / 2 == pytest.approx(-0.01, abs=2e-3)
    assert parts["hand"]["xyz"][2] - parts["hand"]["size"][2] / 2 >= f["xyz"][2] + f["size"][2] / 2 - 0.005
    gaps = [k["gap"] for k in view["track"]["keyframes"]]
    assert gaps == pytest.approx([0.08, 0.04, 0.08, 0.0, 0.0])  # carrying: the narrow width, capped at 0.08
    assert [k["carrying"] for k in view["track"]["keyframes"]] == [None, "red cube", "wide box", None, None]
    assert view["track"]["start"] == {"xyz": pytest.approx([0.3, 0.0, 0.4], abs=0.005), "yaw": 0.0,
                                      "state": "open", "gap": 0.08}  # the true start pose of the arm


@pytest.mark.assets
def test_track_times_match_the_real_arm():
    from armmind.sim.arm import Arm
    from armmind.types import Action
    env = _env("cubes", 0)
    try:
        kfs = (Keyframe((0.45, 0.25, 0.30), 40.0, "open", None, "a"),
               Keyframe((0.45, 0.25, 0.20), 40.0, "closed", None, "b"),
               Keyframe((0.60, -0.30, 0.25), -30.0, "closed", None, "c"),
               Keyframe((0.60, -0.30, 0.25), -30.0, "open", None, "d"),
               Keyframe((0.9, 0.0, 0.3), 0.0, "open", None, "out of reach: clamped"))
        track = env.plan_view((), kfs)["track"]
        w = env.world
        arm = Arm(w)
        t0, done = w.time, []
        state = "open"
        for k in kfs:
            arm.act(Action("move_to", k.xyz, yaw=k.yaw))
            if k.state != state:
                arm.act(Action("open" if k.state == "open" else "close"))
                state = k.state
            done.append(w.time - t0)
    finally:
        env.close()
    assert [k["t_done"] for k in track["keyframes"]] == pytest.approx(done, abs=0.03)
    assert track["keyframes"][-1]["xyz"] == pytest.approx([0.75, 0.0, 0.3])  # clamped as move_to does
    assert track["duration"] == pytest.approx(done[-1], abs=0.03)


@pytest.mark.assets
def test_plan_view_is_plain_json():
    import json
    env = _env("buttons", 0)
    try:
        view = env.plan_view((), (Keyframe((0.62, 0.0, 0.1), 0.0, "closed", None),), [{"ok": True, "checks": {}}])
    finally:
        env.close()
    assert json.loads(json.dumps(view)) == view
    assert view["marks"] == [{"ok": True, "checks": {}}] and view["version"] == 1


# ---- The page player ----
def test_player_has_no_arm_or_mujoco_names():
    js = (ROOT / "web" / "plan3d.js").read_text().lower()
    for word in ("mujoco", "panda", "franka", "gripper", "finger", "cube", "tray", "drawer", "button", "front",
                 "\"top\"", "'top'", "red_", "open", "closed", "kitchen"):
        assert word not in js, word


def test_page_loads_one_exact_three_version():
    html = (ROOT / "web" / "index.html").read_text()
    urls = re.findall(r'src="(https://[^"]+)"', html)
    three = [u for u in urls if "three" in u]
    assert three == ["https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"]
    assert '<script src="plan3d.js"></script>' in html
    assert html.index("three.min.js") < html.index("plan3d.js") < html.index("app.js")
