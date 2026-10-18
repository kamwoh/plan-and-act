"""The camera data of a normal run, and the projection and the path overlay of the page."""
import json
import subprocess
from pathlib import Path

import pytest

from armmind.agent.loop import Agent, AgentConfig
from armmind.llm.fake import FakeBackend
from armmind.scenes import SCENES
from tests.fakes import Recorder, direct

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "tests" / "js"


def _env(scene="cubes", seed=0):
    from armmind.env import MujocoEnv
    return MujocoEnv(SCENES[scene], None, seed, "")


@pytest.mark.assets
@pytest.mark.parametrize("scene", ["cubes", "drawer"])
async def test_run_start_has_the_cameras_of_the_scene(scene):
    from armmind.sim.planview import cameras
    env, rec = _env(scene), Recorder()
    try:
        await Agent(FakeBackend([]), AgentConfig(max_steps=1)).run(env, direct, rec)
        want = cameras(env.world, env.cams.w, env.cams.h)
    finally:
        env.close()
    got = dict(rec.events)["run_start"]["cameras"]
    assert got == want
    assert {c["name"] for c in got} == {"front", "top"} and all((c["width"], c["height"]) == (640, 480) for c in got)
    front = next(c for c in got if c["name"] == "front")
    assert front["label"] == SCENES[scene].camera_labels["front"]
    assert front["grid"] == (scene != "drawer")  # the drawer scene has its own side camera


async def test_a_run_without_camera_data_has_none_in_run_start():
    from tests.fakes import FakeEnv
    rec = Recorder()
    await Agent(FakeBackend([]), AgentConfig(max_steps=1)).run(FakeEnv(), direct, rec)
    assert "cameras" not in dict(rec.events)["run_start"]


@pytest.mark.assets
@pytest.mark.parametrize("scene", ["cubes", "drawer"])
def test_the_page_projection_matches_mujoco(scene, tmp_path):
    from armmind.sim.planview import cameras
    env = _env(scene)
    try:
        cube = env.world.objects[0].name
        points = [[x, y, 0.0] for x in (-1.0, 1.0) for y in (-1.0, 1.0)]  # the corners of the table plane
        points += [[0.2, -0.4, 0.0], [0.8, 0.4, 0.0], [float(v) for v in env.world.object_pos(cube)]]
        cases = [{"camera": c, "points": points, "pixels": env.cams.project(c["name"], points).tolist()}
                 for c in cameras(env.world, env.cams.w, env.cams.h)]
    finally:
        env.close()
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    r = subprocess.run(["node", str(JS / "camproj_check.js"), str(tmp_path / "cases.json")],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr


async def test_the_step_events_hold_what_the_overlay_draws():
    from tests.fakes import FakeEnv, good
    rec = Recorder()
    acts = [{"type": "move_to", "xyz": [0.5, 0.1, 0.2], "yaw": 90}, {"type": "close", "expect": {"holding": "red cube"}},
            {"type": "move_to", "xyz": [0.4, 0.0, 0.3]}, {"type": "wait", "seconds": 1.0}]
    await Agent(FakeBackend([good(acts)]), AgentConfig(max_steps=1)).run(FakeEnv(has_check=False), direct, rec)
    obs, reply = dict(rec.events)["observation"], dict(rec.events)["reply"]
    assert obs["tcp"] == [0.3, 0.0, 0.4] and obs["yaw"] == 0.0  # where the path starts
    assert reply["action_data"] == [{"type": "move_to", "xyz": [0.5, 0.1, 0.2], "yaw": 90.0},
                                    {"type": "close", "expect": "red cube"},
                                    {"type": "move_to", "xyz": [0.4, 0.0, 0.3]}, {"type": "wait", "seconds": 1.0}]
    assert [d["index"] for k, d in rec.events if k == "action"] == [0, 1, 2, 3]


# ---- The page ----
WEB = ROOT / "web"


def _body(js, start, end="\n}\n"):
    body = js[js.index(start):]
    return body[:body.index(end)]


def test_every_page_script_parses():
    scripts = sorted(WEB.glob("*.js"))
    assert {p.name for p in scripts} >= {"app.js", "plan3d.js", "camproj.js", "overlay.js"}
    for p in scripts:
        subprocess.run(["node", "--check", str(p)], check=True)


def test_the_overlay_state_logic_and_layout():
    r = subprocess.run(["node", str(JS / "overlay_check.js")], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_overlay_and_the_projection_hold_no_arm_names():
    import re
    for name in ("overlay.js", "camproj.js"):
        js = (WEB / name).read_text()
        words = r"\b(arm|panda|franka|gripper|finger\w*|tcp|cube\w*|drawer|tray|move_to|grasp\w*)\b"
        bad = re.findall(words, js, re.IGNORECASE)
        assert not bad, f"{name}: {bad}"


def test_the_plan_video_labels_use_the_shared_projection():
    js = (WEB / "plan3d.js").read_text()
    labels = _body(js, "    drawLabels() {", "\n    }\n")
    assert "CamProj.project" in labels and ".project(this.camera)" not in js


def test_the_path_switch_is_on_by_default_and_its_storage_is_guarded():
    html, js = (WEB / "index.html").read_text(), (WEB / "app.js").read_text()
    assert '<input id="show_path" type="checkbox" checked>' in html
    for fn in ("function readShowPath()", "function writeShowPath(on)"):
        body = _body(js, fn)
        assert "localStorage" in body and "try {" in body and "catch (err)" in body
    assert "PATH_KEY" not in js.replace(_body(js, "function readShowPath()"), "").replace(
        _body(js, "function writeShowPath(on)"), "").replace('const PATH_KEY = "armmind.showPath";', "")


def test_the_overlay_clears_on_a_new_run_a_preview_and_a_reset():
    js = (WEB / "app.js").read_text()
    assert "clearPath();" in _body(js, "function clearPlan()")
    for caller in ("async function loadPreview()", "function beginRun(text)"):
        assert "clearPlan();" in _body(js, caller)
    handler = _body(js, "function onEvent(ev, replay = false)")
    assert 'ev.kind === "action"' in handler and "setPath(" in handler and "onPathStep(ev)" in handler
