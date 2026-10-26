"""The camera data of a normal run, and the projection and the path overlay of the page."""
import json
import re
import subprocess
from pathlib import Path

import pytest

from plan_and_act.agent.loop import Agent, AgentConfig
from plan_and_act.llm.fake import FakeBackend
from plan_and_act.scenes import SCENES
from tests.fakes import Recorder, direct

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "tests" / "js"


def _env(scene="cubes", seed=0):
    from plan_and_act.env import MujocoEnv
    return MujocoEnv(SCENES[scene], None, seed, "")


def _seen_centre(pos, camera):
    x, y, z = (float(v) for v in pos)
    return [x, y, z + HALF_CUBE] if camera == "top" else [x, y, z]


@pytest.mark.assets
@pytest.mark.parametrize("scene", ["cubes", "drawer"])
async def test_run_start_has_the_cameras_of_the_scene(scene):
    from plan_and_act.sim.planview import cameras
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


def _colour_centroid(img, rgba):
    """The centre of the pixels of a pure-coloured object, in the pixel convention of the projection: the
    centre of pixel (0, 0) is at (0.5, 0.5)."""
    import numpy as np
    ch = int(np.argmax(rgba[:3]))
    a, b = (i for i in range(3) if i != ch)
    px = img.astype(int)
    mask = (px[..., ch] > 60) & (px[..., ch] > px[..., a] + 70) & (px[..., ch] > px[..., b] + 70)
    ys, xs = np.nonzero(mask)
    assert len(xs) > 100, "the object is not in the image"
    # The arm has a few pixels of the same colour. Keep only the pixels near the median of the mask.
    mx, my = np.median(xs), np.median(ys)
    near = np.hypot(xs - mx, ys - my) < 25
    return [float(xs[near].mean()) + 0.5, float(ys[near].mean()) + 0.5]


# The page projection against MuJoCo's own render. The front camera sees the top and the sides of a cube, so
# its colour centroid lies near the cube centre. The top camera sees mostly the top face, so the test projects
# the centre of that face. The front camera of the drawer scene is its side camera.
CUBES = {"cubes": ["red_cube", "blue_cube", "green_cube"], "drawer": ["red_cube"]}
HALF_CUBE = 0.02


@pytest.mark.assets
@pytest.mark.parametrize("scene", ["cubes", "drawer"])
@pytest.mark.parametrize("seed", [0])
def test_the_page_projection_matches_the_mujoco_render(scene, seed, tmp_path):
    from plan_and_act.sim.planview import cameras
    env = _env(scene, seed)
    try:
        cases = []
        for c in cameras(env.world, env.cams.w, env.cams.h):
            img = env.cams.render(c["name"])
            objs = [o for o in env.world.objects if o.name in CUBES[scene]]
            cases.append({"camera": c, "limit": 2.0,
                          "points": [_seen_centre(env.world.object_pos(o.name), c["name"]) for o in objs],
                          "pixels": [_colour_centroid(img, [float(v) for v in o.rgba.split()]) for o in objs]})
    finally:
        env.close()
    assert {c["camera"]["name"] for c in cases} == {"front", "top"}
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
        _body(js, "function writeShowPath(on)"), "").replace('const PATH_KEY = "plan-and-act.showPath";', "")


def test_the_overlay_clears_on_a_new_run_a_preview_and_a_reset():
    js = (WEB / "app.js").read_text()
    assert "clearPath();" in _body(js, "function clearPlan()")
    for caller in ("async function loadPreview()", "function beginRun(text)"):
        assert "clearPlan();" in _body(js, caller)
    handler = _body(js, "function onEvent(ev, replay = false)")
    assert 'ev.kind === "action"' in handler and "setPath(" in handler and "onPathStep(ev)" in handler


async def test_a_camera_data_error_still_gives_run_start_and_run_end(caplog):
    from tests.fakes import FakeEnv

    class BrokenCameras(FakeEnv):
        def cameras(self):
            raise RuntimeError("no renderer")

    rec = Recorder()
    status = await Agent(FakeBackend([]), AgentConfig(max_steps=1)).run(BrokenCameras(), direct, rec)
    assert rec.kinds()[0] == "run_start" and rec.kinds()[-1] == "run_end" and status == "llm_error"
    assert "cameras" not in dict(rec.events)["run_start"]
    assert "no camera data" in caplog.text


def test_a_plan_from_another_tab_clears_the_old_path():
    handler = _body((WEB / "app.js").read_text(), "function onEvent(ev, replay = false)")
    other = handler[handler.index("if (ev.run_id !== planRunId) {"):]
    other = other[:other.index("}")]
    assert "clearPath();" in other and "runKey++;" in other


def test_the_path_shows_only_over_an_image_of_its_run():
    js = (WEB / "app.js").read_text()
    assert "imageKey = runKey;" in _body(js, "function showImage(id, src, free = false)")
    assert "key: runKey" in _body(js, "function setPath(step, cams, start, actions, byKeyframe = false)")
    assert "imageKey === stepPath.key" in _body(js, "function drawPath()")
    assert "runKey++;" in _body(js, "async function loadPreview()")


def test_the_overlay_text_keeps_its_screen_size():
    js = (WEB / "app.js").read_text()
    assert "cam.width / shown" in _body(js, "function drawPath()") and "new ResizeObserver(" in js
    overlay = (WEB / "overlay.js").read_text()
    assert "function draw(canvas, cam, path, actionStates, scale = 1)" in overlay
    assert re.search(r"font = `600 \$\{\d+ \* k\}px", overlay) and re.search(r"font = `700 \$\{\d+ \* k\}px", overlay)


def test_the_label_scale_has_a_cap():
    js = (WEB / "overlay.js").read_text()
    assert "const MAX_SCALE = 2;" in js and "Math.min(scale, MAX_SCALE)" in js


def test_the_memo_card_has_only_the_lessons():
    html, js, css = (WEB / "index.html").read_text(), (WEB / "app.js").read_text(), (WEB / "style.css").read_text()
    memo = _body(html, '<section class="card memo">', "</section>")
    assert memo.count('class="memo-col"') == 1 and "<h2>Lessons</h2>" in memo
    for gone in ('id="todo"', 'id="notes"', 'id="first_plan"', "<h2>To-do</h2>", "<h2>Notes</h2>"):
        assert gone not in html, gone
    for gone in ('$("todo")', '$("notes")', '$("first_plan")', "showFirstPlan", "markClipped"):
        assert gone not in js, gone
    assert ".first-plan" not in css and ".todo" not in css and ".notes" not in css
    reply = _body(js, "function onReply(ev)")  # the step mode keeps its todo and notes in the chat
    assert 'field("notes"' in reply and 'field("todo"' in reply
