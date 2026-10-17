"""Static checks of the plan parts of the page."""
import re
import subprocess
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"


def _web(name):
    return (WEB / name).read_text()


def _body(js, start, end="\n}\n"):
    body = js[js.index(start):]
    return body[:body.index(end)]


def test_app_js_parses():
    subprocess.run(["node", "--check", str(WEB / "app.js")], check=True)
    subprocess.run(["node", "--check", str(WEB / "plan3d.js")], check=True)


def test_plan_button_sits_next_to_start():
    buttons = _body(_web("index.html"), '<div class="buttons">', "</div>")
    ids = re.findall(r'<button[^>]*id="([a-z_]+)"', buttons)
    assert ids == ["start_btn", "plan_btn", "stop", "lessons_btn", "history_btn", "prompt_btn"]


def test_camera_boxes_hold_the_plan_canvases_and_a_player_bar():
    html = _web("index.html")
    for cam in ("front", "top"):
        fig = _body(html, f'<figure class="cam" id="cam_{cam}">', "</figure>")
        assert f'id="plan_{cam}"' in fig and f'id="plan_{cam}_labels"' in fig
    bar = _body(html, '<div id="player"', "\n      </div>\n")
    for part in ('id="play_btn"', 'id="play_time"', 'id="play_slider"', 'id="play_kf"', 'data-mode="imagined"',
                 'data-mode="real"', 'data-mode="both"', 'data-mode="camera"'):
        assert part in bar


def test_plan_request_uses_the_same_fields_as_start():
    js = _web("app.js")
    body = _body(js, "async function startPlan")
    assert 'postJson("/api/plan", body)' in body
    for field in ("scene: sceneName", "instruction:", "seed:", "use_lessons:"):
        assert field in body
    assert "show_positions" not in body  # a plan never gets the true positions


def test_the_live_handler_knows_the_plan_events():
    handler = _body(_web("app.js"), "function onEvent(ev, replay = false)")
    for kind in ('"plan"', '"exec_start"', '"exec_end"'):
        assert f"ev.kind === {kind}" in handler
    assert "showPlan(ev.view)" in handler
    assert "clearPlan()" in handler  # a new run drops the old video


def test_history_load_still_goes_through_the_live_handler():
    body = _body(_web("app.js"), "async function loadRun")
    assert "onEvent(ev, true)" in body and "showPlan(" not in body


def test_history_rows_show_a_plan_badge():
    body = _body(_web("app.js"), "function historyRow")
    assert 'run.mode === "plan"' in body and "run.exec_status" in body


def test_plan_check_has_the_table_and_the_execute_button():
    js = _web("app.js")
    body = _body(js, "function planCheckEntry")
    for part in ("xy", "size", "yaw", "missing", "invented", "Execute plan", "XY_LIMIT_CM", "YAW_LIMIT_DEG"):
        assert part in body
    assert "const XY_LIMIT_CM = 2" in js and "const YAW_LIMIT_DEG = 15" in js
    assert "/execute`" in _body(js, "async function executePlan")


def test_reset_scene_seed_and_start_clear_the_plan():
    js = _web("app.js")
    assert "clearPlan()" in _body(js, "async function loadPreview")
    assert "clearPlan()" in _body(js, "function beginRun")


def test_a_replayed_execution_keeps_the_imagined_video():
    handler = _body(_web("app.js"), "function onEvent(ev, replay = false)")
    start = handler[handler.index('ev.kind === "exec_start"'):handler.index('ev.kind === "exec_end"')]
    live = start[start.index("if (!replay) {"):start.index("onExecStart(ev)")]
    assert 'setViewMode("camera")' in live  # only a live execution switches to the camera images


# ---- Review follow-ups ----
NODE_POSES = r"""
const vm = require("vm"), fs = require("fs");
const ctx = { window: {}, console };
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), ctx);
const P = ctx.window.Plan3D;
const kf = (xyz, carrying, t) => ({ xyz, yaw: 0, carrying, gap: 0, changes: false, t, t_done: t + 0.5 });
const view = {
  track: { start: { xyz: [0.3, 0, 0.4], yaw: 0, gap: 0.08 },
           keyframes: [kf([0.5, 0, 0.02], "red cube", 1), kf([0.5, 0, 0.3], "red cube", 3)], duration: 3.5 },
  believed: [{ name: "red cube", xyz: [0.5, 0, 0.02], yaw: 0 }],
  true_start: [{ name: "red_cube", xyz: [0.54, 0, 0.02], yaw: 0 }],
  true_match: { red_cube: "red cube" },
  marks: JSON.parse(process.argv[2]),
};
const c = P.carriers(view);
const b = P.posesAt(view.track, view.believed, 3.5, c.believed)["red cube"].xyz[2];
const t = P.posesAt(view.track, view.true_start, 3.5, c.truth)["red_cube"].xyz[2];
console.log(JSON.stringify([b, t]));
"""


def _poses(marks):
    import json
    out = subprocess.run(["node", "-e", NODE_POSES, str(WEB / "plan3d.js"), json.dumps(marks)], check=True,
                         capture_output=True, text=True).stdout
    return json.loads(out)


def test_real_start_moves_a_true_object_only_when_the_marks_say_it_is_held():
    held = [{"ok": True, "holds": "red_cube"}, {"ok": True, "holds": "red_cube"}]
    missed = [{"ok": False, "holds": None}, {"ok": True, "holds": None}]
    assert _poses(held) == [0.3, 0.3]
    assert _poses(missed) == [0.3, 0.02]  # the belief moves, the true cube stays
    assert _poses([]) == [0.3, 0.3]  # an old view without marks: the name match decides


def test_clear_plan_forgets_the_plan_and_disables_execute():
    js = _web("app.js")
    body = _body(js, "function clearPlan")
    assert "planRunId = null" in body and ".exec-btn" in body
    assert "!planRunId" in _body(js, "function setRunning")
    handler = _body(js, "function onEvent(ev, replay = false)")
    start = handler[handler.index('ev.kind === "exec_start"'):handler.index('ev.kind === "exec_end"')]
    assert "chooseScene(ev.scene, true, !replay)" in start


def test_scripts_are_deferred_in_order():
    html = _web("index.html")
    tags = re.findall(r"<script[^>]*>", html)
    assert all(" defer" in t for t in tags) and len(tags) == 3
    assert html.index("three.min.js") < html.index("plan3d.js") < html.index("app.js")


def test_without_a_player_the_bar_hides_and_the_message_stays():
    body = _body(_web("app.js"), "function showPlan")
    assert '$("player").hidden = true' in body and "WebGL" in body and "addEntry(" in body


def test_the_camera_view_stops_the_hidden_video():
    js = _web("app.js")
    assert "player.sleep()" in _body(js, "function setViewMode")
    p3 = _web("plan3d.js")
    assert "sleep()" in p3 and "wake()" in p3
