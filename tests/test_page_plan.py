"""Static checks of the plan parts of the page."""
import re
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"


def _web(name):
    return (WEB / name).read_text()


def _body(js, start, end="\n}\n"):
    body = js[js.index(start):]
    return body[:body.index(end)]


def test_app_js_parses():
    subprocess.run(["node", "--check", str(WEB / "app.js")], check=True)
    subprocess.run(["node", "--check", str(WEB / "plan3d.js")], check=True)


def test_plan_button_comes_first_with_the_auto_and_continue_switches():
    buttons = _body(_web("index.html"), '<div class="buttons">', "</form>")
    ids = re.findall(r'<(?:button|input)[^>]*id="([a-z_]+)"', buttons)
    assert ids == ["plan_btn", "auto", "continue", "stop", "lessons_btn", "history_btn", "prompt_btn"]


def test_camera_boxes_hold_the_plan_canvases_and_a_player_bar():
    html = _web("index.html")
    for cam in ("front", "top"):
        fig = _body(html, f'<figure class="cam" id="cam_{cam}">', "</figure>")
        assert f'id="plan_{cam}"' in fig and f'id="plan_{cam}_labels"' in fig
    bar = _body(html, '<div id="player"', "\n      </div>\n")
    for part in ('id="play_btn"', 'id="play_time"', 'id="play_slider"', 'id="play_kf"', 'data-mode="imagined"',
                 'data-mode="camera"'):
        assert part in bar
    assert 'data-mode="real"' not in html and 'data-mode="both"' not in html  # v1.16: two view buttons only


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

def test_clear_plan_forgets_the_plan_and_disables_execute():
    js = _web("app.js")
    body = _body(js, "function clearPlan")
    assert "planRunId = null" in body and ".exec-btn" in body
    assert "!planRunId" in _body(js, "function setRunning")
    handler = _body(js, "function onEvent(ev, replay = false)")
    start = handler[handler.index('ev.kind === "exec_start"'):handler.index('ev.kind === "exec_end"')]
    assert "chooseScene(ev.scene, true, !replay)" in start


def test_scripts_never_wait_for_the_cdn():
    html = _web("index.html")
    tags = re.findall(r"<script[^>]*>", html)
    assert len(tags) == 5
    assert " async" in tags[0] and "three.min.js" in tags[0]  # a slow CDN never holds the page
    assert all(" defer" in t for t in tags[1:])
    for first, then in (("camproj.js", "plan3d.js"), ("camproj.js", "overlay.js"), ("plan3d.js", "app.js"),
                        ("overlay.js", "app.js")):
        assert html.index(first) < html.index(then)


def test_the_player_reads_three_only_when_it_is_made():
    js = _web("plan3d.js")
    assert "const T = window.THREE" not in js
    assert "T = window.THREE" in _body(js, "player(slots)", "\n    },\n")
    assert "Plan3D.loaded()" in _body(_web("app.js"), "function ensurePlayer")


def test_sleep_tells_the_play_button():
    assert "this.seek(this.time)" in _body(_web("plan3d.js"), "sleep()", "\n    }\n")


def test_without_a_player_the_bar_hides_and_the_message_stays():
    body = _body(_web("app.js"), "function showPlan")
    assert '$("player").hidden = true' in body and "WebGL" in body and "addEntry(" in body


def test_the_camera_view_stops_the_hidden_video():
    js = _web("app.js")
    assert "player.sleep()" in _body(js, "function setViewMode")
    p3 = _web("plan3d.js")
    assert "sleep()" in p3 and "wake()" in p3


def test_a_slider_drag_reads_the_new_time_before_the_pause():
    body = _body(_web("app.js"), '$("play_slider").addEventListener("input"', "\n});\n")
    assert body.index('const t = Number($("play_slider").value)') < body.index("player.pause()")
    assert "player.seek(t)" in body


# ---- v1.11: predicted moves in the video and the Plan check ----
NODE_MOVES = r"""
const vm = require("vm"), fs = require("fs");
const ctx = { window: {}, console };
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), ctx);
const P = ctx.window.Plan3D;
const move = { name: "green cube", xyz: [0.6, 0.16, 0.02], yaw: 0 };
const view = {
  track: { start: { xyz: [0.6, 0.27, 0.02], yaw: 0, gap: 0 },
           keyframes: [{ xyz: [0.6, 0.19, 0.02], yaw: 0, carrying: null, gap: 0, changes: false, t: 1, t_done: 1.3,
                         moves: [move] }], duration: 1.3 },
  believed: [{ name: "green cube", xyz: [0.6, 0.21, 0.02], yaw: 0 }],
};
const c = P.carriers(view);
const at = (t) => [P.posesAt(view.track, view.believed, t, c.believed, c.believedMoves)["green cube"].xyz[1]];
console.log(JSON.stringify([at(0), at(0.5), at(1.3)]));
"""


def test_the_belief_follows_the_moves():
    import json
    out = subprocess.run(["node", "-e", NODE_MOVES, str(WEB / "plan3d.js")], check=True, capture_output=True,
                         text=True).stdout
    (b0,), (b1,), (b2,) = json.loads(out)
    assert b0 == pytest.approx(0.21) and b1 == pytest.approx(0.185) and b2 == pytest.approx(0.16)


def test_a_moves_keyframe_has_its_own_marker_and_text():
    js = _web("plan3d.js")
    assert "MOVE_COLOR" in _body(js, "    addPath(view) {", "\n    }\n")
    app = _web("app.js")
    assert "moves" in _body(app, "function showPlayerTime")
    body = _body(app, "function planCheckEntry")
    assert 'field("moves"' in body and "movesText" in body
    assert "planCheckEntry(ev.scores || {}, planImpossible ? [] : ev.keyframes || [], Boolean(ev.done))" in app


def test_exec_end_adds_the_prediction_entry_also_in_a_replay():
    app = _web("app.js")
    handler = _body(app, "function onEvent(ev, replay = false)")
    end = handler[handler.index('ev.kind === "exec_end"'):handler.index('ev.kind === "loop_end"')]
    assert "const pred = ev.prediction ? predictionEntry(ev.prediction) : null" in end and "addEntry(pred)" in end
    assert "replay" not in end.split("predictionEntry")[0].split("onExecEnd")[1]  # a History load shows it too
    body = _body(app, "function predictionEntry")
    for part in ("predicted move", "real move", "PRED_LIMIT_CM", "s.partial", "agree", "p.keyframes"):
        assert part in body
    assert "const PRED_LIMIT_CM = 3;" in app


def test_the_plan_check_shows_predicted_poses_and_the_prediction_shows_missed_objects():
    app = _web("app.js")
    check = _body(app, "function planCheckEntry")
    assert "at the predicted pose" in check and "kf-muted" in check
    assert "shownPredicted.has(n)" in check  # the note shows once, not on every later keyframe
    assert "Real start" not in check
    pred = _body(app, "function predictionEntry")
    assert "not in the belief: " in pred and "xy more than 2 cm" in pred
