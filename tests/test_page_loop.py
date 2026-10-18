"""Static checks of the v1.12 page: no Start button, the Auto switch, Rounds, Re-plan and the loop end."""
import re
import subprocess

import pytest

from tests.test_page_plan import WEB, _body, _web


@pytest.mark.parametrize("name", sorted(p.name for p in WEB.glob("*.js")))
def test_every_page_script_parses(name):
    subprocess.run(["node", "--check", str(WEB / name)], check=True)


def test_the_page_has_no_start_button():
    html, js = _web("index.html"), _web("app.js")
    assert 'id="start_btn"' not in html and not re.search(r">\s*Start\s*<", html)
    assert '"/api/run"' not in js and "start_btn" not in js
    assert 'type="submit" class="btn primary" id="plan_btn"' in html


def test_enter_in_the_instruction_runs_plan():
    js = _web("app.js")
    submit = js[js.index('$("run").addEventListener("submit"'):]
    submit = submit[:submit.index("});")]
    assert "startPlan()" in submit and "preventDefault" in submit


def test_rounds_replaces_max_steps():
    html = _web("index.html")
    assert 'id="max_steps"' not in html
    assert re.search(r'<input id="rounds" type="number" value="3" min="1" max="5">', html)
    body = _body(_web("app.js"), "async function startPlan")
    assert 'auto: $("auto").checked' in body and 'rounds: Number($("rounds").value)' in body


def test_the_auto_switch_storage_uses_try_catch():
    js = _web("app.js")
    for name in ("function readAuto", "function writeAuto"):
        body = _body(js, name)
        assert "localStorage" in body and "try {" in body and "catch (err)" in body
    assert '$("auto").checked = readAuto()' in js


def test_replan_button_follows_the_execute_rules():
    js = _web("app.js")
    button = _body(js, "function replanButton")
    assert "exec-btn" in button and "Re-plan" in button and "running || !planRunId" in button
    assert "/replan`" in _body(js, "async function replan")
    handler = _body(js, "function onEvent(ev, replay = false)")
    end = handler[handler.index('ev.kind === "exec_end"'):handler.index('ev.kind === "loop_end"')]
    assert "planRound < MAX_ROUND" in end and "replanButton()" in end
    assert "const MAX_ROUND = 5" in js


def test_the_page_follows_the_auto_loop():
    js = _web("app.js")
    handler = _body(js, "function onEvent(ev, replay = false)")
    assert 'ev.kind === "loop_end"' in handler and "loopEndEntry(ev)" in handler
    assert "autoLoop = Boolean(ev.auto_rounds)" in handler
    assert "if (!replay && !autoLoop) setRunning(false)" in handler  # busy from round to round
    body = _body(js, "function loopEndEntry")
    assert "rounds_run" in body and "reason" in body and "success" in body


def test_rounds_show_in_plan_entries_and_history_rows():
    js = _web("app.js")
    assert "roundText(planRound)" in _body(js, "function onPlan")
    assert "round > 1" in _body(js, "function roundText")
    row = _body(js, "function historyRow")
    assert "run.round > 1" in row and "run.parent" in row


def test_plan_lessons_show_in_the_claude_entry():
    js = _web("app.js")
    assert "ev.lessons" in _body(js, "function onPlan")
    handler = _body(js, "function onEvent(ev, replay = false)")
    plan = handler[handler.index('ev.kind === "plan"'):handler.index('ev.kind === "exec_start"')]
    assert "runLessons.push" in plan and "showLessons()" in plan


def test_enter_in_the_instruction_submits_the_form():
    js = _web("app.js")
    key = js[js.index('$("instruction").addEventListener("keydown"'):]
    key = key[:key.index("});")]
    assert '"Enter"' in key and "e.shiftKey" in key and '$("run").requestSubmit()' in key
