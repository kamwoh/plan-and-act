"""Static checks of the v1.12 page: no Start button, the Auto switch, Rounds, Re-plan and the loop end."""
import re
import subprocess

import pytest

from tests.test_page_plan import WEB, _body, _web


@pytest.mark.parametrize("name", sorted(p.name for p in WEB.glob("*.js")))
def test_every_page_script_parses(name):
    subprocess.run(["node", "--check", str(WEB / name)], check=True)


def test_enter_in_the_instruction_runs_plan():
    js = _web("app.js")
    submit = js[js.index('$("run").addEventListener("submit"'):]
    submit = submit[:submit.index("});")]
    assert "startPlan()" in submit and "preventDefault" in submit


def test_rounds_replaces_max_steps():
    html = _web("index.html")
    assert 'id="max_steps"' not in html
    assert re.search(r'<input id="rounds" type="number" value="3" min="1" max="30">', html)
    body = _body(_web("app.js"), "function planState()")
    assert 'auto: $("auto").checked' in body and 'rounds: roundsValue($("rounds").value)' in body


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
    assert "const MAX_ROUND = 30" in js


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


def test_the_page_reads_the_auto_loop_from_the_state():
    js = _web("app.js")
    assert "autoLoop = Boolean(state.auto)" in _body(js, "async function renderPreview")
    sync = _body(js, "async function syncState")
    assert "/api/state" in sync and "autoLoop = Boolean(state.auto)" in sync
    assert "syncState()" in _body(js, "function connect")


def test_exec_start_sets_the_round_of_the_plan():
    handler = _body(_web("app.js"), "function onEvent(ev, replay = false)")
    start = handler[handler.index('ev.kind === "exec_start"'):handler.index('ev.kind === "exec_end"')]
    assert "planRound = ev.round || 1" in start


def test_prompt_button_says_it_edits_the_step_prompt():
    html = _web("index.html")
    button = re.search(r'<button[^>]*id="prompt_btn"[^>]*>', html).group(0)
    assert "step-by-step" in button and "/api/run" in button and "Plan uses its own prompt" in button
    head = re.search(r'<h2 id="prompt_title">([^<]*)</h2>', html).group(1)
    assert "step-by-step" in head.lower()
    dialog = _body(html, '<dialog id="prompt_dialog"', "</dialog>")
    assert "Plan uses its own prompt" in dialog


def test_a_second_execute_revokes_the_old_end_images():
    handler = _body(_web("app.js"), "function onEvent(ev, replay = false)")
    end = handler[handler.index('ev.kind === "exec_end"'):handler.index('ev.kind === "loop_end"')]
    assert "URL.revokeObjectURL" in end
    assert end.index("URL.revokeObjectURL") < end.index("stepImages[key] = ")


@pytest.mark.parametrize("value, want", [("", 3), ("0", 1), ("9", 9), ("31", 30), ("2", 2), ("abc", 3), ("2.6", 3)])
def test_rounds_value_falls_back_and_clamps(value, want):
    body = _body(_web("app.js"), "function roundsValue")
    script = f"const MAX_ROUND = 30;\n{body}\n}}\nconsole.log(roundsValue({value!r}));"
    out = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True).stdout
    assert int(out) == want
    assert 'rounds: roundsValue($("rounds").value)' in _body(_web("app.js"), "function planState()")
