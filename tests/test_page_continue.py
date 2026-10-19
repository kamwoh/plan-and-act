"""Static checks of the v1.13 page: the Continue switch, the round modes and Claude's done verdict."""
import re

from tests.test_page_plan import _body, _web


def test_the_continue_switch_is_next_to_auto_and_on_by_default():
    html = _web("index.html")
    switches = html[html.index('<div class="loop-switches">'):]
    switches = switches[:switches.index("</div>")]
    assert 'id="auto"' in switches and re.search(r'<input id="continue" type="checkbox" checked>', switches)
    assert ">Continue</label>" in switches


def test_the_continue_storage_uses_try_catch_and_defaults_to_on():
    js = _web("app.js")
    for name in ("function readContinue", "function writeContinue"):
        body = _body(js, name)
        assert "localStorage" in body and "try {" in body and "catch (err)" in body
    assert '!== "false"' in _body(js, "function readContinue") and "return true;" in _body(js, "function readContinue")
    assert '$("continue").checked = readContinue()' in js


def test_the_switch_sets_the_mode_of_replan_and_of_the_auto_loop():
    js = _web("app.js")
    assert '$("continue").checked ? "continue" : "start"' in _body(js, "function roundMode")
    assert "mode: roundMode()," in _body(js, "async function startPlan")
    body = _body(js, "async function replan")
    assert "roundMode();" in body and "/replan`, body)" in body and "{ mode, max_keyframes:" in body


def test_a_plan_after_round_1_shows_its_mode():
    js = _web("app.js")
    assert '" · continue"' in _body(js, "function modeText") and '" · from start"' in _body(js, "function modeText")
    assert "modeText(planMode)" in _body(js, "function onPlan")
    handler = _body(js, "function onEvent(ev, replay = false)")
    assert "planMode = ev.round_mode || null;" in handler
    assert 'planMode !== "continue" ? endImages' in handler  # a continue round has no end images


def test_a_done_plan_has_no_execute_button():
    js = _web("app.js")
    check = _body(js, "function planCheckEntry")
    assert "Claude says the task is done." in check
    assert check.index("if (!keyframes.length) return item;") < check.index('"▶ Execute plan"')
    handler = _body(js, "function onEvent(ev, replay = false)")
    assert "Boolean(ev.done)" in handler and 'setStatus("Claude says the task is done"' in handler


def test_the_loop_end_shows_the_verdict():
    js = _web("app.js")
    body = _body(js, "function loopEndEntry")
    assert 'field("verdict"' in body and 'ev.judge === "claude"' in body
    assert "Claude says done in round" in _body(js, "function onEvent(ev, replay = false)")


def test_the_page_shows_claudes_verdict_next_to_the_auto_check():
    js = _web("app.js")
    handler = _body(js, "function onEvent(ev, replay = false)")
    assert 'ev.kind === "verdict"' in handler and "verdictEntry(ev)" in handler
    assert "!ev.verdict_next) setRunning(false)" in handler  # the page stays busy for the verdict call
    text = _body(js, "function verdictText")
    assert '"Claude: done"' in text and '"Claude: not done"' in text and "auto-check:" in text
    entry = _body(js, "function verdictEntry")
    assert "ev.reason" in entry and "disagree" in entry
    end = _body(js, "function loopEndEntry")
    assert "ev.verdicts" in end and "ev.agree" in end and "ev.compared" in end


def test_a_replan_after_an_execution_error_falls_back_to_from_start():
    js = _web("app.js")
    assert 'const mode = fromStart ? "start" : roundMode();' in _body(js, "async function replan")
    button = _body(js, "function replanButton")
    assert "if (!fromStart) return b;" in button and "starts from the start, also with Continue on" in button
    handler = _body(js, "function onEvent(ev, replay = false)")
    assert 'execError = ev.status === "error";' in handler
    assert handler.index('execError = ev.status === "error";') < handler.index("replanButton()")
