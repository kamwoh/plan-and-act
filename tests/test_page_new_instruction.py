"""Static checks of the v1.17 page: Plan from here gives Claude a new instruction in the same chat."""
from tests.test_page_plan import _body, _web

TITLE = "Claude gets the new instruction in the same chat, and the arm goes on from where it is now"


def test_the_page_keeps_the_newest_instruction_of_the_chain():
    js = _web("app.js")
    handler = _body(js, "function onEvent(ev, replay = false)")
    assert handler.count('planInstruction = ev.instruction || "";') == 2  # run_start, and exec_start of another tab
    assert "showReplanLabels();" in handler


def test_the_replan_button_says_plan_from_here_for_a_new_instruction():
    js = _web("app.js")
    body = _body(js, "function newInstruction")
    assert '$("instruction").value.trim()' in body and "planInstruction" in body
    label = _body(js, "function replanLabel")
    assert '"▶ Plan from here"' in label and '"↻ Re-plan"' in label and "b.dataset.run === planRunId" in label
    assert TITLE in js
    labels = _body(js, "function showReplanLabels")
    assert 'querySelectorAll(".replan-btn")' in labels and "replanLabel(b)" in labels
    assert "replanLabel(b);" in _body(js, "function replanButton")
    assert "showReplanLabels();" in _body(js, "function showMode")  # the box changes: the label follows
    assert '$("instruction").addEventListener("input", showMode);' in js


def test_plan_from_here_sends_the_instruction_and_the_auto_loop():
    body = _body(_web("app.js"), "async function replan")
    assert "const text = newInstruction();" in body
    assert 'mode = text ? "continue"' in body
    assert "instruction: text, auto: $(\"auto\").checked, rounds: roundsValue($(\"rounds\").value)" in body
    assert "/replan`, body)" in body


def test_the_round_divider_shows_a_new_instruction():
    js = _web("app.js")
    divider = _body(js, "function roundDivider")
    assert "· new instruction" in divider and '"round-instr"' in divider
    assert "startRound(planMode, ev.new_instruction ? ev.instruction : null)" in _body(js, "function onEvent(ev, replay = false)")
    assert ".round-instr" in _web("style.css")


def test_a_plan_without_an_execution_has_a_plan_from_here_button():
    js = _web("app.js")
    check = _body(js, "function planCheckEntry")
    assert "replanButton(true)" in check
    assert check.index("replanButton(true)") < check.index("if (!keyframes.length) return item;")
    label = _body(js, "function replanLabel")
    assert "if (b.dataset.onlyNew) b.hidden = !fromHere || planExecuted;" in label
    handler = _body(js, "function onEvent(ev, replay = false)")
    assert "planExecuted = false;" in handler and "planExecuted = true;" in handler


def test_plan_from_here_after_an_execution_error_sends_continue():
    body = _body(_web("app.js"), "async function replan")
    assert 'const mode = text ? "continue" : fromStart' in body  # the server starts it where the round started
