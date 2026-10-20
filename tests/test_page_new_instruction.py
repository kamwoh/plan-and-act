"""Static checks of the v1.17 page: a new instruction goes to Claude in the same chat (see test_page_plan_goes_on)."""
from tests.test_page_plan import _body, _web


def test_the_page_keeps_the_newest_instruction_of_the_chain():
    js = _web("app.js")
    handler = _body(js, "function onEvent(ev, replay = false)")
    assert handler.count('planInstruction = ev.instruction || "";') == 2  # run_start, and exec_start of another tab
    assert "labelPlanButton();" in handler


def test_the_round_divider_shows_a_new_instruction():
    js = _web("app.js")
    divider = _body(js, "function roundDivider")
    assert "· new instruction" in divider and '"round-instr"' in divider
    assert "startRound(planMode, ev.new_instruction ? ev.instruction : null)" in _body(js, "function onEvent(ev, replay = false)")
    assert ".round-instr" in _web("style.css")


def test_a_run_of_another_tab_or_of_history_puts_its_instruction_in_the_box():
    js = _web("app.js")
    show = _body(js, "function showInstruction")
    assert '$("instruction").value = planInstruction;' in show and "showMode();" in show
    handler = _body(js, "function onEvent(ev, replay = false)")
    assert "const own = !replay && (ownStart || (autoLoop && chained));" in handler
    assert "ownStart = false;" in handler and "if (!own) showInstruction();" in handler
    assert handler.count("showInstruction();") == 2  # run_start, and exec_start of another tab
    for name in ("async function startPlan", "async function replan("):
        body = _body(js, name)
        assert "ownStart = true;" in body and "else ownStart = false;" in body


def test_the_loop_end_text_uses_the_absolute_round():
    handler = _body(_web("app.js"), "function onEvent(ev, replay = false)")
    assert "const last = ev.round ?? ev.rounds_run;" in handler
    assert "`Claude says done in round ${last}`" in handler and "`solved in round ${last}`" in handler
