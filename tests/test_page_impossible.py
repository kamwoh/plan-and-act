"""v1.16 page: a plan that Claude stops as not possible."""
from tests.test_page_plan import _body, _web


def test_a_not_possible_plan_has_a_red_entry_and_no_execute():
    js = _web("app.js")
    assert 'entry("impossible", "✕ Claude stopped: not possible")' in _body(js, "function impossibleEntry(reason)")
    assert "planImpossible ? [] : ev.keyframes" in js  # no keyframes: planCheckEntry adds no Execute button
    assert "if (planImpossible) addEntry(impossibleEntry(planImpossible));" in js
    assert 'setStatus("not possible", "bad")' in _body(js, "function onEvent(ev, replay = false)")
    assert ".entry.impossible { border-left-color: var(--bad);" in _web("style.css")


def test_the_loop_end_says_claude_gave_up():
    end = _body(_web("app.js"), "function loopEndEntry(ev)")
    assert "`Claude gave up in round ${gaveUp.round}: ${gaveUp.reason}`" in end
