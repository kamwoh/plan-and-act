"""v1.16 page: the camera badge and the Execution entry show the outcome of an execution."""
import json
import subprocess

import pytest

from tests.test_page_plan import _body, _web

STOPPED = {"status": "stopped", "stopped": {"after": 5, "of": 12, "reason": "the gripper closed on nothing"}}


def badge(ev):
    js = _web("app.js")
    fn = "".join(_body(js, f"function {name}(ev)") + "\n}\n" for name in ("stopKeyframe", "stopText", "outcomeBadge"))
    out = subprocess.run(["node", "-e", f"{fn}\nconsole.log(JSON.stringify(outcomeBadge({json.dumps(ev)})));"],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)


@pytest.mark.parametrize("ev, text, cls, edge", [
    (STOPPED, "■ STOPPED · action 5/12", "bad", True),
    ({"status": "cancelled"}, "■ CANCELLED", "muted", False),
    ({"status": "success"}, "✓ SOLVED", "ok", False),
    ({"status": "done"}, "DONE", "plain", False),
    ({"status": "failed"}, "✗ NOT SOLVED", "bad", False),
    ({"status": "error", "message": "boom"}, "ERROR", "bad", False),
])
def test_the_badge_of_each_status(ev, text, cls, edge):
    b = badge(ev)
    assert (b["text"], b["cls"], b["edge"]) == (text, cls, edge)


def test_the_title_has_the_full_stop_reason():
    assert badge(STOPPED)["title"] == "Stopped after action 5 of 12: the gripper closed on nothing."
    assert badge({"status": "error", "message": "boom"})["title"] == "boom"


def test_the_badge_shows_in_the_camera_mode_until_the_next_run():
    js = _web("app.js")
    paint = _body(js, "function paintBadges()")
    assert 'const out = !plan && outcome && (camState === "ended" || camState === "run") ? outcome : null;' in paint
    assert '$(id).classList.toggle("stop-edge", Boolean(out && out.edge));' in paint and "badge.title" in paint
    assert 'if (state === "preview" || state === "live") outcome = null;' in _body(js, "function setCamState(")
    handler = _body(js, "function onEvent(ev, replay = false)")
    assert handler.count("outcome = null;") == 2 and "outcome = outcomeBadge(ev);" in handler
    assert ".cam.stop-edge .frame { border: 2px solid var(--bad); }" in _web("style.css")


def test_the_execution_entry_has_the_outcome_tag():
    end = _body(_web("app.js"), "function onExecEnd(ev)")
    assert "el(\"span\", tag.text, `out-tag out-${tag.cls}`)" in end
