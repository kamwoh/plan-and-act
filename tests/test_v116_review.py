"""v1.16 review follow-ups: a stop names its keyframe, a partial run says so in each message and on the page."""
import pytest

from plan_and_act.agent.prompts import build_chain_verdict_text, build_replan_text, build_verdict_text
from plan_and_act.plan.formats import Keyframe
from plan_and_act.scenes import SCENES
from tests.test_page_outcome import badge
from tests.test_page_plan import _body, _web
from tests.test_replan import OBS, feedback

PARTIAL = "Only the first 2 of your 5 keyframes ran, by design. The arm is now at the end of keyframe 2."


# ---- 1. A stop names its keyframe ----
@pytest.mark.assets
def test_the_stop_of_an_execution_names_its_keyframe():
    from plan_and_act.env import MujocoEnv
    sc = SCENES["cubes"]
    kfs = (Keyframe((0.35, 0.0, 0.15), 0.0, "open", None, "above"),
           Keyframe((0.35, 0.0, 0.02), 0.0, "closed", "red cube", "grasp"),
           Keyframe((0.35, 0.0, 0.2), 0.0, "closed", "red cube", "lift"))
    env = MujocoEnv(sc, sc.tasks["pick"], 0, sc.tasks["pick"].instruction)
    try:
        out = env.execute_plan(kfs, lambda: False)
    finally:
        env.close()
    assert out["stopped"] == {"after": 3, "of": 4, "keyframe": 2, "reason": "the gripper closed on nothing"}


def test_the_replan_stop_line_names_the_keyframe():
    stopped = {"after": 2, "of": 4, "keyframe": 1, "reason": "the gripper closed on nothing"}
    text = build_replan_text(OBS, None, feedback(stopped=stopped))
    assert "The execution stopped in keyframe 1 of 1, at action 2 of 4 (close): the gripper closed on nothing." in text
    old = build_replan_text(OBS, None, feedback())  # a run from before v1.16 has no keyframe
    assert "The execution stopped at action 2 of 4 (close)" in old


def test_the_badge_counts_keyframes():
    new = {"status": "stopped", "keyframes": 7, "stopped": {"after": 9, "of": 10, "keyframe": 6, "reason": "blocked"}}
    b = badge(new)
    assert b["text"] == "■ STOPPED · keyframe 6/7"
    assert b["title"] == "Stopped in keyframe 6 of 7 (action 9 of 10): blocked."
    old = {"status": "stopped", "stopped": {"after": 9, "of": 10, "reason": "blocked"},
           "frames": [{"keyframe": 5, "of": 7, "at": "end"}, {"keyframe": 6, "of": 7, "at": "stop"}]}
    assert badge(old)["text"] == "■ STOPPED · keyframe 6/7"  # the stop frame gives the keyframe
    oldest = {"status": "stopped", "stopped": {"after": 9, "of": 10, "reason": "blocked"}}
    assert badge(oldest)["text"] == "■ STOPPED · action 9/10"


def test_the_chat_line_and_the_status_count_keyframes():
    js = _web("app.js")
    assert "stopText(ev)" in _body(js, "function onExecEnd(ev)")
    assert "stopKeyframe(ev)" in _body(js, "function execText(ev)")


# ---- 2. The Prediction entry of a partial run ----
def test_the_prediction_of_a_partial_run_says_it_ran_by_design():
    js = _web("app.js")
    body = _body(js, "function predictionEntry(p, ev = {})")
    assert "`Run per round: keyframes 1 to ${ev.ran_keyframes} of ${ev.keyframes} ran by design. This compares them.`" \
        in body
    assert "predictionEntry(ev.prediction, ev)" in js


# ---- 3. The verdict after a partial run ----
def test_the_verdict_texts_say_that_a_partial_run_ran_by_design():
    assert PARTIAL in build_chain_verdict_text(None, (), (2, 5))
    assert PARTIAL in build_verdict_text("s", "t", (), (), None, (), (2, 5))
    assert "Only the first" not in build_chain_verdict_text(None)
    assert PARTIAL in build_replan_text(OBS, None, feedback(status="partial", stopped=None, ran_keyframes=(2, 5)))


# ---- 4. The pill of a re-plan after a partial run ----
def test_a_replan_after_a_partial_run_says_continue():
    js = _web("app.js")
    assert "const partial = execPartial;" in _body(js, "function replanButton()")
    body = _body(js, "async function replan(")
    assert 'const mode = fromStart ? "start" : partial ? "continue" : roundMode();' in body


# ---- 5. stepSent follows each attempt that sends images ----
def test_step_sent_follows_each_attempt():
    ask = _body(_web("app.js"), "function onAsk(ev)")
    line = 'if (typeof ev.images === "number" && ev.images > 0) stepSent[stepKey(ev.step)] = ev.images;'
    assert line in ask and ask.index(line) < ask.index("if (ev.attempt === 1)")
