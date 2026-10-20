"""v1.18 page: Plan goes on from the chain on the page, Reset starts over."""
import json
import subprocess

import pytest

from tests.test_page_plan import _body, _web

FROM_HERE = ("Claude gets the new instruction in the same chat, and the arm goes on from where it is now. "
             "Press Reset to start over")
TEXT = "Put the red cube in the bowl"


def plan_request(**state):
    """Run planRequest of the page in node with this state of the page."""
    js = _web("app.js")
    consts = "".join(_body(js, f"const {name} =", ";\n") + ";\n"
                     for name in ("MAX_ROUND", "PLAN_TITLE", "REPLAN_TITLE", "FROM_HERE_TITLE"))
    fn = _body(js, "function planRequest(s)") + "\n}\n"
    s = {"runId": None, "round": 1, "newest": "", "instruction": TEXT, "scene": "cubes", "seed": 7,
         "useLessons": True, "auto": False, "rounds": 3, "mode": "continue", "perRound": 2,
         "execError": False, "execPartial": False, **state}
    out = subprocess.run(["node", "-e", f"{consts}{fn}console.log(JSON.stringify(planRequest({json.dumps(s)})));"],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def new_chain(auto=False):
    return {"scene": "cubes", "instruction": TEXT, "seed": 7, "use_lessons": True, "auto": auto, "rounds": 3,
            "mode": "continue", "max_keyframes": 2}


# A chain ends with clearPlan (Reset, a scene, a seed or a new plan), so the page then has no planRunId.
@pytest.mark.parametrize("case", ["no chain", "after Reset", "after a seed change"])
def test_without_a_chain_plan_starts_a_new_chain(case):
    r = plan_request(runId=None, newest=TEXT)
    assert (r["label"], r["url"], r["body"]) == ("▶ Plan", "/api/plan", new_chain())
    assert r["title"].startswith("Claude writes its belief of the scene")


def test_a_new_instruction_goes_on_from_the_newest_round():
    r = plan_request(runId="20261020-101010", round=2, newest="Stack the cubes")
    assert (r["label"], r["title"]) == ("▶ Plan from here", FROM_HERE)
    assert r["url"] == "/api/plan/20261020-101010/replan"
    assert r["body"] == {"mode": "continue", "max_keyframes": 2, "instruction": TEXT, "auto": False, "rounds": 3}


def test_a_new_instruction_after_an_execution_error_still_sends_continue():
    r = plan_request(runId="r2", newest="Stack the cubes", execError=True, mode="start")
    assert r["body"]["mode"] == "continue"  # the server starts it where the round started


@pytest.mark.parametrize("case", ["live chain", "after a History load"])
def test_the_same_instruction_re_plans_the_newest_round(case):
    r = plan_request(runId="r3", round=3, newest=TEXT, instruction=TEXT)
    assert r["label"] == "↻ Re-plan" and "Press Reset to start over." in r["title"]
    assert (r["url"], r["body"]) == ("/api/plan/r3/replan", {"mode": "continue", "max_keyframes": 2,
                                                              "auto": False, "rounds": 3})


@pytest.mark.parametrize("state, mode, k", [
    ({"mode": "start"}, "start", 0),
    ({"mode": "start", "execPartial": True}, "continue", 2),  # a partial run always continues
    ({"mode": "continue", "execError": True}, "start", 0),  # no end state after an error
])
def test_the_re_plan_mode_follows_the_continue_switch_and_the_last_execution(state, mode, k):
    body = plan_request(runId="r1", newest=TEXT, **state)["body"]
    assert (body["mode"], body["max_keyframes"]) == (mode, k)


def test_auto_starts_the_loop_with_the_rounds_field():
    assert plan_request(auto=True, rounds=5)["body"] == {**new_chain(True), "rounds": 5}
    for newest in ("Stack the cubes", TEXT):
        body = plan_request(runId="r1", newest=newest, auto=True, rounds=5)["body"]
        assert (body["auto"], body["rounds"]) == (True, 5)


def test_the_last_round_starts_a_new_chain():
    r = plan_request(runId="r30", round=30, newest=TEXT)
    assert (r["label"], r["url"]) == ("▶ Plan", "/api/plan")


def test_the_plan_button_uses_the_request():
    js = _web("app.js")
    start = _body(js, "async function startPlan()")
    assert "const req = planRequest(planState());" in start
    assert 'if (req.url === "/api/plan") beginRun("planning");' in start  # a re-plan keeps the chat
    assert "postJson(req.url, req.body)" in start and "ownStart = true;" in start
    state = _body(js, "function planState()")
    for part in ("runId: planRunId", "round: planRound", "newest: planInstruction",
                 '$("instruction").value.trim()', "execError", "execPartial", "roundsValue(", "roundMode()",
                 "perRound()"):
        assert part in state
    label = _body(js, "function labelPlanButton()")
    assert "planRequest(planState())" in label and '$("plan_btn")' in label


def test_the_label_follows_the_box_and_the_chain():
    js = _web("app.js")
    assert "labelPlanButton();" in _body(js, "function showMode")  # each key in the box
    assert _body(js, "function clearPlan").rstrip().endswith("labelPlanButton();")
    handler = _body(js, "function onEvent(ev, replay = false)")
    assert handler.count("labelPlanButton();") >= 3  # run_start, exec_start, exec_end


def test_reset_a_scene_and_a_seed_end_the_chain():
    js = _web("app.js")
    assert "clearPlan();" in _body(js, "async function loadPreview()")  # Reset, a scene chip and Shuffle
    assert "loadPreview();" in _body(js, '$("reset").addEventListener("click"', "\n});\n")
    seed = _body(js, '$("seed").addEventListener("input"', "\n});\n")
    assert "clearPlan();" in seed  # at once: Plan in the next 300 ms must not go on with the old seed
    assert "Reset: start over from the start layout of this scene and seed. The next Plan starts a new chat" \
        in _web("index.html")


def test_a_history_load_opens_the_chain_at_the_opened_round():
    handler = _body(_web("app.js"), "function onEvent(ev, replay = false)")
    start = handler[:handler.index('ev.kind === "observation"')]
    assert 'planRunId = ev.mode === "plan" ? ev.run_id : null;' in start  # also in a replay: the last round wins
    assert "if (!own) showInstruction();" in start


def test_the_chat_replan_ignores_the_box():
    js = _web("app.js")
    body = _body(js, "async function replan(fromStart = false, partial = false)")
    assert "newInstruction" not in js and '$("instruction")' not in body and "instruction:" not in body
    button = _body(js, "function replanButton()")
    assert 'el("button", "↻ Re-plan", "btn exec-btn replan-btn")' in button and "b.title = REPLAN_TITLE;" in button
    assert "replanLabel" not in js and "showReplanLabels" not in js


def test_the_plan_check_has_no_plan_from_here_button():
    js = _web("app.js")
    assert "replanButton(" not in _body(js, "function planCheckEntry")
    assert "onlyNew" not in js and "planExecuted" not in js
