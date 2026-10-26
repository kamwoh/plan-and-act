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
                     for name in ("MAX_ROUND", "LAST_ROUND_TITLE", "PLAN_TITLE", "REPLAN_TITLE", "FROM_HERE_TITLE"))
    fn = _body(js, "function planRequest(s)") + "\n}\n"
    s = {"runId": None, "round": 1, "newest": "", "instruction": TEXT, "scene": "cubes", "seed": 7,
         "useLessons": True, "auto": False, "rounds": 3, "mode": "continue", "perRound": 2,
         "execError": False, "execPartial": False, **state}  # the last execution: planRequest ignores it
    out = subprocess.run(["node", "-e", f"{consts}{fn}console.log(JSON.stringify(planRequest({json.dumps(s)})));"],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def new_chain(auto=False, scene_text=None):
    return {"scene": "cubes", "scene_text": scene_text, "instruction": TEXT, "seed": 7, "use_lessons": True, "auto": auto, "rounds": 3,
            "mode": "continue", "max_keyframes": 2}


# A chain ends with clearPlan (Reset, a scene, a seed or a new plan), so the page then has no planRunId.
@pytest.mark.parametrize("case", ["no chain", "after Reset", "after a seed change"])
def test_without_a_chain_plan_starts_a_new_chain(case):
    r = plan_request(runId=None, newest=TEXT)
    assert (r["label"], r["url"], r["body"]) == ("▶ Plan", "/api/plan", new_chain())
    assert r["title"].startswith("Claude writes its belief of the scene")


def test_a_new_chain_sends_the_scene_description_of_the_box():
    r = plan_request(runId=None, sceneText="Objects: red cube.")
    assert r["body"] == new_chain(scene_text="Objects: red cube.")


def test_a_new_instruction_goes_on_from_the_newest_round():
    r = plan_request(runId="20261020-101010", round=2, newest="Stack the cubes")
    assert (r["label"], r["title"]) == ("▶ Plan from here", FROM_HERE)
    assert r["url"] == "/api/plan/20261020-101010/replan"
    assert r["body"] == {"mode": "continue", "max_keyframes": 2, "instruction": TEXT, "auto": False, "rounds": 3}


def test_a_new_instruction_always_sends_continue():
    r = plan_request(runId="r2", newest="Stack the cubes", mode="start")
    assert r["body"]["mode"] == "continue"  # after an error, the server starts it where the round started


@pytest.mark.parametrize("case", ["live chain", "after a History load"])
def test_the_same_instruction_re_plans_the_newest_round(case):
    r = plan_request(runId="r3", round=3, newest=TEXT, instruction=TEXT)
    assert r["label"] == "↻ Re-plan" and "Press Reset to start over." in r["title"]
    assert (r["url"], r["body"]) == ("/api/plan/r3/replan", {"mode": "continue", "max_keyframes": 2,
                                                              "auto": False, "rounds": 3})


# The Auto loop keeps the mode for each later round. After an error or a partial run, the server starts the
# next round where it must (see test_an_auto_continue_after_an_execution_error_keeps_continue_for_the_loop).
@pytest.mark.parametrize("state, mode, k", [
    ({"mode": "start"}, "start", 0),
    ({"mode": "continue"}, "continue", 2),
    ({"mode": "start", "execPartial": True}, "start", 0),
    ({"mode": "continue", "execError": True}, "continue", 2),
])
def test_the_re_plan_mode_is_the_continue_switch(state, mode, k):
    body = plan_request(runId="r1", newest=TEXT, auto=True, **state)["body"]
    assert (body["mode"], body["max_keyframes"]) == (mode, k)


def test_auto_starts_the_loop_with_the_rounds_field():
    assert plan_request(auto=True, rounds=5)["body"] == {**new_chain(True), "rounds": 5}
    for newest in ("Stack the cubes", TEXT):
        body = plan_request(runId="r1", newest=newest, auto=True, rounds=5)["body"]
        assert (body["auto"], body["rounds"]) == (True, 5)


def test_the_last_round_starts_a_new_chain():
    r = plan_request(runId="r30", round=30, newest=TEXT)
    assert (r["label"], r["url"]) == ("▶ Plan", "/api/plan")
    assert r["title"] == "Round 30 is the last round. Plan starts a new chain from the fresh scene"
    assert plan_request(runId=None, round=30)["title"].startswith("Claude writes its belief")


def test_the_plan_button_keeps_its_width_and_the_row_does_not_wrap():
    css = _web("style.css")
    rule = css[css.index("#plan_btn {"):]
    rule = rule[:rule.index("}")]
    assert "min-width:" in rule and "justify-content: center;" in rule
    assert ".controls .buttons { gap: 2px; flex-wrap: nowrap; }" in css


def test_the_page_has_no_prompt_editor():
    html, js, css = _web("index.html"), _web("app.js"), _web("style.css")
    for part in ("prompt_btn", "prompt_dialog", "prompt_tag"):
        assert part not in html and part not in js and part not in css
    assert "loadPrompt" not in js and "PROMPT_KEY" not in js and ".tag.edited" not in css


def test_the_plan_button_uses_the_request():
    js = _web("app.js")
    start = _body(js, "async function startPlan()")
    assert "const req = planRequest(planState());" in start
    assert 'if (req.url === "/api/plan") beginRun("planning");' in start  # a re-plan keeps the chat
    assert "postJson(req.url, req.body)" in start and "ownStart = true;" in start
    state = _body(js, "function planState()")
    for part in ("runId: planRunId", "round: planRound", "newest: planInstruction",
                 '$("instruction").value.trim()', "roundsValue(", "roundMode()",
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
