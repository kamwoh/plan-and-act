"""v1.14 page: one chat for each chain of rounds, a divider for each round, and History loads the whole chain."""
import re

from tests.test_page_plan import _body, _web


def handler():
    return _body(_web("app.js"), "function onEvent(ev, replay = false)")


def test_a_later_round_of_the_plan_on_the_page_keeps_the_chat_and_the_totals():
    start = handler()[:handler().index('ev.kind === "observation"')]
    assert 'const chained = ev.mode === "plan" && Boolean(ev.parent) && ev.parent === planRunId;' in start
    assert start.index("const chained") < start.index("clearPlan()")  # read the plan id before it goes
    assert re.search(r"if \(!chained\) \{\s*resetChat\(\);\s*resetTotals\(\);\s*\}", start)
    assert "if (chained) startRound(planMode, ev.new_instruction ? ev.instruction : null);" in start


def test_each_round_has_its_own_step_keys():
    js = _web("app.js")
    assert "return `${chatRound}:${step}`;" in _body(js, "function stepKey(step)")
    assert "stepLink(item, ev.step)" not in js and "stepImages[ev.step]" not in js and "stepEvents[ev.step]" not in js
    assert "addEntry(stepLink(item, stepKey(ev.step)));" in _body(js, "function onPlan(ev)")


def test_older_rounds_fold_to_one_line():
    js, css = _web("app.js"), _web("style.css")
    start = _body(js, "function startRound(mode, instruction = null)")
    assert "foldRound(Number(r), true)" in start and "addEntry(roundDivider(chatRound, mode, instruction))" in start
    div = _body(js, "function roundDivider(round, mode, instruction = null)")
    assert '`── Round ${round}${instruction ? " · new instruction" : modeText(mode)} ──`' in div
    assert ".entry[hidden] { display: none; }" in css  # .entry is a flex box: hidden needs its own rule


def test_the_buttons_of_an_older_round_stay_off():
    js = _web("app.js")
    assert "b.dataset.run !== planRunId" in _body(js, "function setRunning(value)")
    assert 'exec.dataset.run = planRunId || "";' in _body(js, "function planCheckEntry")
    assert 'b.dataset.run = planRunId || "";' in _body(js, "function replanButton()")


def test_a_lost_chat_shows_a_note():
    lost = handler()[handler().index('ev.kind === "chat_lost"'):handler().index('ev.kind === "loop_end"')]
    assert "ev.message" in lost and "addEntry(item)" in lost


def test_history_loads_the_whole_chain_and_the_video_of_the_opened_round():
    body = _body(_web("app.js"), "async function loadRun")
    assert "/chain`" in body and "replayShowsPlan = i === runs.length - 1;" in body
    assert "(!replay || replayShowsPlan)" in handler()


def test_the_chain_endpoint(tmp_path):
    from fastapi.testclient import TestClient

    from plan_and_act.llm.fake import FakeBackend
    from plan_and_act.server.app import create_app
    from tests.fakes import judged
    from tests.test_agent_plan import good_plan
    from tests.test_continue_api import factory
    from tests.test_plan_api import PICK
    from tests.test_server import wait_idle
    app = create_app(FakeBackend([good_plan(), judged(), good_plan()]), factory, tmp_path / "runs",
                     lessons_dir=tmp_path / "lessons")
    with TestClient(app) as c:
        first = c.post("/api/plan", json={"instruction": PICK, "seed": 5}).json()["run_id"]
        wait_idle(c)
        assert c.post(f"/api/plan/{first}/execute").status_code == 200
        wait_idle(c)
        second = c.post(f"/api/plan/{first}/replan", json={"mode": "start"}).json()["run_id"]
        wait_idle(c)
        assert c.get(f"/api/runs/{second}/chain").json() == [first, second]
        assert c.get(f"/api/runs/{first}/chain").json() == [first]
        assert c.get("/api/runs/20261018-999999/chain").status_code == 404


def test_the_plan_started_entry_says_the_same_chat_only_when_the_round_continues_it():
    js = _web("app.js")
    start = _body(js, "function onRunStart(ev, chained = false)")
    assert "const same = chained && Boolean(ev.session_id);" in start  # a branch has no session to resume
    assert "same ? SAME_CHAT : chained ? NEW_CHAT_TEXT" in start
    lost = handler()[handler().index('ev.kind === "chat_lost"'):handler().index('ev.kind === "loop_end"')]
    assert "roundChatText.textContent = NEW_CHAT_TEXT" in lost  # a full or lost chat also starts a new one


async def test_a_branch_round_starts_without_a_session_to_resume():
    from plan_and_act.agent.loop import BRANCH_CHAT, Agent, Chain
    from plan_and_act.llm.fake import FakeBackend
    from tests.fakes import Recorder, direct
    from tests.test_agent_plan import PlanEnv, good_plan
    from tests.test_replan import feedback
    rec = Recorder()
    await Agent(FakeBackend([good_plan()]), run_id="r3", chain=Chain("r1", note=BRANCH_CHAT)).plan(
        PlanEnv(), direct, rec, feedback=feedback())
    assert rec.events[0][1]["session_id"] is None


def test_a_plan_event_ends_the_chat_text_of_its_round():
    handler = _body(_web("app.js"), "function onEvent(ev, replay = false)")
    plan = handler[handler.index('ev.kind === "plan"'):]
    assert plan.index("roundChatText = null") < plan.index("onPlan(ev)")
