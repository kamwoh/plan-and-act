"""v1.17: a re-plan with a new instruction goes on from where the arm is now, in the same chat."""
import pytest

from armmind.agent.prompts import build_plan_text, build_replan_text
from armmind.llm.fake import FakeBackend
from armmind.scenes import SCENES
from armmind.sim.state import START_STATE
from tests.fakes import judged
from tests.test_agent_plan import good_plan
from tests.test_auto_loop import folders
from tests.test_chain import SECRETS, app
from tests.test_continue_api import ENVS, StateEnv, plan_and_execute
from tests.test_plan_api import PICK
from tests.test_replan import OBS, feedback
from tests.test_replan_api import run_events
from tests.test_server import wait_idle

STACK = SCENES["cubes"].tasks["stack"].instruction
CONTINUES = ("New instruction from the person: Put the red cube on top of the blue cube and let go of it. "
             "This replaces the earlier task. The scene continues from where the last execution ended.")
SAME_START = ("New instruction from the person: Put the red cube on top of the blue cube and let go of it. "
              "This replaces the earlier task. The scene is the same as at the start of your last plan.")


@pytest.fixture(autouse=True)
def _clear():
    ENVS.clear()
    StateEnv.executions = 0


def assert_no_secrets(text):
    for secret in SECRETS:
        assert secret not in text, secret


def replan(c, run_id, **body):
    r = c.post(f"/api/plan/{run_id}/replan", json=body)
    assert r.status_code == 200, r.text
    wait_idle(c, 10)
    return r.json()["run_id"]


# ---- The prompts ----
def test_the_replan_text_of_a_new_instruction_starts_with_it():
    text = build_replan_text(OBS, None, feedback(mode="continue", instruction=STACK))
    assert text.startswith(CONTINUES + "\n")
    assert "Execution status: stopped." in text and "the earlier task is not solved" in text
    assert "Plan the new task now, in one reply." in text and "only what is left" not in text
    assert_no_secrets(text)


def test_the_replan_text_of_a_new_instruction_without_an_execution():
    f = feedback(mode="start", instruction=STACK, executed=False, status="not run", images=(), stopped=None,
                 stopped_action=None, end_step=None)
    text = build_replan_text(OBS, None, f, start_seen=True)
    assert text.startswith(SAME_START + "\n")
    assert "Execution status" not in text and "ran once" not in text and "end of the last execution" not in text
    summary = build_plan_text("Objects: red cube.", STACK, OBS, None, (), f)
    assert f"Task: {STACK}" in summary and SAME_START in summary and "Your last plan:" in summary


# ---- The server ----
def test_a_new_instruction_continues_from_the_end_state_in_the_same_chat(tmp_path):
    backend = FakeBackend([good_plan(), judged(), good_plan()])
    with app(tmp_path, backend) as c:
        first = plan_and_execute(c)
        second = replan(c, first, instruction=f"  {STACK}  ")
    folder = tmp_path / "runs" / second
    assert ENVS[-1].start == folder / START_STATE and ENVS[-1].start_marker == 1.0
    assert ENVS[-1].instruction == STACK and ENVS[-1].check_name == "stack"  # the task of the new instruction
    start = run_events(tmp_path, second)[0]
    assert (start["instruction"], start["round_mode"], start["new_instruction"]) == (STACK, "continue", True)
    assert (start["round"], start["parent"], start["check"]) == (2, first, "stack")
    assert (start["chain"], start["session_id"]) == (first, "fake-session-1")
    assert backend.opens[-1]["resume"] == "fake-session-1" and len(backend.chats) == 1
    msg = backend.sent[-1][0]
    assert msg.text.startswith(CONTINUES + "\n") and "Execution status: stopped." in msg.text
    assert msg.images == (b"front", b"top")
    for sent, _ in backend.sent:
        assert_no_secrets(sent.text)


def test_a_free_text_instruction_has_no_auto_check(tmp_path):
    with app(tmp_path, FakeBackend([good_plan(), judged(), good_plan()])) as c:
        second = replan(c, plan_and_execute(c), instruction="Wave at the camera.")
    start = run_events(tmp_path, second)[0]
    assert start["instruction"] == "Wave at the camera." and start["check"] is None and start["new_instruction"]


def test_a_new_instruction_without_an_execution_starts_from_the_start_of_that_round(tmp_path):
    backend = FakeBackend([good_plan(), good_plan()])
    with app(tmp_path, backend) as c:
        first = c.post("/api/plan", json={"instruction": PICK, "seed": 5}).json()["run_id"]
        wait_idle(c)
        second = replan(c, first, instruction=STACK)
    assert ENVS[-1].start is None  # round 1 started at the fresh scene: so does this round
    start = run_events(tmp_path, second)[0]
    assert (start["round_mode"], start["new_instruction"], start["end_step"]) == ("start", True, None)
    assert start["parent_check"] is None and start["session_id"] == "fake-session-1"
    msg = backend.sent[-1][0]
    assert msg.text.startswith(SAME_START + "\n") and "Execution status" not in msg.text
    assert msg.images == ()  # the chat has the start images of this scene


def test_a_new_instruction_without_an_execution_keeps_the_start_state_of_a_round_that_continues(tmp_path):
    with app(tmp_path, FakeBackend([good_plan(), judged(), good_plan(), good_plan()])) as c:
        first = plan_and_execute(c)
        second = replan(c, first)  # continues from the end of execution 1: no execution of its own
        third = replan(c, second, instruction=STACK)
    assert ENVS[-1].start == tmp_path / "runs" / third / START_STATE and ENVS[-1].start_marker == 1.0
    start = run_events(tmp_path, third)[0]
    assert (start["round"], start["round_mode"], start["new_instruction"]) == (3, "continue", True)


def test_mode_start_with_a_new_instruction_is_409(tmp_path):
    with app(tmp_path, FakeBackend([good_plan(), judged()])) as c:
        first = plan_and_execute(c)
        r = c.post(f"/api/plan/{first}/replan", json={"mode": "start", "instruction": STACK})
    assert r.status_code == 409 and "continue" in r.json()["detail"]


def test_the_same_instruction_is_a_plain_replan(tmp_path):
    with app(tmp_path, FakeBackend([good_plan(), judged(), good_plan()])) as c:
        second = replan(c, plan_and_execute(c), instruction=f" {PICK}\n")
    start = run_events(tmp_path, second)[0]
    assert start["instruction"] == PICK and start["new_instruction"] is False


def test_later_rounds_keep_the_newest_instruction(tmp_path):
    backend = FakeBackend([good_plan(), judged(), good_plan(), judged(), good_plan()])
    with app(tmp_path, backend) as c:
        second = replan(c, plan_and_execute(c), instruction=STACK)
        assert c.post(f"/api/plan/{second}/execute").status_code == 200
        wait_idle(c)
        third = replan(c, second)
    ex = next(e for e in run_events(tmp_path, second) if e["kind"] == "exec_start")
    assert (ex["instruction"], ex["check"]) == (STACK, "stack")  # the execution checks the new task
    start = run_events(tmp_path, third)[0]
    assert (start["instruction"], start["check"], start["new_instruction"]) == (STACK, "stack", False)
    assert not backend.sent[-1][0].text.startswith("New instruction")


def test_an_auto_replan_with_a_new_instruction_starts_the_loop(tmp_path):
    backend = FakeBackend([good_plan(), judged(), good_plan(), good_plan(), judged()])
    with app(tmp_path, backend) as c:
        first = plan_and_execute(c)
        second = replan(c, first, instruction=STACK, auto=True, rounds=2)
    runs = [p.name for p in folders(tmp_path)]
    assert runs[0] == first and runs[1] == second and len(runs) == 3
    starts = [run_events(tmp_path, r)[0] for r in runs[1:]]
    assert [s["instruction"] for s in starts] == [STACK, STACK]
    assert [s["round"] for s in starts] == [2, 3] and [s["auto_rounds"] for s in starts] == [2, 2]
    assert [s["new_instruction"] for s in starts] == [True, False]
    assert [s["chain"] for s in starts] == [first, first] and len(backend.chats) == 1
    assert starts[0]["round_mode"] == "continue" and ENVS[2].start_marker == 1.0
    execs = [next(e for e in run_events(tmp_path, r) if e["kind"] == "exec_start") for r in runs[1:]]
    assert [(e["round"], e["check"]) for e in execs] == [(2, "stack"), (3, "stack")]
    end = run_events(tmp_path, runs[2])[-1]
    assert end["kind"] == "loop_end" and end["rounds_run"] == 2 and end["rounds_max"] == 2
    assert [v["round"] for v in end["verdicts"]] == [2, 3]


def test_the_rounds_of_an_auto_replan_end_at_the_last_round(tmp_path):
    with app(tmp_path, FakeBackend([good_plan(), judged()])) as c:
        first = plan_and_execute(c)
        r = c.post(f"/api/plan/{first}/replan", json={"instruction": STACK, "auto": True, "rounds": 31})
    assert r.status_code == 422


def test_a_new_instruction_after_an_execution_error_starts_from_the_start_of_that_round(tmp_path, monkeypatch):
    def broken(self, keyframes, should_stop, on_action=None):
        raise RuntimeError("the simulation broke")

    monkeypatch.setattr(StateEnv, "execute_plan", broken)
    backend = FakeBackend([good_plan(), good_plan()])
    with app(tmp_path, backend) as c:
        first = c.post("/api/plan", json={"instruction": PICK, "seed": 5}).json()["run_id"]
        wait_idle(c)
        assert c.post(f"/api/plan/{first}/execute").status_code == 200
        wait_idle(c)
        end = [e for e in run_events(tmp_path, first) if e["kind"] == "exec_end"][-1]
        assert end["status"] == "error"
        second = replan(c, first, instruction=STACK, mode="continue")  # the mode that the page sends
    assert ENVS[-1].start is None  # no end state: the round starts where round 1 started
    start = run_events(tmp_path, second)[0]
    assert (start["round_mode"], start["new_instruction"], start["instruction"]) == ("start", True, STACK)
    msg = backend.sent[-1][0].text
    assert msg.startswith(SAME_START + "\n")
    assert "The last execution failed with an error in the simulation." in msg and "Execution status" not in msg
