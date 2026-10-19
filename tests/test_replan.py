"""Re-plan with feedback: the message, the plan lessons and the run fields of a later round."""
import hashlib
import logging

import pytest

from armmind.agent.loop import Agent
from armmind.agent.parse import LESSON_CHARS, ReplyError, parse_plan
from armmind.agent.prompts import PLAN_REPLY_SCHEMA, PLAN_SYSTEM, SYSTEM, Feedback, build_plan_text
from armmind.llm.fake import FakeBackend
from armmind.store.lessons import LessonStore
from armmind.types import Observation
from tests.fakes import Recorder, direct
from tests.test_agent_plan import PLAN, PlanEnv, good_plan

OBS = Observation(b"front", b"top", (0.3, 0.0, 0.4), 0.08, ())
LAST_OBJECTS = ({"name": "red cube", "shape": "box", "size": [0.04, 0.04, 0.04], "xyz": [0.5, -0.1, 0.02],
                 "yaw": 0.0},)
LAST_KEYFRAMES = ({"xyz": [0.5, -0.1, 0.02], "yaw": 0.0, "gripper": "closed", "holding": "red cube",
                   "note": "grasp it"},)


def feedback(**kw):
    base = {"objects": LAST_OBJECTS, "keyframes": LAST_KEYFRAMES, "status": "stopped", "check": True,
            "success": False, "stopped": {"after": 2, "of": 4, "reason": "the gripper closed on nothing"},
            "stopped_action": "close", "images": (b"end-front", b"end-top"), "parent": "20261018-100000",
            "round": 2, "end_step": 2}
    return Feedback(**{**base, **kw})


def test_system_is_unchanged():
    assert hashlib.sha256(SYSTEM.encode()).hexdigest() == \
        "9dfe9f6753b2b90d980e24c17d1d5f13f595fbcfb7787926f2b0b329c5739792"


def test_plan_system_speaks_of_the_replan_and_the_lessons():
    low = PLAN_SYSTEM.lower()
    assert "re-plan" in low and "end images" in low and "from the start" in low
    assert "In a re-plan from the start, the scene is back at its start." in PLAN_SYSTEM
    assert "from the start. The scene is back at its start." not in PLAN_SYSTEM
    assert '"lessons"' in PLAN_SYSTEM and "cause and effect" in low


def test_plan_schema_has_optional_lessons():
    assert PLAN_REPLY_SCHEMA["properties"]["lessons"] == {"type": "array", "items": {"type": "string"},
                                                          "maxItems": 3}
    assert "lessons" not in PLAN_REPLY_SCHEMA["required"]


def test_parse_plan_lessons():
    assert parse_plan(PLAN).lessons == ()
    got = parse_plan({**PLAN, "lessons": ["  A low grasp held.  ", "", "b", "c", "d"]}).lessons
    assert got == ("A low grasp held.", "b", "c")
    assert parse_plan({**PLAN, "lessons": ["x" * 500]}).lessons == ("x" * LESSON_CHARS,)
    for bad in ("one lesson", [1], {"a": 1}):
        with pytest.raises(ReplyError, match="lessons"):
            parse_plan({**PLAN, "lessons": bad})


def test_round_one_text_has_no_feedback():
    text = build_plan_text("Scene: s", "Pick it up.", OBS)
    assert "last plan" not in text.lower() and "Image 3" not in text


def test_replan_text_has_the_last_plan_the_result_and_the_end_images():
    text = build_plan_text("Objects: red cube.", "Pick it up.", OBS, None, ["Old lesson."], feedback())
    assert "Plan the whole task now, in one reply." in text and "Old lesson." in text
    assert "round 2" in text.lower()
    assert '"keyframes":[{"xyz":[0.5,-0.1,0.02]' in text and '"holding":"red cube"' in text  # compact JSON
    assert '"name":"red cube"' in text
    assert "stopped" in text and "action 2 of 4 (close)" in text and "the gripper closed on nothing" in text
    assert "not solved" in text
    assert "Image 1: front camera. Image 2: top camera." in text
    assert "Image 3: the front camera at the end of the last execution" in text and "Image 4: the top camera" in text


def test_replan_text_for_a_side_camera_and_no_check():
    side = "Image 1: side camera, looking along +y. Image 2: top camera."
    text = build_plan_text("s", "Open the drawer.", OBS, side, (),
                           feedback(check=False, status="done", stopped=None, stopped_action=None))
    assert "Image 3: the side camera at the end" in text and "All actions ran." in text
    assert "no automatic check" in text.lower() and "not solved" not in text


def test_replan_text_without_end_images():
    text = build_plan_text("s", "t", OBS, None, (), feedback(images=()))
    assert "Image 3" not in text and "no images of the end" in text.lower()


async def test_replan_sends_four_images_and_records_the_round(tmp_path):
    backend, rec = FakeBackend([good_plan()]), Recorder()
    agent = Agent(backend, store=LessonStore(tmp_path), run_id="20261018-100500")
    assert await agent.plan(PlanEnv(), direct, rec, seed=3, feedback=feedback()) == "planned"
    msg, _ = backend.sent[0]
    assert msg.images == (b"front", b"top", b"end-front", b"end-top")
    start = rec.events[0][1]
    assert start["parent"] == "20261018-100000" and start["round"] == 2 and start["end_step"] == 2
    assert "Your last plan" in msg.text


async def test_round_one_has_round_1_and_no_parent():
    rec = Recorder()
    await Agent(FakeBackend([good_plan()])).plan(PlanEnv(), direct, rec, seed=0)
    start = rec.events[0][1]
    assert start["round"] == 1 and start["parent"] is None and start["end_step"] is None


async def test_replan_lessons_are_saved_with_the_new_run_and_the_execution_status(tmp_path):
    store = LessonStore(tmp_path)
    body = {**PLAN, "lessons": ["Closing 2 cm too far left missed the cube."]}
    rec = Recorder()
    agent = Agent(FakeBackend([good_plan(body)]), store=store, run_id="20261018-100500")
    await agent.plan(PlanEnv(), direct, rec, seed=0, feedback=feedback(status="failed"))
    saved = store.load("cubes")
    assert [(s["text"], s["run"], s["outcome"]) for s in saved] == [
        ("Closing 2 cm too far left missed the cube.", "20261018-100500", "failed")]
    assert dict(rec.events)["plan"]["lessons"] == ["Closing 2 cm too far left missed the cube."]


async def test_round_one_lessons_are_not_saved(tmp_path, caplog):
    store = LessonStore(tmp_path)
    body = {**PLAN, "lessons": ["A guess before any result."]}
    with caplog.at_level(logging.INFO):
        await Agent(FakeBackend([good_plan(body)]), store=store, run_id="r1").plan(PlanEnv(), direct, Recorder())
    assert store.load("cubes") == [] and "not saved" in caplog.text


async def test_replan_lessons_with_the_switch_off_are_not_saved(tmp_path):
    store = LessonStore(tmp_path)
    body = {**PLAN, "lessons": ["Something new."]}
    agent = Agent(FakeBackend([good_plan(body)]), store=store, use_lessons=False, run_id="r2")
    await agent.plan(PlanEnv(), direct, Recorder(), feedback=feedback())
    assert store.load("cubes") == []


def test_replan_text_after_a_cancelled_execution():
    text = build_plan_text("s", "t", OBS, None, (),
                           feedback(status="cancelled", stopped=None, stopped_action=None, ran=3, total=10))
    assert "The person stopped the execution after action 3 of 10." in text
    assert "automatic check" not in text and "All actions ran." not in text and "Image 3" in text


def test_replan_text_after_an_error_has_no_check_line():
    text = build_plan_text("s", "t", OBS, None, (), feedback(status="error", stopped=None, stopped_action=None))
    assert "automatic check" not in text and "All actions ran." not in text


def test_replan_text_after_a_stop_before_the_first_action():
    text = build_plan_text("s", "t", OBS, None, (),
                           feedback(status="cancelled", stopped=None, stopped_action=None, ran=0, total=10))
    assert "The person stopped the execution before the first action of 10." in text
    assert "after action 0" not in text
