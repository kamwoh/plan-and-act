from armmind.agent.prompts import REPLY_SCHEMA, SYSTEM, StepSummary, build_user_text
from armmind.types import Observation

OBS = Observation(b"", b"", (0.3, 0.0, 0.4), 0.08, ())


def test_user_text_has_task_and_gripper():
    text = build_user_text("Touch the red cube.", 2, 30, OBS, "", (), [])
    assert "Touch the red cube." in text and "Step 2 of 30" in text
    assert "(0.300, 0.000, 0.400)" in text and "touching: nothing" in text.lower()
    assert "debug" not in text.lower()


def test_user_text_shows_positions_only_with_debug():
    obs = Observation(b"", b"", (0.3, 0.0, 0.4), 0.08, ("table",), {"red_cube": (0.5, -0.1, 0.02)})
    text = build_user_text("t", 1, 30, obs, "n1", ("grasp",), [StepSummary(1, "go", ("r1", "r2"))])
    assert "red_cube (0.500, -0.100, 0.020)" in text
    assert "step 1: go -> r1; r2" in text and "n1" in text and "grasp" in text


def test_system_and_schema_agree_on_limits():
    assert "0.25" in SYSTEM and "0.75" in SYSTEM and "8 actions" in SYSTEM
    assert REPLY_SCHEMA["properties"]["actions"]["maxItems"] == 8


def test_schema_and_system_know_done():
    assert REPLY_SCHEMA["properties"]["done"] == {"type": "boolean"}
    assert "done" in SYSTEM and "true" in SYSTEM
