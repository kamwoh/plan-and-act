from armmind.agent.prompts import REPLY_SCHEMA, SYSTEM, StepSummary, build_user_text
from armmind.types import Observation

OBS = Observation(b"", b"", (0.3, 0.0, 0.4), 0.08, ())
SCENE = "Objects: red cube, 4 cm."


def test_user_text_has_task_and_gripper():
    text = build_user_text(SCENE, "Touch the red cube.", 2, 30, OBS, "", (), [])
    assert "Touch the red cube." in text and "Step 2 of 30" in text
    assert "(0.300, 0.000, 0.400)" in text and "touching: nothing" in text.lower()
    assert "debug" not in text.lower()


def test_user_text_shows_positions_only_with_debug():
    obs = Observation(b"", b"", (0.3, 0.0, 0.4), 0.08, ("table",), {"red_cube": (0.5, -0.1, 0.02)})
    text = build_user_text(SCENE, "t", 1, 30, obs, "n1", ("grasp",), [StepSummary(1, "go", ("r1", "r2"))])
    assert "red_cube (0.500, -0.100, 0.020)" in text
    assert "step 1: go -> r1; r2" in text and "n1" in text and "grasp" in text


def test_system_and_schema_agree_on_limits():
    assert "0.25" in SYSTEM and "0.75" in SYSTEM and "8 actions" in SYSTEM
    assert REPLY_SCHEMA["properties"]["actions"]["maxItems"] == 8


def test_schema_and_system_know_done():
    assert REPLY_SCHEMA["properties"]["done"] == {"type": "boolean"}
    assert "done" in SYSTEM and "true" in SYSTEM


def test_user_text_starts_with_the_scene():
    text = build_user_text(SCENE, "t", 1, 30, OBS, "", (), [])
    assert text.splitlines()[0] == "Scene: Objects: red cube, 4 cm."


def test_system_speaks_of_objects_not_cubes():
    assert "cube" not in SYSTEM.lower()
    assert "To grasp an object: open, move 10 cm above it, move down to its centre height, close, then move up." in SYSTEM
    assert "step message lists them with their sizes" in SYSTEM


def test_schema_and_system_know_yaw():
    assert REPLY_SCHEMA["properties"]["actions"]["items"]["properties"]["yaw"] == {"type": "number"}
    assert '"yaw"' in SYSTEM and "straight line" in SYSTEM


def test_user_text_shows_yaw():
    obs = Observation(b"", b"", (0.3, 0.0, 0.4), 0.08, (), gripper_yaw=30.0)
    text = build_user_text(SCENE, "t", 1, 30, obs, "", (), [])
    assert "Gripper at (0.300, 0.000, 0.400), yaw 30°, width 0.080 m." in text


def test_user_text_lists_events_and_debug_joints():
    obs = Observation(b"", b"", (0.3, 0.0, 0.4), 0.08, (), events=("red button pressed", "green button pressed"),
                      joints=("drawer slide 0.120 m (0 = closed)", "red button depth 0.004 m"))
    text = build_user_text(SCENE, "t", 1, 30, obs, "", (), [])
    assert "Events: red button pressed; green button pressed." in text
    assert "Moving parts (debug): drawer slide 0.120 m (0 = closed); red button depth 0.004 m." in text
    assert "Events" not in build_user_text(SCENE, "t", 1, 30, OBS, "", (), [])


def test_image_text_replaces_the_last_line():
    assert build_user_text(SCENE, "t", 1, 30, OBS, "", (), []).splitlines()[-1] == \
        "Image 1: front camera. Image 2: top camera."
    text = build_user_text(SCENE, "t", 1, 30, OBS, "", (), [], image_text="Image 1: side. Image 2: top.")
    assert text.splitlines()[-1] == "Image 1: side. Image 2: top."
