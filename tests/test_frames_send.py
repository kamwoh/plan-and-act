"""v1.15: the re-plan and the verdict messages send the frames of the execution with the poses Claude imagined."""
import pytest
from fastapi.testclient import TestClient

from plan_and_act.agent.loop import CHAT_TOO_LONG, MAX_CHAT_IMAGES, Agent, Chain
from plan_and_act.agent.prompts import (
    Frame,
    build_chain_verdict_text,
    build_plan_text,
    build_replan_text,
    build_verdict_text,
    frame_label,
)
from plan_and_act.llm.fake import FakeBackend
from plan_and_act.plan.formats import build_track
from plan_and_act.plan.track import moved_after
from plan_and_act.server.app import create_app
from tests.fakes import Recorder, direct, judged
from tests.test_agent_plan import PlanEnv, good_plan
from tests.test_chain import assert_no_secrets
from tests.test_continue_api import StateEnv
from tests.test_plan_api import PICK, fake_frames
from tests.test_replan import OBS, feedback
from tests.test_replan_api import TRUE_XYZ, run_events
from tests.test_server import make_env, wait_idle

RED = (("red cube", (0.5, -0.1, 0.02), 0.0),)
FRAMES = (Frame(2, 5, "down to the cube", "end", (), b"frame2"), Frame(3, 5, "grasp", "end", RED, b"frame3"),
          Frame(4, 5, "lift", "stop", (("red cube", (0.5, -0.1, 0.15), 0.0), ("green can", (0.45, 0.2, 0.05), None)),
                b"frame4"))


# ---- The labels and the text ----
def test_a_frame_label_has_the_keyframe_the_note_and_the_imagined_poses():
    assert frame_label(FRAMES[0], "front") == ('Front camera after keyframe 2 of 5. Note: "down to the cube". '
                                               "No object moves in your plan up to here.")
    assert frame_label(FRAMES[1], "front") == ('Front camera after keyframe 3 of 5. Note: "grasp". '
                                               "You imagined red cube at (0.500, -0.100, 0.020), yaw 0°.")
    assert frame_label(FRAMES[2], "side") == (
        'Side camera at the stop in keyframe 4 of 5. Note: "lift". At the end of keyframe 4, you imagined red cube at '
        "(0.500, -0.100, 0.150), yaw 0°. At the end of keyframe 4, you imagined green can at (0.450, 0.200, 0.050).")
    empty_stop = Frame(2, 5, "down", "stop", (), b"f")
    assert frame_label(empty_stop, "front").endswith("At the end of keyframe 2, no object moves in your plan.")
    without_track = Frame(1, 2, "up", "end", None, b"f")
    assert frame_label(without_track, "front") == 'Front camera after keyframe 1 of 2. Note: "up".'


COMPARE = "Compare each frame with the poses that you imagined, and name the first keyframe where they differ."


def _lines(text):
    return text.split("\n")


def test_the_replan_text_numbers_the_frames_before_the_end_images():
    seen = _lines(build_replan_text(OBS, None, feedback(frames=FRAMES), start_seen=True))
    i = seen.index(COMPARE)
    assert seen[i - 4].startswith("Images 1 to 3 show the front camera during the execution, in order.")
    assert seen[i - 3].startswith("- Image 1: Front camera after keyframe 2 of 5.")
    assert seen[i - 1].startswith("- Image 3: Front camera at the stop in keyframe 4 of 5.")
    assert "Image 4: the front camera at the end of the last execution. Image 5: the top camera at the end." in seen
    full = build_replan_text(OBS, None, feedback(frames=FRAMES))
    assert "Images 3 to 5 show the front camera" in full and "- Image 5: Front camera at the stop" in full
    assert "Images 6 and 7 show the end of the last execution." in full and "Image 6: the front camera" in full
    summary = build_plan_text("Scene: s", PICK, OBS, feedback=feedback(frames=FRAMES))
    assert "Images 3 to 5 show" in summary and "Images 6 and 7 show the end" in summary and COMPARE in summary
    goes_on = build_replan_text(OBS, None, feedback(frames=FRAMES, mode="continue", images=()))
    assert "Images 3 to 5 show" in goes_on and "Images 1 and 2 show the scene now" in goes_on


def test_without_frames_the_text_is_as_before():
    assert COMPARE not in build_replan_text(OBS, None, feedback())
    assert "Image 3: the front camera" in build_replan_text(OBS, None, feedback())
    assert build_chain_verdict_text(None) == ("Your last plan ran once in the physics. Give your verdict now, not a "
                                              "plan.\nThe images show the scene after the run. Image 1: front camera. "
                                              "Image 2: top camera.\nIs the task complete?")


def test_the_verdict_texts_list_the_frames_then_the_end_images():
    for text in (build_chain_verdict_text(None, FRAMES), build_verdict_text("s", PICK, (), (), None, FRAMES)):
        assert "Images 1 to 3 show the front camera during the execution" in text and COMPARE in text
        assert "Images 4 and 5 show the scene after the run: the front camera, then the top camera." in text
        assert "- Image 2: Front camera after keyframe 3 of 5." in text
    side = build_chain_verdict_text("Image 1: side camera, looking along +y. Image 2: top camera.", FRAMES[:1])
    assert "Image 1 shows the side camera" in side and "Images 2 and 3 show the scene after the run: the side" in side


def test_the_labels_never_have_true_values():
    obs = PlanEnv().observe()
    for text in (build_replan_text(obs, None, feedback(frames=FRAMES)), build_chain_verdict_text(None, FRAMES)):
        assert_no_secrets(text)


# ---- The agent ----
async def test_the_replan_message_sends_the_frames_before_the_end_images():
    backend = FakeBackend([good_plan()])
    backend.store.chats["s1"] = []
    chain = Chain("r1", "s1", images=2, start=True)
    rec = Recorder()
    await Agent(backend, run_id="r2", chain=chain).plan(PlanEnv(), direct, rec, feedback=feedback(frames=FRAMES))
    assert backend.sent[-1][0].images == (b"frame2", b"frame3", b"frame4", b"end-front", b"end-top")
    assert dict(rec.events)["plan"]["chat_images"] == 2 + 5
    assert dict(rec.events)["ask"]["images"] == 5  # the page then leaves out the start images


async def test_the_verdict_sends_the_frames_before_the_end_images():
    backend = FakeBackend([judged()])
    backend.store.chats["s1"] = []
    _, detail = await Agent(backend, run_id="r1", chain=Chain("r1", "s1", images=2, start=True)).verdict(
        "s", PICK, (), (), (b"f", b"t"), frames=FRAMES)
    assert backend.sent[-1][0].images == (b"frame2", b"frame3", b"frame4", b"f", b"t")
    assert detail["chat_images"] == 7


@pytest.mark.parametrize("frames, new_chat", [((), False), (FRAMES, True)])
async def test_the_image_budget_counts_the_frames(frames, new_chat):
    backend, rec = FakeBackend([good_plan()]), Recorder()
    backend.store.chats["s1"] = []
    chain = Chain("r1", "s1", images=MAX_CHAT_IMAGES - 4, start=True)  # 2 end images fit, 3 more frames do not
    await Agent(backend, run_id="r2", chain=chain).plan(PlanEnv(), direct, rec, feedback=feedback(frames=frames))
    lost = [d for k, d in rec.events if k == "chat_lost"]
    assert bool(lost) is new_chat
    if new_chat:
        assert lost[0]["message"] == CHAT_TOO_LONG and backend.opens[-1]["resume"] is None
        assert backend.sent[-1][0].images == (b"front", b"top", b"frame2", b"frame3", b"frame4", b"end-front",
                                              b"end-top")
        assert "Your last plan:" in backend.sent[-1][0].text and COMPARE in backend.sent[-1][0].text


# ---- The predicted poses ----
CARRY = {"objects": [{"name": "red cube", "shape": "box", "size": [0.04, 0.04, 0.04], "xyz": [0.5, -0.1, 0.02],
                      "yaw": 0},
                     {"name": "green can", "shape": "cylinder", "size": [0.06, 0.06, 0.1], "xyz": [0.4, 0.2, 0.05],
                      "yaw": 0}],
         "keyframes": [
             {"xyz": [0.5, -0.1, 0.15], "yaw": 0, "gripper": "open", "holding": "nothing", "note": "above the cube"},
             {"xyz": [0.5, -0.1, 0.02], "yaw": 0, "gripper": "open", "holding": "nothing", "note": "down to the cube"},
             {"xyz": [0.5, -0.1, 0.02], "yaw": 0, "gripper": "closed", "holding": "red cube", "note": "grasp"},
             {"xyz": [0.5, -0.1, 0.15], "yaw": 0, "gripper": "closed", "holding": "red cube", "note": "lift"},
             {"xyz": [0.5, 0.1, 0.15], "yaw": 30, "gripper": "closed", "holding": "red cube", "note": "carry right",
              "moves": [{"object": "green can", "xyz": [0.45, 0.2, 0.05], "yaw": 0}]}],
         "summary": "Pick up the red cube and carry it to the right."}


def _track(belief, keyframes):
    return build_track({"xyz": [0.3, 0.0, 0.4], "yaw": 0.0, "state": "open"}, keyframes,
                       move_s=lambda a, b, ya, yb: 1.0, settle_s=0.1, change_s=0.2, gap=lambda k: 0.08)


def test_moved_after_gives_the_carried_and_moved_entities_of_each_keyframe():
    from plan_and_act.agent.parse import parse_plan
    reply = parse_plan(CARRY)
    believed = [o.to_dict() for o in reply.objects]
    moved = moved_after(_track(reply.objects, reply.keyframes), believed)
    assert [list(m) for m in moved] == [[], [], ["red cube"], ["red cube"], ["red cube", "green can"]]
    assert moved[3]["red cube"]["xyz"] == pytest.approx([0.5, -0.1, 0.15])
    assert moved[4]["red cube"]["yaw"] == pytest.approx(30.0) and moved[4]["green can"]["xyz"] == [0.45, 0.2, 0.05]


# ---- The server ----
class FrameEnv(StateEnv):
    """Its view has Claude's belief and a track. Its execution gives one frame for each keyframe."""
    stop_in: int | None = None

    def plan_view(self, belief, keyframes, marks=None):
        return {"believed": [b.to_dict() for b in belief], "track": _track(belief, keyframes), "marks": marks,
                "true": [{"name": "red_cube", "xyz": TRUE_XYZ, "yaw": 33.3}]}

    def execute_plan(self, keyframes, should_stop, on_action=None):
        out = super().execute_plan(keyframes, should_stop, on_action)
        stopped = FrameEnv.stop_in is not None
        return {**out, "status": "stopped" if stopped else "failed", "stopped": out["stopped"] if stopped else None,
                "frames": fake_frames(keyframes, FrameEnv.stop_in)}


def frame_factory(scene, task, instruction, seed, show_positions, on_frame, start=None):
    env = make_env(scene, task, instruction, cls=FrameEnv)
    env.seed, env.start = seed, start
    return env


def _run(tmp_path, mode, stop_in=None):
    FrameEnv.stop_in = stop_in
    backend = FakeBackend([good_plan(CARRY), judged(), good_plan(CARRY)])
    with TestClient(create_app(backend, frame_factory, tmp_path / "runs", lessons_dir=tmp_path / "lessons")) as c:
        first = c.post("/api/plan", json={"instruction": PICK, "seed": 5}).json()["run_id"]
        wait_idle(c)
        assert c.post(f"/api/plan/{first}/execute").status_code == 200
        wait_idle(c)
        assert c.post(f"/api/plan/{first}/replan", json={"mode": mode}).status_code == 200
        wait_idle(c)
    FrameEnv.stop_in = None
    verdict, replan = backend.sent[1][0], backend.sent[2][0]
    for msg in (verdict, replan):
        assert_no_secrets(msg.text)
    return run_events(tmp_path, first), verdict, replan


def test_the_verdict_and_the_replan_get_the_frames_with_the_imagined_poses(tmp_path):
    events, verdict, replan = _run(tmp_path, "start")
    end = next(e for e in events if e["kind"] == "exec_end")
    assert [f["sent"] for f in end["frames"]] == [False, True, True, True, False]
    assert verdict.images == (b"frame2", b"frame3", b"frame4", b"front", b"top")
    assert "- Image 1: Front camera after keyframe 2 of 5. Note: \"down to the cube\". No object moves" in verdict.text
    assert ('- Image 3: Front camera after keyframe 4 of 5. Note: "lift". '
            "You imagined red cube at (0.500, -0.100, 0.150), yaw 0°.") in verdict.text
    assert "Images 4 and 5 show the scene after the run" in verdict.text
    assert replan.images == (b"frame2", b"frame3", b"frame4", b"front", b"top")  # the chat has the start images
    assert "Images 1 to 3 show the front camera" in replan.text and COMPARE in replan.text
    assert "Image 4: the front camera at the end of the last execution." in replan.text


def test_a_replan_that_continues_gets_the_frames_after_the_images_now(tmp_path):
    _, _, replan = _run(tmp_path, "continue")
    assert replan.images == (b"front", b"top", b"frame2", b"frame3", b"frame4")
    assert "Images 3 to 5 show the front camera" in replan.text


def test_a_stopped_run_sends_the_stop_frame(tmp_path):
    _, verdict, replan = _run(tmp_path, "start", stop_in=5)
    assert verdict.images == (b"frame2", b"frame3", b"frame4", b"frame5", b"front", b"top")
    assert ('- Image 4: Front camera at the stop in keyframe 5 of 5. Note: "carry right". At the end of keyframe 5, '
            "you imagined red cube at (0.500, 0.100, 0.150), yaw 30°. At the end of keyframe 5, you imagined green can "
            "at (0.450, 0.200, 0.050).") in verdict.text
    assert "stop in keyframe 5 of 5. Note: \"carry right\". You imagined" not in verdict.text  # the poses of the end
    assert "Image 5: the front camera at the end of the last execution." in replan.text
