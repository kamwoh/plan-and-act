"""v1.15: one front camera frame at the end of each keyframe of an execution, and the pick of the frames for Claude."""
import pytest

from armmind.plan.formats import Keyframe
from armmind.plan.frames import MAX_FRAMES, pick_frames
from armmind.scenes import SCENES
from tests.test_plan_scores import scripted_keyframes, with_holding


def frames(n, stop=False):
    out = [{"keyframe": k, "of": n, "note": f"kf {k}", "at": "end"} for k in range(1, n + 1)]
    if stop:
        out[-1]["at"] = "stop"
    return out


@pytest.mark.parametrize("n, sent", [(1, [0]), (3, [0, 1, 2]), (4, [0, 1, 2, 3]), (5, [1, 2, 3, 4]),
                                     (12, [2, 5, 8, 11])])
def test_pick_takes_at_most_4_at_even_gaps_with_the_last(n, sent):
    assert pick_frames(frames(n), end_images=False) == sent
    assert MAX_FRAMES == 4


@pytest.mark.parametrize("n, sent", [(1, []), (3, [0, 1]), (4, [0, 1, 2]), (5, [1, 2, 3]), (12, [2, 5, 8])])
def test_pick_skips_the_last_frame_when_the_end_image_shows_it(n, sent):
    assert pick_frames(frames(n), end_images=True) == sent


def test_pick_keeps_the_last_frame_of_a_stop_or_a_cancel():
    stopped = frames(5, stop=True)
    assert pick_frames(stopped, end_images=True) == [1, 2, 3, 4]
    cancelled = frames(5)[:3]  # keyframes 1 to 3 of 5 ended before the cancel
    assert pick_frames(cancelled, end_images=True) == [0, 1, 2]
    assert pick_frames([], end_images=True) == []


def _env(task="pick", seed=0):
    from armmind.env import MujocoEnv
    sc = SCENES["cubes"]
    return MujocoEnv(sc, sc.tasks[task], seed, sc.tasks[task].instruction)


def _labels(out):
    return [(f["keyframe"], f["of"], f["note"], f["at"]) for f in out["frames"]]


@pytest.mark.assets
def test_execute_captures_one_frame_at_the_end_of_each_keyframe():
    env = _env()
    try:
        kfs = tuple(with_holding(scripted_keyframes(SCENES["cubes"].tasks["pick"].solution(env.world)), env.world))
        kfs = tuple(Keyframe(k.xyz, k.yaw, k.state, k.carrying, f"note {i}", k.moves) for i, k in enumerate(kfs, 1))
        out = env.execute_plan(kfs, lambda: False)
    finally:
        env.close()
    n = len(kfs)
    assert out["status"] == "success" and len(out["true_after"]) == n
    assert _labels(out) == [(k, n, f"note {k}", "end") for k in range(1, n + 1)]
    assert all(f["png"].startswith(b"\x89PNG") for f in out["frames"])


@pytest.mark.assets
def test_a_stop_inside_a_keyframe_adds_a_stop_frame():
    kfs = (Keyframe((0.35, 0.0, 0.15), 0.0, "open", None, "above"), Keyframe((0.35, 0.0, 0.02), 0.0, "closed",
           "red cube", "grasp"), Keyframe((0.35, 0.0, 0.2), 0.0, "closed", "red cube", "lift"))
    env = _env()
    try:
        out = env.execute_plan(kfs, lambda: False)
    finally:
        env.close()
    assert out["status"] == "stopped"  # the close of keyframe 2 closes on nothing
    assert _labels(out) == [(1, 3, "above", "end"), (2, 3, "grasp", "stop")]


@pytest.mark.assets
def test_a_cancel_keeps_the_frames_that_it_captured():
    kfs = tuple(Keyframe((0.4, y, 0.3), 0.0, "open", None, f"point {i}") for i, y in enumerate((0.0, 0.1, 0.2), 1))
    calls = []

    def stop():
        calls.append(1)
        return len(calls) > 2  # the third action never starts

    env = _env()
    try:
        out = env.execute_plan(kfs, stop)
    finally:
        env.close()
    assert out["status"] == "cancelled"
    assert _labels(out) == [(1, 3, "point 1", "end"), (2, 3, "point 2", "end")]


PLAN5 = {"objects": [{"name": "red cube", "shape": "box", "size": [0.04, 0.04, 0.04], "xyz": [0.5, -0.1, 0.02],
                      "yaw": 0}],
         "keyframes": [{"xyz": [0.5, -0.1, z], "yaw": 0, "gripper": "open", "holding": "nothing", "note": f"step {k}"}
                       for k, z in enumerate((0.3, 0.25, 0.2, 0.15, 0.1), 1)],
         "summary": "Go down above the cube."}


def _executed(tmp_path, factory=None):
    from fastapi.testclient import TestClient

    from armmind.llm.fake import FakeBackend
    from armmind.server.app import create_app
    from tests.test_agent_plan import good_plan
    from tests.test_plan_api import PICK, plan_factory
    from tests.test_server import wait_idle
    app = create_app(FakeBackend([good_plan(PLAN5)]), factory or plan_factory, tmp_path / "runs",
                     lessons_dir=tmp_path / "lessons")
    with TestClient(app) as c:
        run_id = c.post("/api/plan", json={"instruction": PICK}).json()["run_id"]
        wait_idle(c)
        with c.websocket_connect("/ws") as ws:
            assert c.post(f"/api/plan/{run_id}/execute").status_code == 200
            live = None
            while live is None:
                m = ws.receive_json()
                live = m if m.get("kind") == "exec_end" else None
        wait_idle(c)
        replay = c.get(f"/api/runs/{run_id}").json()
        end = next(e for e in replay if e["kind"] == "exec_end")
        served = [c.get(f["url"]) for f in end["frames"]]
    return run_id, live, end, served


def test_exec_end_lists_the_saved_frames_in_order(tmp_path):
    run_id, live, end, served = _executed(tmp_path)
    names = [f"step002_kf{k:02d}.png" for k in range(1, 6)]
    assert [f["file"] for f in live["frames"]] == names
    assert [(f["keyframe"], f["of"], f["note"], f["at"]) for f in live["frames"]] == [
        (k, 5, f"step {k}", "end") for k in range(1, 6)]
    assert [f["sent"] for f in live["frames"]] == [False, True, True, True, False]  # 3 frames and the end images
    assert [(tmp_path / "runs" / run_id / n).read_bytes() for n in names] == [f"frame{k}".encode() for k in range(1, 6)]
    assert [f["url"] for f in end["frames"]] == [f"/api/runs/{run_id}/images/{n}" for n in names]
    assert [r.status_code for r in served] == [200] * 5 and served[0].content == b"frame1"


def test_a_stop_frame_is_saved_and_sent(tmp_path):
    from tests.test_plan_api import fake_frames, plan_factory

    def factory(*args):
        env = plan_factory(*args)

        def execute_plan(keyframes, should_stop, on_action=None):
            return {"actions": ["a"], "results": ["r"], "stopped": {"after": 1, "of": 9, "reason": "blocked"},
                    "success": False, "status": "stopped", "frames": fake_frames(keyframes, stop_in=3)}

        env.execute_plan = execute_plan
        return env

    _, _, end, _ = _executed(tmp_path, factory)
    assert [(f["file"], f["at"], f["sent"]) for f in end["frames"]] == [
        ("step002_kf01.png", "end", True), ("step002_kf02.png", "end", True), ("step002_kf03_stop.png", "stop", True)]


def test_frame_names_are_valid_image_names():
    from armmind.store.runs import valid_image_name
    assert valid_image_name("step002_kf01.png") and valid_image_name("step012_kf40_stop.png")
    assert not valid_image_name("step002_kf1.png") and not valid_image_name("step002_kf01_x.png")
