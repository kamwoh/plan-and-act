"""The poses of the entities along a track: carries, releases and the predicted moves."""
import pytest

from plan_and_act.plan.track import poses_after, poses_at, wrap_deg


def kf(xyz, carrying=None, t=1.0, moves=(), yaw=0.0):
    return {"xyz": list(xyz), "yaw": yaw, "state": "closed" if carrying else "open", "carrying": carrying,
            "gap": 0.0, "changes": False, "t": t, "t_done": t + 0.5, "moves": list(moves)}


def track(*kfs):
    return {"start": {"xyz": [0.3, 0.0, 0.4], "yaw": 0.0, "state": "open", "gap": 0.08}, "keyframes": list(kfs),
            "duration": kfs[-1]["t_done"]}


CUBE = {"name": "cube", "xyz": [0.5, 0.0, 0.02], "yaw": 0.0}
BALL = {"name": "ball", "xyz": [0.6, 0.2, 0.02], "yaw": 0.0}


def xyz(poses, name):
    return pytest.approx(poses[name]["xyz"], abs=1e-9)


def test_no_move_keeps_every_entity_still():
    tr = track(kf((0.5, 0.0, 0.1), t=1.0), kf((0.5, 0.1, 0.1), t=3.0))
    for t in (0.0, 1.2, 2.0, 3.5):
        p = poses_at(tr, [CUBE, BALL], t)
        assert p["cube"]["xyz"] == CUBE["xyz"] and p["ball"]["xyz"] == BALL["xyz"]


def test_a_move_follows_the_agent_timing_and_then_stays():
    push = {"name": "cube", "xyz": [0.5, 0.1, 0.02], "yaw": 30.0}
    tr = track(kf((0.5, -0.05, 0.02), t=1.0), kf((0.5, 0.07, 0.02), t=3.5, moves=[push]), kf((0.5, 0.07, 0.2), t=5.0))
    assert poses_at(tr, [CUBE], 1.5)["cube"]["xyz"] == CUBE["xyz"]  # the segment starts at t_done of keyframe 1
    half = poses_at(tr, [CUBE, BALL], 2.5)  # half way: smoothstep(0.5) is 0.5
    assert half["cube"]["xyz"] == xyz({"cube": {"xyz": [0.5, 0.05, 0.02]}}, "cube")
    assert half["cube"]["yaw"] == pytest.approx(15.0) and half["ball"]["xyz"] == BALL["xyz"]
    for t in (3.5, 4.0, 5.5):
        p = poses_at(tr, [CUBE], t)
        assert p["cube"]["xyz"] == xyz({"cube": push}, "cube") and p["cube"]["yaw"] == pytest.approx(30.0)
    after = poses_after(tr, [CUBE])
    assert [a["cube"]["xyz"][1] for a in after] == pytest.approx([0.0, 0.1, 0.1])


def test_a_move_turns_the_short_way_round():
    tr = track(kf((0.5, 0.0, 0.1), t=1.0, moves=[{"name": "cube", "xyz": CUBE["xyz"], "yaw": -170.0}]))
    start = {**CUBE, "yaw": 170.0}
    assert poses_at(tr, [start], 0.5)["cube"]["yaw"] == pytest.approx(180.0)  # 170 -> 190, not 170 -> -170
    assert wrap_deg(poses_at(tr, [start], 1.0)["cube"]["yaw"]) == pytest.approx(-170.0)


def test_a_move_after_a_release_starts_where_the_hand_let_go():
    place = {"name": "cube", "xyz": [0.4, 0.3, 0.02], "yaw": 0.0}
    tr = track(kf((0.5, 0.0, 0.02), "cube", t=1.0), kf((0.5, 0.2, 0.05), "cube", t=2.0),
               kf((0.5, 0.2, 0.05), None, t=3.0), kf((0.5, 0.3, 0.05), t=4.0, moves=[place]))
    after = poses_after(tr, [CUBE])
    assert after[1]["cube"]["xyz"] == xyz({"cube": {"xyz": [0.5, 0.2, 0.05]}}, "cube")  # carried
    assert after[2]["cube"]["xyz"] == xyz({"cube": {"xyz": [0.5, 0.2, 0.05]}}, "cube")  # let go there
    assert after[3]["cube"]["xyz"] == xyz({"cube": place}, "cube")
    mid = poses_at(tr, [CUBE], 3.5 + 0.25)  # half way through the segment into keyframe 4
    assert mid["cube"]["xyz"] == pytest.approx([0.45, 0.25, 0.035])


def test_a_carry_after_a_move_starts_from_the_moved_pose():
    push = {"name": "cube", "xyz": [0.5, 0.1, 0.02], "yaw": 0.0}
    tr = track(kf((0.5, 0.06, 0.02), t=1.0, moves=[push]), kf((0.5, 0.1, 0.02), "cube", t=2.0),
               kf((0.5, 0.1, 0.3), "cube", t=3.0))
    after = poses_after(tr, [CUBE])
    assert after[1]["cube"]["xyz"] == xyz({"cube": push}, "cube")
    assert after[2]["cube"]["xyz"] == pytest.approx([0.5, 0.1, 0.3])


def test_a_carried_entity_ignores_a_move_during_the_segment():
    tr = track(kf((0.5, 0.0, 0.02), "cube", t=1.0),
               kf((0.5, 0.0, 0.3), None, t=3.0, moves=[{"name": "cube", "xyz": [0.7, 0.0, 0.02], "yaw": 0.0}]))
    mid = poses_at(tr, [CUBE], 2.25)
    assert mid["cube"]["xyz"][0] == pytest.approx(0.5) and 0.02 < mid["cube"]["xyz"][2] < 0.3
    assert poses_at(tr, [CUBE], 3.0)["cube"]["xyz"] == pytest.approx([0.7, 0.0, 0.02])


def test_timed_keyframes_keep_the_moves_of_believed_entities_only():
    from plan_and_act.plan.formats import Entity, Keyframe, Move
    from plan_and_act.sim.planview import timed_keyframes
    belief = [Entity("blue cube", "box", (0.04, 0.04, 0.04), (0.5, 0.1, 0.02))]
    k = Keyframe((0.5, 0.0, 0.02), 0.0, "open", None, "push",
                 (Move("Blue_Cube", (0.5, 0.15, 0.02), 5.0), Move("ghost", (0.4, 0.0, 0.02))))
    assert timed_keyframes(belief, [k])[0].moves == (Move("blue cube", (0.5, 0.15, 0.02), 5.0),)


# ---- The same poses in the page player ----
def test_python_and_the_page_player_give_the_same_poses():
    import json
    import subprocess
    from pathlib import Path
    case_file = Path(__file__).parent / "js" / "track_case.json"
    case = json.loads(case_file.read_text())
    out = subprocess.run(["node", str(case_file.parent / "track_check.js"), str(case_file)], check=True,
                         capture_output=True, text=True).stdout
    page = json.loads(out)
    assert len(page) == len(case["times"])
    for t, js in zip(case["times"], page, strict=True):
        py = poses_at(case["track"], case["entities"], t)
        assert set(py) == set(js)
        for name in py:
            assert py[name]["xyz"] == pytest.approx(js[name]["xyz"], abs=1e-9), (t, name)
            assert py[name]["yaw"] == pytest.approx(js[name]["yaw"], abs=1e-9), (t, name)
    for e in case["expect"]:
        p = poses_at(case["track"], case["entities"], e["t"])[e["name"]]
        assert p["xyz"] == pytest.approx(e["xyz"], abs=1e-9) and p["yaw"] == pytest.approx(e["yaw"], abs=1e-9)


def test_a_release_drops_onto_the_table_or_the_highest_entity_below():
    cube = {**CUBE, "shape": "box", "size": [0.04, 0.04, 0.04]}
    plate = {"name": "plate", "shape": "cylinder", "size": [0.12, 0.12, 0.02], "xyz": [0.5, 0.25, 0.01], "yaw": 0.0}
    tr = track(kf((0.5, 0.0, 0.02), "cube", t=1.0), kf((0.5, 0.25, 0.15), "cube", t=2.0),
               kf((0.5, 0.25, 0.15), None, t=3.0), kf((0.5, 0.25, 0.3), t=4.0))
    after = poses_after(tr, [cube, plate])
    assert after[1]["cube"]["xyz"][2] == pytest.approx(0.15)
    assert after[2]["cube"]["xyz"] == pytest.approx([0.5, 0.25, 0.04])  # on the plate top at 0.02
    alone = poses_after(tr, [cube])
    assert alone[2]["cube"]["xyz"] == pytest.approx([0.5, 0.25, 0.02])  # on the table
