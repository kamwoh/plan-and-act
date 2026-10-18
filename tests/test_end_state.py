"""The end state of an execution: save it, and start a new env from it."""
import numpy as np
import pytest

from armmind.env import MujocoEnv
from armmind.llm.fake import DEMO_PLAN_LOW
from armmind.plan.formats import Keyframe
from armmind.scenes import SCENES
from armmind.sim.state import StateMismatch, scene_version, state_problem

CUBES = SCENES["cubes"]
PICK = CUBES.tasks["pick"]


def low_keyframes():
    out = []
    for k in DEMO_PLAN_LOW["keyframes"]:
        carrying = None if k["holding"] == "nothing" else k["holding"]
        out.append(Keyframe(tuple(k["xyz"]), k["yaw"], k["gripper"], carrying, k["note"]))
    return tuple(out)


def poses(env):
    return {o.name: env.world.object_pos(o.name) for o in env.world.objects}


@pytest.fixture(scope="module")
def ended(tmp_path_factory):
    """An env after the low pick of the demo, its end state file and its values at the end."""
    path = tmp_path_factory.mktemp("state") / "end_state.npz"
    env = MujocoEnv(CUBES, PICK, 0, PICK.instruction)
    out = env.execute_plan(low_keyframes(), lambda: False)
    env.save_state(path)
    end = {"poses": poses(env), "width": env.world.gripper_width(), "q": env.world.data.qpos[:7].copy(),
           "tcp": env.world.tcp(), "yaw": env.world.gripper_yaw(), "out": out, "obs": env.observe()}
    env.close()
    return path, end


@pytest.mark.assets
def test_a_new_env_starts_at_the_saved_end_state(ended):
    path, end = ended
    assert end["out"]["status"] == "failed" and end["poses"]["red_cube"][2] > 0.05  # held up, not 10 cm
    env = MujocoEnv(CUBES, PICK, 0, PICK.instruction, start=path)
    try:
        for name, p in poses(env).items():
            assert np.linalg.norm(p - end["poses"][name]) < 1e-3, name
        assert env.world.gripper_width() == pytest.approx(end["width"], abs=1e-6)
        assert np.allclose(env.world.data.qpos[:7], end["q"], atol=1e-9)
        assert np.allclose(env.world.tcp(), end["tcp"], atol=1e-6)
        obs = env.observe()
        assert (obs.tcp, obs.gripper_width, obs.gripper_yaw) == (end["obs"].tcp, end["obs"].gripper_width,
                                                                  end["obs"].gripper_yaw)
        assert obs.contacts == end["obs"].contacts
        assert env.gripper_state() == "closed" and env.arm.held() == ("red_cube",)
    finally:
        env.close()


@pytest.mark.assets
def test_the_saved_state_keeps_the_scene_seed_and_version(ended):
    path, _ = ended
    with np.load(path, allow_pickle=False) as f:
        assert (str(f["scene"]), int(f["seed"]), str(f["version"])) == ("cubes", 0, scene_version(CUBES))
        assert {"qpos", "qvel", "act", "ctrl", "time"} <= set(f.files)
    assert state_problem(path, CUBES, 0) is None


@pytest.mark.assets
def test_a_state_of_another_scene_or_seed_is_refused(ended, tmp_path):
    path, _ = ended
    tray = SCENES["tray"]
    assert "cubes" in state_problem(path, tray, 0)
    assert "seed" in state_problem(path, CUBES, 4)
    assert "no end state" in state_problem(tmp_path / "missing.npz", CUBES, 0)
    (tmp_path / "broken.npz").write_bytes(b"not a file of numpy")
    assert "cannot read" in state_problem(tmp_path / "broken.npz", CUBES, 0)
    with pytest.raises(StateMismatch):
        MujocoEnv(tray, None, 0, "", start=path)


@pytest.mark.assets
def test_an_old_scene_version_is_refused(ended, tmp_path):
    path, _ = ended
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    data["version"] = np.array("0000")
    old = tmp_path / "old.npz"
    np.savez(old, **data)
    assert "version" in state_problem(old, CUBES, 0)


@pytest.mark.assets
def test_execute_from_the_saved_state_goes_on_from_there(ended):
    path, end = ended
    env = MujocoEnv(CUBES, PICK, 0, PICK.instruction, start=path)
    try:
        lift = (Keyframe((0.56, -0.11, 0.20), 0.0, "closed", "red cube", "lift it higher"),)
        out = env.execute_plan(lift, lambda: False)
        assert out["actions"] == ["move_to(0.560, -0.110, 0.200, yaw 0°)"]  # no close: the gripper is closed already
        assert out["true_start"]["red_cube"]["xyz"] == pytest.approx(list(end["poses"]["red_cube"]), abs=1e-3)
        assert out["status"] == "success"
    finally:
        env.close()


@pytest.mark.assets
def test_the_plan_check_and_the_view_start_with_the_cube_in_the_hand(ended):
    from armmind.plan.formats import Entity
    from armmind.plan.scores import prediction
    path, end = ended
    env = MujocoEnv(CUBES, PICK, 0, PICK.instruction, start=path)
    try:
        red = end["poses"]["red_cube"]
        belief = (Entity("red cube", "box", (0.04, 0.04, 0.04), tuple(float(v) for v in red), 0.0),)
        lift = (Keyframe((0.56, -0.11, 0.20), 0.0, "closed", "red cube", "lift it higher"),)
        scores = env.check_plan(belief, lift)
        assert scores["marks"][0]["holds"] == "red_cube" and "grasp" not in scores["marks"][0]["checks"]
        assert scores["perception"][0]["xy_cm"] < 1.0  # against the cube where it is now, not at the reset
        view = env.plan_view(belief, lift, scores["marks"])
        assert view["track"]["start"]["state"] == "closed" and view["track"]["start"]["carrying"] == "red cube"
        true_red = next(e for e in view["true_start"] if e["name"] == "red_cube")
        assert true_red["xyz"] == pytest.approx(list(red), abs=1e-3)
        out = env.execute_plan(lift, lambda: False)
        score = prediction(view["track"], view["believed"], out["true_start"], out["true_after"], view["true_match"])
        row = score["rows"][0]
        assert row["agree"] is True and abs(row["final_z_cm"]) < 2.0, score  # the believed cube rose with the hand
    finally:
        env.close()


def test_a_track_can_start_with_an_entity_in_the_hand_in_python_and_on_the_page(tmp_path):
    import json
    import subprocess
    from pathlib import Path

    from armmind.plan.track import poses_at
    cube = {"name": "cube", "shape": "box", "size": [0.04, 0.04, 0.04], "xyz": [0.5, 0.0, 0.08], "yaw": 0.0}
    track = {"start": {"xyz": [0.5, 0.0, 0.08], "yaw": 0.0, "state": "closed", "gap": 0.04, "carrying": "cube"},
             "keyframes": [{"xyz": [0.5, 0.1, 0.2], "yaw": 30.0, "state": "closed", "carrying": "cube", "gap": 0.04,
                            "changes": False, "t": 1.0, "t_done": 1.3, "moves": []},
                           {"xyz": [0.5, 0.1, 0.2], "yaw": 30.0, "state": "open", "carrying": None, "gap": 0.08,
                            "changes": True, "t": 1.3, "t_done": 1.9, "moves": []}],
             "duration": 1.9}
    times = [0.0, 0.5, 1.0, 1.9]
    py = [poses_at(track, [cube], t) for t in times]
    assert py[1]["cube"]["xyz"][2] > 0.08 and py[2]["cube"]["xyz"] == pytest.approx([0.5, 0.1, 0.2])
    assert py[3]["cube"]["xyz"] == pytest.approx([0.5, 0.1, 0.02])  # let go: it falls onto the table
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"track": track, "entities": [cube], "times": times}))
    script = Path(__file__).parent / "js" / "track_check.js"
    js = json.loads(subprocess.run(["node", str(script), str(case)], check=True, capture_output=True,
                                   text=True).stdout)
    for a, b in zip(py, js, strict=True):
        assert a["cube"]["xyz"] == pytest.approx(b["cube"]["xyz"], abs=1e-9)
        assert a["cube"]["yaw"] == pytest.approx(b["cube"]["yaw"], abs=1e-9)
