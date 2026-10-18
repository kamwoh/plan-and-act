"""Scores of a plan: the belief against the true scene, and the checks of each keyframe."""
import pytest

from armmind.plan.formats import Entity, Keyframe
from armmind.plan.scores import drop, perception, plan_checks, yaw_error
from armmind.scenes import SCENES
from armmind.types import Action

PARTS = [  # the agent of the arm environment, as in sim/planview.py
    {"name": "hand", "shape": "box", "size": [0.20, 0.058, 0.075], "xyz": [0.0, 0.0, 0.0825], "slide": 0},
    {"name": "finger_left", "shape": "box", "size": [0.027, 0.021, 0.057], "xyz": [0.0135, 0.0, 0.0185], "slide": 1},
    {"name": "finger_right", "shape": "box", "size": [0.027, 0.021, 0.057], "xyz": [-0.0135, 0.0, 0.0185], "slide": -1},
]
START = {"xyz": (0.3, 0.0, 0.4), "yaw": 0.0, "state": "open"}
CUBE = Entity("red_cube", "box", (0.04, 0.04, 0.04), (0.5, -0.1, 0.02), 0.0)
BLUE = Entity("blue_cube", "box", (0.04, 0.04, 0.04), (0.5, 0.1, 0.02), 0.0)
LONG = Entity("long_box", "box", (0.04, 0.10, 0.05), (0.5, 0.0, 0.025), 0.0)
ALIASES = {"red_cube": ["red cube, 4 cm", "red cube"], "blue_cube": ["blue cube, 4 cm", "blue cube"],
           "long_box": ["long box"], "can": ["tin can"]}


def no_reach_limit(xyz, yaw):
    return None


def arm_gap(state, held):  # the gap rule of the arm environment
    if state == "open":
        return 0.08
    return min(0.08, held.size[0], held.size[1]) if held is not None else 0.0


def checks(keyframes, truth, round_names=()):
    return plan_checks(START, keyframes, truth, ALIASES, set(round_names), PARTS, no_reach_limit,
                       open_state="open", gap=arm_gap)


# ---- Perception ----
def test_a_perfect_belief_has_no_error():
    p = perception([CUBE, BLUE], [CUBE, BLUE], ALIASES, set())
    assert [r["true_name"] for r in p["rows"]] == ["red_cube", "blue_cube"]
    for r in p["rows"]:
        assert r["xy_cm"] == 0 and r["z_cm"] == 0 and r["size_cm"] == [0, 0, 0] and r["yaw_deg"] == 0
    assert p["missing"] == [] and p["invented"] == []
    assert p["summary"] == {"mean_xy_cm": 0.0, "max_xy_cm": 0.0, "matched": 2, "missing": 0, "invented": 0}


def test_a_5_cm_shift_gives_a_5_cm_xy_error():
    moved = Entity("Red Cube", "box", (0.05, 0.04, 0.03), (0.53, -0.06, 0.03), 0.0)
    p = perception([moved], [CUBE], ALIASES, set())
    [r] = p["rows"]
    assert r["name"] == "Red Cube" and r["xy_cm"] == pytest.approx(5.0)
    # The sizes compare in the canonical form, with the shorter side first: 5 x 4 cm becomes 4 x 5 cm.
    assert r["z_cm"] == pytest.approx(1.0) and r["size_cm"] == pytest.approx([0.0, 1.0, -1.0])
    assert p["summary"]["mean_xy_cm"] == pytest.approx(5.0) and p["summary"]["max_xy_cm"] == pytest.approx(5.0)


@pytest.mark.parametrize("true, believed, want", [
    (CUBE, 95.0, 5.0),  # a square footprint repeats every 90°
    (CUBE, 44.0, 44.0),
    (CUBE, 46.0, 44.0),
    (LONG, 185.0, 5.0),  # a box repeats every 180°
    (LONG, 90.0, 90.0),
    (LONG, -170.0, 10.0),
    (Entity("can", "cylinder", (0.06, 0.06, 0.1), (0.5, 0, 0.05)), 77.0, None),  # round: no yaw error
])
def test_yaw_error_folds_by_the_symmetry(true, believed, want):
    got = yaw_error(true, believed, round_names=set())
    assert got == pytest.approx(want) if want is not None else got is None


def test_a_round_mesh_has_no_yaw_error():
    bowl = Entity("bowl", "box", (0.15, 0.15, 0.06), (0.6, 0.2, 0.03), 30.0)
    assert yaw_error(bowl, 0.0, round_names={"bowl"}) is None


def test_missing_and_invented_objects():
    ghost = Entity("purple ball", "box", (0.04, 0.04, 0.04), (0.4, 0.0, 0.02), 0.0)
    blue_label = Entity("blue cube", "box", (0.04, 0.04, 0.04), (0.5, 0.1, 0.02), 0.0)
    p = perception([ghost, blue_label], [CUBE, BLUE], ALIASES, set())
    assert [r["true_name"] for r in p["rows"]] == ["blue_cube"]  # the label matches too
    assert p["missing"] == ["red_cube"] and p["invented"] == ["purple ball"]
    assert p["summary"]["missing"] == 1 and p["summary"]["invented"] == 1


# ---- Keyframe checks ----
def pick(obj: Entity, yaw=0.0):
    x, y, z = obj.xyz
    return [Keyframe((x, y, 0.15), yaw, "open", None, "above"), Keyframe((x, y, z), yaw, "closed", obj.name, "grasp"),
            Keyframe((x, y, 0.2), yaw, "closed", obj.name, "lift")]


def test_a_clean_pick_passes_every_check():
    marks = checks(pick(CUBE), [CUBE, BLUE])
    assert all(m["ok"] for m in marks)
    assert marks[1]["checks"] == {"reach": "ok", "collision": "ok", "grasp": "ok"}


def test_a_path_through_a_true_object_fails_collision():
    # Along x the open fingers hit the cube. Along y they would straddle it.
    kfs = [Keyframe((0.35, 0.1, 0.15), 0.0, "open", None), Keyframe((0.35, 0.1, 0.02), 0.0, "open", None),
           Keyframe((0.65, 0.1, 0.02), 0.0, "open", None, "sweep through the blue cube")]
    marks = checks(kfs, [CUBE, BLUE])
    assert [m["ok"] for m in marks] == [True, True, False]
    assert "blue_cube" in marks[2]["checks"]["collision"]


def test_the_grasp_target_is_ignored_in_the_last_3_cm_before_a_close():
    # A grasp at the centre of a 12 cm tall box: the hand passes the box top 1.5 cm before the close.
    tall = Entity("long_box", "box", (0.04, 0.04, 0.12), (0.5, 0.0, 0.06), 0.0)
    marks = checks(pick(tall), [tall])
    assert marks[1]["checks"]["collision"] == "ok" and marks[1]["checks"]["grasp"] == "ok"
    no_close = [pick(tall)[0], Keyframe((0.5, 0.0, 0.06), 0.0, "open", None)]
    assert "long_box" in checks(no_close, [tall])[1]["checks"]["collision"]  # without a close, the box counts


def test_the_held_object_does_not_collide_and_moves_with_the_hand():
    kfs = [*pick(CUBE), Keyframe((0.5, 0.25, 0.2), 0.0, "closed", "red cube"),
           Keyframe((0.5, 0.25, 0.022), 0.0, "open", None, "put it down"),
           Keyframe((0.5, 0.25, 0.2), 0.0, "open", None),
           Keyframe((0.55, 0.25, 0.02), 0.0, "open", None, "down beside the cube: a finger hits it")]
    marks = checks(kfs, [CUBE, BLUE])
    assert all(m["ok"] for m in marks[:6])
    assert marks[4]["checks"]["release"] == "ok" and "0.2 cm above the table" in marks[4]["info"]
    assert "red_cube" in marks[6]["checks"]["collision"]  # the cube now stands at the release point


def test_a_released_object_falls_and_never_rises():
    high = [*pick(CUBE), Keyframe((0.5, 0.25, 0.2), 0.0, "closed", "red cube"),
            Keyframe((0.5, 0.25, 0.12), 0.0, "open", None, "let go 10 cm up"),
            Keyframe((0.55, 0.25, 0.12), 0.0, "open", None, "a finger would hit the cube in the air")]
    marks = checks(high, [CUBE, BLUE])
    assert marks[4]["info"] == "red_cube: 10.0 cm above the table" and marks[5]["ok"]  # the cube fell
    bowl = Entity("bowl", "cylinder", (0.16, 0.16, 0.06), (0.5, 0.25, 0.03), 0.0)
    by_name = {"red_cube": CUBE, "bowl": bowl}
    sunk = {"red_cube": ((0.5, 0.25, 0.075), 0.0), "bowl": (bowl.xyz, 0.0)}  # its bottom is inside the bowl
    assert drop("red_cube", by_name, sunk) == ((0.5, 0.25, 0.075), 0.0)  # not lifted onto the rim
    above = {"red_cube": ((0.5, 0.25, 0.2), 30.0), "bowl": (bowl.xyz, 0.0)}
    assert drop("red_cube", by_name, above) == ((0.5, 0.25, pytest.approx(0.08)), 30.0)  # it lands on the bowl


def test_a_grasp_across_the_wide_side_fails():
    marks = checks(pick(LONG, yaw=90.0), [LONG])
    assert marks[1]["ok"] is False and "narrow side" in marks[1]["checks"]["grasp"]
    assert checks(pick(LONG, yaw=0.0), [LONG])[1]["checks"]["grasp"] == "ok"
    assert checks(pick(LONG, yaw=-15.0), [LONG])[1]["checks"]["grasp"] == "ok"  # within 20°


def test_a_grasp_needs_a_narrow_object_and_the_centre():
    wide = Entity("long_box", "box", (0.09, 0.12, 0.05), (0.5, 0.0, 0.025), 0.0)
    assert "too wide" in checks(pick(wide), [wide])[1]["checks"]["grasp"]
    off = pick(CUBE)
    off[1] = Keyframe((0.5, -0.07, 0.02), 0.0, "closed", "red_cube")
    assert "cm from the centre" in checks(off, [CUBE])[1]["checks"]["grasp"]
    unknown = pick(CUBE)
    unknown[1] = Keyframe((0.5, -0.1, 0.02), 0.0, "closed", "teapot")
    assert "teapot" in checks(unknown, [CUBE])[1]["checks"]["grasp"]


def test_a_round_object_takes_any_grasp_angle():
    can = Entity("can", "cylinder", (0.06, 0.06, 0.1), (0.5, 0.0, 0.05), 0.0)
    assert checks(pick(can, yaw=55.0), [can], round_names={"can"})[1]["checks"]["grasp"] == "ok"


def test_reach_uses_the_environment_rule():
    def reach(xyz, yaw):
        return "no joint solution for this point" if xyz[1] > 0.3 else None

    kfs = [Keyframe((0.5, 0.35, 0.2), 0.0, "open", None)]
    marks = plan_checks(START, kfs, [], ALIASES, set(), PARTS, reach, open_state="open", gap=arm_gap)
    assert marks[0]["ok"] is False and marks[0]["checks"]["reach"] == "no joint solution for this point"


# ---- The MuJoCo environment ----
def scripted_keyframes(actions: list[Action]) -> list[Keyframe]:
    """The keyframes of a scripted solution: a move is a keyframe, an open or close changes the last one."""
    kfs, state, holding, yaw = [], "open", None, 0.0
    for a in actions:
        if a.type == "move_to":
            yaw = a.yaw if a.yaw is not None else yaw
            kfs.append(Keyframe(tuple(float(v) for v in a.xyz), yaw, state, holding))
        elif a.type in ("open", "close") and kfs:
            state = "open" if a.type == "open" else "closed"
            k = kfs[-1]
            kfs[-1] = Keyframe(k.xyz, k.yaw, state, k.carrying, k.note)
    return kfs


def with_holding(kfs: list[Keyframe], world) -> list[Keyframe]:
    """Set "holding" at each close to the object within 3 cm of the hand in xy, as the scripted plan means."""
    import numpy as np
    out, held, state = [], None, "open"
    for k in kfs:
        if k.state == "closed" and state == "open":
            dist = {o.name: float(np.hypot(*(world.object_pos(o.name)[:2] - np.array(k.xyz[:2]))))
                    for o in world.objects}
            near = min(dist, key=dist.get)
            held = near if dist[near] < 0.03 else None
        held = held if k.state == "closed" else None
        out.append(Keyframe(k.xyz, k.yaw, k.state, held, k.note))
        state = k.state
    return out


@pytest.mark.assets
@pytest.mark.parametrize("scene, task", [("cubes", "pick"), ("tray", "red_in_tray"), ("cubes", "stack"),
                                         ("tray", "all_in_tray"), ("drawer", "cube_in_drawer")])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_true_belief_and_a_scripted_plan_score_clean(scene, task, seed):
    from armmind.env import MujocoEnv
    from armmind.sim.planview import true_entities
    sc = SCENES[scene]
    env = MujocoEnv(sc, sc.tasks[task], seed, sc.tasks[task].instruction)
    try:
        belief = tuple(true_entities(env.world))
        kfs = tuple(with_holding(scripted_keyframes(sc.tasks[task].solution(env.world)), env.world))
        scores = env.check_plan(belief, kfs)
    finally:
        env.close()
    for r in scores["perception"]:
        assert r["xy_cm"] < 0.01 and abs(r["z_cm"]) < 0.01 and r["yaw_deg"] == pytest.approx(0, abs=0.01)
    assert scores["missing"] == [] and scores["invented"] == []
    bad = [(i, m) for i, m in enumerate(scores["marks"]) if not m["ok"]]
    assert bad == [] and scores["summary"]["failed_keyframes"] == 0


@pytest.mark.assets
def test_reach_fails_outside_the_workspace():
    from armmind.env import MujocoEnv
    env = MujocoEnv(SCENES["cubes"], None, 0, "")
    try:
        scores = env.check_plan((), (Keyframe((0.95, 0.0, 0.2), 0.0, "open", None),
                                     Keyframe((0.5, 0.0, 0.2), 0.0, "open", None)))
    finally:
        env.close()
    assert "outside the workspace" in scores["marks"][0]["checks"]["reach"]
    assert scores["marks"][1]["checks"]["reach"] == "ok"
    assert len(scores["missing"]) == 3 and scores["summary"]["failed_keyframes"] == 1


# ---- Review follow-ups ----
def test_a_box_and_the_same_box_turned_90_with_swapped_sizes_score_the_same():
    true = Entity("long_box", "box", (0.065, 0.046, 0.08), (0.5, 0.0, 0.04), 165.2)
    same = Entity("long box", "box", (0.046, 0.065, 0.08), (0.5, 0.0, 0.04), 75.2)
    [r] = perception([same], [true], ALIASES, set())["rows"]
    assert r["yaw_deg"] == 0 and r["size_cm"] == [0, 0, 0]
    turned = Entity("long box", "box", (0.065, 0.046, 0.08), (0.5, 0.0, 0.04), 75.2)
    [r] = perception([turned], [true], ALIASES, set())["rows"]
    assert r["yaw_deg"] == pytest.approx(90.0) and r["size_cm"] == [0, 0, 0]


def test_descend_open_then_close_in_place_exempts_the_target():
    tall = Entity("long_box", "box", (0.04, 0.04, 0.12), (0.5, 0.0, 0.06), 0.0)
    kfs = [Keyframe((0.5, 0.0, 0.2), 0.0, "open", None, "above"),
           Keyframe((0.5, 0.0, 0.06), 0.0, "open", None, "down"),
           Keyframe((0.5, 0.0, 0.06), 0.0, "closed", "long_box", "close in place")]
    marks = checks(kfs, [tall])
    assert [m["checks"]["collision"] for m in marks] == ["ok", "ok", "ok"]
    assert marks[2]["checks"]["grasp"] == "ok"


def test_a_pure_yaw_turn_sweeps_the_hand_through_an_object():
    # The cube floats at hand height. The hand clears it at yaw 0 and at yaw 90, and hits it at yaw 45.
    cube = Entity("blue_cube", "box", (0.02, 0.02, 0.02), (0.55, 0.05, 0.18), 0.0)
    kfs = [Keyframe((0.5, 0.0, 0.1), 0.0, "open", None, "stop"),
           Keyframe((0.5, 0.0, 0.1), 90.0, "open", None, "turn in place")]
    marks = checks(kfs, [cube])
    assert marks[0]["checks"]["collision"] == "ok"
    assert "blue_cube" in marks[1]["checks"]["collision"]


def test_marks_say_what_the_true_scene_holds():
    marks = checks(pick(CUBE), [CUBE, BLUE])
    assert [m["holds"] for m in marks] == [None, "red_cube", "red_cube"]
    missed = pick(CUBE)
    missed[1] = Keyframe((0.5, -0.06, 0.02), 0.0, "closed", "red_cube")  # 4 cm off: the grasp fails
    assert [m["holds"] for m in checks(missed, [CUBE, BLUE])] == [None, None, None]


# ---- v1.11: a predicted move allows contact on its own keyframe only ----
def test_the_collision_check_ignores_a_moved_object_only_on_that_keyframe():
    from armmind.plan.formats import Move
    down = [Keyframe((0.5, 0.03, 0.15), 0.0, "closed", None), Keyframe((0.5, 0.03, 0.02), 0.0, "closed", None)]
    push = Move("blue cube", (0.5, 0.15, 0.02))
    kfs = [*down, Keyframe((0.5, 0.12, 0.02), 0.0, "closed", None, "push", (push,)),
           Keyframe((0.5, 0.03, 0.02), 0.0, "closed", None, "back"),
           Keyframe((0.5, 0.16, 0.02), 0.0, "closed", None, "push again, with no move")]
    marks = checks(kfs, [CUBE, BLUE])
    assert [m["checks"]["collision"] == "ok" for m in marks] == [True, True, True, True, False]
    assert "blue_cube" in marks[4]["checks"]["collision"]
    assert marks[3]["ok"]  # the true cube now stands where the push leaves it
    other = Move("red cube", (0.5, -0.05, 0.02))  # a move of another object does not excuse the blue cube
    marks = checks([*down, Keyframe((0.5, 0.12, 0.02), 0.0, "closed", None, "push", (other,))], [CUBE, BLUE])
    assert "blue_cube" in marks[2]["checks"]["collision"]


def test_a_moved_true_object_shifts_by_the_predicted_displacement():
    from armmind.plan.formats import Move
    belief = [Entity("blue cube", "box", (0.04, 0.04, 0.04), (0.5, 0.08, 0.02))]  # 2 cm short of the truth
    kfs = [Keyframe((0.5, 0.01, 0.15), 0.0, "closed", None), Keyframe((0.5, 0.01, 0.02), 0.0, "closed", None),
           Keyframe((0.5, 0.10, 0.02), 0.0, "closed", None, "push", (Move("blue cube", (0.5, 0.135, 0.02)),)),
           Keyframe((0.5, 0.10, 0.10), 0.0, "closed", None, "up")]
    marks = plan_checks(START, kfs, [CUBE, BLUE], ALIASES, set(), PARTS, no_reach_limit, open_state="open",
                        gap=arm_gap, belief=belief)
    assert all(m["ok"] for m in marks)  # the true cube went 5.5 cm, to y 0.155: the hand leaves it freely
    assert marks[2]["moves"][0]["xyz"][1] == pytest.approx(0.155)
    down = Keyframe((0.5, 0.155, 0.02), 0.0, "closed", None, "down onto the moved cube")
    marks = plan_checks(START, [*kfs, Keyframe((0.5, 0.155, 0.10), 0.0, "closed", None), down], [CUBE, BLUE],
                        ALIASES, set(), PARTS, no_reach_limit, open_state="open", gap=arm_gap, belief=belief)
    assert "blue_cube" in marks[5]["checks"]["collision"]


# ---- v1.11: the prediction score after Execute ----
def _kf(xyz, t, moves=(), carrying=None):
    return {"xyz": list(xyz), "yaw": 0.0, "state": "closed", "carrying": carrying, "gap": 0.0, "changes": False,
            "t": t, "t_done": t + 0.5, "moves": list(moves)}


PUSH = {"name": "cube", "xyz": [0.5, 0.05, 0.02], "yaw": 0.0}
PUSH_TRACK = {"start": {"xyz": [0.5, -0.06, 0.02], "yaw": 0.0, "state": "closed", "gap": 0.0},
              "keyframes": [_kf((0.5, 0.02, 0.02), 1.0, [PUSH]), _kf((0.5, 0.02, 0.15), 2.0)], "duration": 2.5}
BELIEVED = [{"name": "cube", "xyz": [0.5, 0.0, 0.02], "yaw": 0.0}, {"name": "ball", "xyz": [0.6, 0.2, 0.02], "yaw": 0.0}]
MATCH = {"red_cube": "cube", "ball_1": "ball"}


def _true(cube, ball=(0.6, 0.2, 0.02)):
    return {"red_cube": {"xyz": list(cube), "yaw": 0.0}, "ball_1": {"xyz": list(ball), "yaw": 0.0}}


def _score(true_start, true_after, track=PUSH_TRACK, believed=BELIEVED):
    from armmind.plan.scores import prediction
    return prediction(track, believed, true_start, true_after, MATCH)


def test_a_perfect_prediction_has_no_error():
    s = _score(_true((0.5, 0.0, 0.02)), [_true((0.5, 0.05, 0.02))] * 2)
    rows = {r["name"]: r for r in s["rows"]}
    assert rows["cube"]["disp_cm"] == 0 and rows["cube"]["final_xy_cm"] == 0 and rows["cube"]["final_z_cm"] == 0
    assert rows["cube"]["predicted_cm"] == [0.0, 5.0, 0.0] and rows["cube"]["real_cm"] == [0.0, 5.0, 0.0]
    assert rows["cube"]["predicted_moved"] and rows["cube"]["real_moved"] and rows["ball"]["agree"]
    assert s["summary"]["agree"] == 2 and s["summary"]["count"] == 2 and s["summary"]["max_disp_cm"] == 0
    assert s["keyframes"] == [{"keyframe": 1, "name": "cube", "true_name": "red_cube", "disp_cm": 0.0}]
    assert s["summary"]["ran"] == 2 and s["summary"]["partial"] is False


def test_a_push_5_cm_wrong_gives_5_cm():
    s = _score(_true((0.5, 0.0, 0.02)), [_true((0.5, 0.10, 0.02))] * 2)
    cube = s["rows"][0]
    assert cube["disp_cm"] == pytest.approx(5.0) and cube["final_xy_cm"] == pytest.approx(5.0)
    assert s["keyframes"][0]["disp_cm"] == pytest.approx(5.0) and s["summary"]["max_disp_cm"] == pytest.approx(5.0)


def test_a_perception_offset_does_not_change_the_displacement_error():
    s = _score(_true((0.53, -0.02, 0.02)), [_true((0.53, 0.03, 0.02))] * 2)  # 3.6 cm off, the same 5 cm push
    cube = s["rows"][0]
    assert cube["disp_cm"] == 0 and cube["final_xy_cm"] == pytest.approx(3.61)


def test_a_move_of_an_object_that_stays_still_is_a_disagreement():
    track = {**PUSH_TRACK, "keyframes": [_kf((0.5, 0.02, 0.02), 1.0, [PUSH, {"name": "ball", "xyz": [0.6, 0.25, 0.02],
                                                                           "yaw": 0.0}]), PUSH_TRACK["keyframes"][1]]}
    s = _score(_true((0.5, 0.0, 0.02)), [_true((0.5, 0.05, 0.02))] * 2, track=track)
    ball = next(r for r in s["rows"] if r["name"] == "ball")
    assert ball["predicted_moved"] and not ball["real_moved"] and not ball["agree"]
    assert ball["disp_cm"] == pytest.approx(5.0) and s["summary"]["agree"] == 1 and s["summary"]["count"] == 2
    assert [(k["keyframe"], k["name"]) for k in s["keyframes"]] == [(1, "cube"), (1, "ball")]


def test_an_early_stop_compares_only_the_keyframes_that_ran():
    track = {**PUSH_TRACK, "keyframes": [PUSH_TRACK["keyframes"][0], _kf((0.5, 0.02, 0.15), 2.0, [
        {"name": "ball", "xyz": [0.4, 0.2, 0.02], "yaw": 0.0}])]}
    s = _score(_true((0.5, 0.0, 0.02)), [_true((0.5, 0.05, 0.02))], track=track)
    assert s["summary"]["ran"] == 1 and s["summary"]["of"] == 2 and s["summary"]["partial"] is True
    ball = next(r for r in s["rows"] if r["name"] == "ball")
    assert ball["predicted_cm"] == [0.0, 0.0, 0.0] and ball["agree"]  # the move of keyframe 2 never ran
    assert [k["keyframe"] for k in s["keyframes"]] == [1]
    none = _score(_true((0.5, 0.0, 0.02)), [], track=track)
    assert none["rows"] == [] and none["summary"]["ran"] == 0 and none["summary"]["partial"] is True


# ---- v1.11 fix 1: a moved true object settles with drop(), and the agent must clear it ----
RED_BELIEF = [Entity("red cube", "box", (0.04, 0.04, 0.04), (0.5, -0.1, 0.02)),
              Entity("blue cube", "box", (0.04, 0.04, 0.04), (0.5, 0.1, 0.02))]


def _belief_checks(kfs):
    return plan_checks(START, kfs, [CUBE, BLUE], ALIASES, set(), PARTS, no_reach_limit, open_state="open",
                       gap=arm_gap, belief=RED_BELIEF)


def test_a_moved_object_never_sinks_under_the_table():
    from armmind.plan.formats import Move
    kfs = [*pick(CUBE), Keyframe((0.5, 0.25, 0.2), 0.0, "closed", "red cube"),
           Keyframe((0.5, 0.25, 0.08), 0.0, "open", None, "let go 6 cm up"),
           Keyframe((0.5, 0.25, 0.2), 0.0, "closed", None, "up, close"),
           Keyframe((0.6, 0.3, 0.2), 0.0, "closed", None, "aside", (Move("red cube", (0.5, 0.25, 0.02)),)),
           Keyframe((0.5, 0.275, 0.2), 0.0, "closed", None, "above, beside the cube"),
           Keyframe((0.5, 0.275, 0.02), 0.0, "closed", None, "down beside it: a finger hits it")]
    marks = _belief_checks(kfs)
    assert marks[6]["moves"] == [{"name": "red_cube", "xyz": [0.5, 0.25, pytest.approx(0.02)], "yaw": 0.0}]
    assert "red_cube" in marks[8]["checks"]["collision"]  # the cube stands on the table, not 4 cm under it
    assert marks[7]["predicted"] == ["red cube"] and marks[6]["predicted"] == []


def test_the_agent_must_not_overlap_a_moved_object_at_the_end():
    from armmind.plan.formats import Move
    down = [Keyframe((0.5, 0.03, 0.15), 0.0, "closed", None), Keyframe((0.5, 0.03, 0.02), 0.0, "closed", None)]
    short = Keyframe((0.5, 0.12, 0.02), 0.0, "closed", None, "push", (Move("blue cube", (0.5, 0.13, 0.02)),))
    marks = _belief_checks([*down, short])
    assert "overlaps blue cube at its predicted pose" in marks[2]["checks"]["collision"]
    clear = Keyframe((0.5, 0.12, 0.02), 0.0, "closed", None, "push", (Move("blue cube", (0.5, 0.155, 0.02)),))
    assert _belief_checks([*down, clear])[2]["ok"]


def test_moved_or_not_looks_at_xy_only():
    lift = {**PUSH_TRACK, "keyframes": [_kf((0.5, 0.0, 0.02), 1.0, carrying="cube"),
                                        _kf((0.5, 0.0, 0.2), 2.0, carrying="cube")]}
    s = _score(_true((0.5, 0.0, 0.02)), [_true((0.5, 0.0, 0.02)), _true((0.5, 0.0, 0.2))], track=lift)
    cube = s["rows"][0]
    assert cube["predicted_cm"][2] == pytest.approx(18.0) and not cube["predicted_moved"] and not cube["real_moved"]


def test_a_true_object_outside_the_belief_that_moved_is_a_disagreement():
    from armmind.plan.scores import prediction
    start = {**_true((0.5, 0.0, 0.02)), "can": {"xyz": [0.5, 0.1, 0.03], "yaw": 0.0},
             "mug": {"xyz": [0.7, 0.1, 0.03], "yaw": 0.0}}
    after = {**_true((0.5, 0.05, 0.02)), "can": {"xyz": [0.5, 0.14, 0.03], "yaw": 0.0},
             "mug": {"xyz": [0.7, 0.11, 0.03], "yaw": 0.0}}  # the can went 4 cm, the mug only 1 cm
    s = prediction(PUSH_TRACK, BELIEVED, start, [after, after], MATCH)
    assert s["unmatched"] == [{"true_name": "can", "moved_cm": 4.0}]
    assert s["summary"]["agree"] == 2 and s["summary"]["count"] == 3


def test_a_true_object_that_the_shift_puts_in_the_fingers_goes_further_along_the_push():
    from armmind.plan.formats import Move
    belief = [Entity("blue cube", "box", (0.04, 0.04, 0.04), (0.5, 0.104, 0.02))]  # the truth is 4 mm nearer
    down = [Keyframe((0.5, 0.03, 0.15), 0.0, "closed", None), Keyframe((0.5, 0.03, 0.02), 0.0, "closed", None)]
    push = Keyframe((0.5, 0.12, 0.02), 0.0, "closed", None, "push", (Move("blue cube", (0.5, 0.1505, 0.02)),))
    up = Keyframe((0.5, 0.12, 0.15), 0.0, "closed", None, "up")
    marks = plan_checks(START, [*down, push, up], [CUBE, BLUE], ALIASES, set(), PARTS, no_reach_limit,
                        open_state="open", gap=arm_gap, belief=belief)
    assert all(m["ok"] for m in marks), [m["checks"] for m in marks]
    y = marks[2]["moves"][0]["xyz"][1]  # the plain shift gives 0.1465: 4 mm inside the finger face at 0.1305
    assert y - 0.02 == pytest.approx(0.1305 + 0.001, abs=0.0011)  # it touches the finger with a 1 mm gap


def test_a_move_with_no_displacement_clears_the_agent_away_from_its_centre():
    from armmind.plan.formats import Move
    belief = [Entity("blue cube", "box", (0.04, 0.04, 0.04), (0.5, 0.15, 0.02))]  # believed 5 cm off the truth
    kfs = [Keyframe((0.5, 0.06, 0.15), 0.0, "closed", None), Keyframe((0.5, 0.06, 0.02), 0.0, "closed", None),
           Keyframe((0.5, 0.075, 0.02), 0.0, "closed", None, "touch", (Move("blue cube", (0.5, 0.15, 0.02)),))]
    marks = plan_checks(START, kfs, [CUBE, BLUE], ALIASES, set(), PARTS, no_reach_limit, open_state="open",
                        gap=arm_gap, belief=belief)
    assert marks[2]["moves"][0]["xyz"][1] - 0.02 >= 0.075 + 0.0105  # pushed from the hand centre outward
