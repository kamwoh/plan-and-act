"""The loop stops an action list at the first surprise."""
import pytest

from armmind.agent.loop import Agent, AgentConfig
from armmind.agent.parse import ReplyError, parse_reply
from armmind.agent.prompts import REPLY_SCHEMA, SYSTEM, StepSummary, build_user_text
from armmind.llm.fake import FakeBackend
from armmind.types import Action, Observation
from tests.fakes import Recorder, ScriptEnv, direct, good

MOVE = {"type": "move_to", "xyz": [0.5, 0.0, 0.1]}
OPEN, CLOSE = {"type": "open"}, {"type": "close"}


async def run_one(actions, outcomes, steps=1, more=()):
    env, rec = ScriptEnv(outcomes), Recorder()
    backend = FakeBackend([good(actions), *more])
    await Agent(backend, AgentConfig(max_steps=steps)).run(env, direct, rec)
    return env, [d for k, d in rec.events if k == "step"], backend


def test_schema_and_system_know_expect():
    props = REPLY_SCHEMA["properties"]["actions"]["items"]["properties"]
    assert props["expect"] == {"type": "object", "additionalProperties": False, "required": ["holding"],
                               "properties": {"holding": {"type": "string"}}}
    assert '"expect": {"holding": "<object>"}' in SYSTEM and "first failure" in SYSTEM


def test_parse_reads_expect():
    r = parse_reply({"observation": "", "plan": "", "todo": [], "actions": [{**CLOSE, "expect": {"holding": "Red cube"}}]})
    assert r.actions[0] == Action("close", expect="Red cube")


@pytest.mark.parametrize("expect", ["red", {"holding": 3}, {}, {"holding": "a", "x": 1}, None])
def test_parse_rejects_a_bad_expect(expect):
    with pytest.raises(ReplyError, match="expect"):
        parse_reply({"observation": "", "plan": "", "todo": [], "actions": [{**CLOSE, "expect": expect}]})


async def test_no_stop_when_all_goes_as_planned():
    env, (step,), _ = await run_one([OPEN, MOVE, CLOSE, MOVE], [{}, {}, {"gripper_width": 0.04, "contacts": ("red_cube",),
                                                                       "held_after": ("red_cube",)},
                                                                  {"gripper_width": 0.04, "held_before": ("red_cube",),
                                                                   "held_after": ("red_cube",)}])
    assert len(env.acts) == 4 and step["stopped"] is None
    assert "skipped" not in "; ".join(step["results"])


@pytest.mark.parametrize("status", ["blocked", "unreachable"])
async def test_rule_1_a_failed_move_stops_the_list(status):
    env, (step,), _ = await run_one([MOVE, CLOSE, MOVE], [{"status": status}])
    assert len(env.acts) == 1
    assert step["stopped"] == {"after": 1, "of": 3, "reason": f"the move to (0.500, 0.000, 0.100) was {status}"}
    assert step["results"][1:] == ["close: skipped", "move_to(0.500, 0.000, 0.100): skipped"]


async def test_rule_2_a_close_on_nothing_stops_the_list():
    env, (step,), _ = await run_one([CLOSE, MOVE], [{"gripper_width": 0.001}])
    assert len(env.acts) == 1 and step["stopped"]["reason"] == "the gripper closed on nothing"


async def test_rule_2_does_not_apply_with_expect_nothing():
    env, (step,), _ = await run_one([{**CLOSE, "expect": {"holding": "nothing"}}, MOVE], [{"gripper_width": 0.001}])
    assert len(env.acts) == 2 and step["stopped"] is None


async def test_rule_3_a_dropped_object_stops_the_list():
    env, (step,), _ = await run_one([MOVE, MOVE], [{"held_before": ("red_cube",), "held_after": ()}])
    assert len(env.acts) == 1 and step["stopped"]["reason"] == "dropped red_cube"
    assert "dropped red_cube" in step["results"][0]


async def test_rule_3_does_not_apply_to_open():
    env, (step,), _ = await run_one([OPEN, MOVE], [{"held_before": ("red_cube",), "held_after": ()}])
    assert len(env.acts) == 2 and step["stopped"] is None


async def test_rule_4_expect_mismatch_stops_the_list():
    env, (step,), _ = await run_one([{**CLOSE, "expect": {"holding": "blue cube"}}, MOVE],
                                    [{"gripper_width": 0.04, "contacts": ("red_cube",), "held_after": ("red_cube",)}])
    assert len(env.acts) == 1 and step["stopped"]["reason"] == "expected to hold blue_cube, holds red_cube"


@pytest.mark.parametrize("name", ["red_cube", "RED CUBE", "red cube, 4 cm", "Red_Cube"])
async def test_expect_matches_name_or_label_without_case(name):
    env, (step,), _ = await run_one([{**CLOSE, "expect": {"holding": name}}, MOVE],
                                    [{"gripper_width": 0.04, "contacts": ("red_cube",), "held_after": ("red_cube",)}])
    assert len(env.acts) == 2 and step["stopped"] is None


async def test_expect_nothing_mismatch():
    _, (step,), _ = await run_one([{**MOVE, "expect": {"holding": "nothing"}}, MOVE],
                                    [{"held_before": ("red_cube",), "held_after": ("red_cube",)}])
    assert step["stopped"]["reason"] == "expected to hold nothing, holds red_cube"


async def test_unknown_expect_name_gives_a_note_and_no_stop():
    env, (step,), _ = await run_one([{**MOVE, "expect": {"holding": "unicorn"}}, MOVE], [{}])
    assert len(env.acts) == 2 and step["stopped"] is None
    assert "unknown object 'unicorn' in expect, ignored" in step["results"][0]


async def test_stop_reaches_the_history_and_the_next_message():
    _, steps, backend = await run_one([CLOSE, MOVE, MOVE], [{"gripper_width": 0.001}], steps=2, more=[good()])
    text = backend.sent[1][0].text
    assert "Stopped after action 1 of 3: the gripper closed on nothing." in text
    lines = text.splitlines()
    i = next(n for n, x in enumerate(lines) if x.startswith("Gripper touching"))
    assert lines[i + 1] == ("Your last action list stopped early: the gripper closed on nothing. "
                            "Look at the images and plan again. Write a lesson if it helps.")
    assert "stopped early" not in backend.sent[0][0].text
    assert steps[1]["stopped"] is None


def test_history_line_shows_the_stop():
    obs = Observation(b"", b"", (0.3, 0.0, 0.4), 0.08, ())
    text = build_user_text("s", "t", 2, 30, obs, "", (), [StepSummary(1, "go", ("r1",), "Stopped after action 1 of 2: x.")])
    assert "- step 1: go -> r1. Stopped after action 1 of 2: x." in text


# ---- A real world ----

def _real_run(scene, actions):
    from armmind.env import MujocoEnv
    from armmind.scenes import SCENES
    env, rec = MujocoEnv(SCENES[scene], None, 0, "t"), Recorder()
    try:
        import asyncio
        asyncio.run(Agent(FakeBackend([good(actions)]), AgentConfig(max_steps=1)).run(env, direct, rec))
    finally:
        env.close()
    return next(d for k, d in rec.events if k == "step")


@pytest.mark.assets
def test_real_close_far_from_objects_stops():
    step = _real_run("buttons", [{"type": "move_to", "xyz": [0.35, -0.3, 0.1]}, CLOSE, MOVE])
    assert step["stopped"] == {"after": 2, "of": 3, "reason": "the gripper closed on nothing"}


@pytest.mark.assets
def test_real_move_into_the_panel_stops():
    step = _real_run("buttons", [CLOSE | {"expect": {"holding": "nothing"}},
                                 {"type": "move_to", "xyz": [0.62, 0.05, 0.01]}, MOVE])
    assert step["stopped"]["after"] == 2 and step["stopped"]["reason"].endswith("was blocked")


@pytest.mark.assets
def test_real_button_press_with_expect_nothing_runs_without_a_stop():
    from armmind.scenes.buttons import press
    moves = [{"type": "move_to", "xyz": list(a.xyz)} for a in press("green")]
    step = _real_run("buttons", [{**CLOSE, "expect": {"holding": "nothing"}}, *moves])
    assert step["stopped"] is None and len(step["results"]) == 4


# ---- What counts as holding, in a real world ----

def _run_actions(scene_name, task_name, actions=None, seed=0):
    """Run actions (or the task solution) through check_result. Gives (results, stop reasons)."""
    from armmind.agent.loop import check_result
    from armmind.scenes import SCENES
    from armmind.sim.arm import Arm
    from armmind.sim.world import World
    scene = SCENES[scene_name]
    task = scene.tasks[task_name] if task_name else None
    w = World(scene)
    w.reset(scene.layout(seed), task.initial_qpos if task else None)
    arm = Arm(w)
    arm.reset_ready()
    labels = {o.name: o.label for o in scene.objects}
    results, stops = [], []
    for a in actions if actions is not None else task.solution(w):
        r, why = check_result(a, arm.act(a), labels)
        results.append(r)
        if why:
            stops.append((a.describe(), why))
    return results, stops


@pytest.mark.assets
@pytest.mark.parametrize("side", [(0, -1), (-1, 0)])
def test_real_push_with_the_closed_gripper_is_not_holding(side):
    from armmind.scenes import SCENES
    bx, by, _ = SCENES["cubes"].layout(0)["blue_cube"]
    dx, dy = side
    seq = [Action("close", expect="nothing"), Action("move_to", (bx + dx * 0.08, by + dy * 0.08, 0.15)),
           Action("move_to", (bx + dx * 0.08, by + dy * 0.08, 0.02)),
           Action("move_to", (bx + dx * 0.01, by + dy * 0.01, 0.02)),  # push into the cube
           Action("move_to", (bx + dx * 0.01, by + dy * 0.01, 0.15))]
    results, stops = _run_actions("cubes", None, seq)
    assert stops == []
    assert all(r.held_after == () for r in results)
    assert all("now holding" not in r.summary() for r in results)


@pytest.mark.assets
@pytest.mark.parametrize(("scene", "task", "obj"), [("cubes", "pick", "red_cube"), ("kitchen", "bottle_in_bowl", "bottle"),
                                                    ("toys", "panda_in_tray", "panda")])
def test_real_grasp_counts_as_held_and_the_solution_has_no_stop(scene, task, obj):
    results, stops = _run_actions(scene, task)
    assert stops == []
    first_close = next(r for r in results if r.action.type == "close")
    assert first_close.held_after == (obj,)
