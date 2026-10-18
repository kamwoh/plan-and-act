"""The plan reply: parse_plan, the plan schema and the plan system prompt."""
import pytest

from armmind.agent.parse import ReplyError, parse_plan
from armmind.agent.prompts import PLAN_REPLY_SCHEMA, PLAN_SYSTEM, SYSTEM
from armmind.plan.formats import Entity, Keyframe, Move


def obj(**kw):
    return {"name": "red_cube", "shape": "box", "size": [0.04, 0.04, 0.04], "xyz": [0.5, -0.1, 0.02], "yaw": 0,
            **kw}


def kf(**kw):
    return {"xyz": [0.5, -0.1, 0.15], "yaw": 0, "gripper": "open", "holding": "nothing", "note": "above", **kw}


def plan(objects=None, keyframes=None, **kw):
    return {"objects": [obj()] if objects is None else objects,
            "keyframes": [kf()] if keyframes is None else keyframes, "summary": "pick it", **kw}


def test_good_plan_maps_to_entities_and_keyframes():
    p = parse_plan(plan(keyframes=[kf(), kf(xyz=[0.5, -0.1, 0.02], gripper="closed", holding="red cube",
                                          note="grasp")]))
    assert p.objects == (Entity("red_cube", "box", (0.04, 0.04, 0.04), (0.5, -0.1, 0.02), 0.0),)
    assert p.keyframes[0] == Keyframe((0.5, -0.1, 0.15), 0.0, "open", None, "above")
    assert p.keyframes[1].state == "closed" and p.keyframes[1].carrying == "red cube"
    assert p.summary == "pick it" and p.warnings == ()


def test_nothing_is_no_carried_entity_in_any_case():
    p = parse_plan(plan(keyframes=[kf(holding=" Nothing ")]))
    assert p.keyframes[0].carrying is None


def test_cylinder_and_no_objects_are_fine():
    assert parse_plan(plan(objects=[obj(shape="cylinder")])).objects[0].shape == "cylinder"
    assert parse_plan(plan(objects=[])).objects == ()


def test_yaw_is_clamped_as_move_to_does_with_a_note():
    p = parse_plan(plan(keyframes=[kf(yaw=120), kf(yaw=-95.5)]))
    assert [k.yaw for k in p.keyframes] == [90.0, -90.0]
    assert p.warnings == ("keyframe 1: yaw clamped to 90", "keyframe 2: yaw clamped to -90")


def test_long_text_is_cut():
    p = parse_plan(plan(keyframes=[kf(note="n" * 500)], summary="s" * 3000))
    assert len(p.keyframes[0].note) == 200 and len(p.summary) == 1000


@pytest.mark.parametrize("bad, word", [
    ("not a dict", "object"),
    (plan(objects="x"), "objects"),
    (plan(objects=[obj()] * 13), "12"),
    (plan(objects=[obj(name="")]), "name"),
    (plan(objects=[obj(name=3)]), "name"),
    (plan(objects=[obj(shape="sphere")]), "shape"),
    (plan(objects=[obj(size=[0.04, 0.04])]), "size"),
    (plan(objects=[obj(size=[0.04, 0.04, -0.01])]), "size"),
    (plan(objects=[obj(size=[0.04, 0.04, 0])]), "size"),
    (plan(objects=[obj(xyz=[0.5, "a", 0.02])]), "xyz"),
    (plan(objects=[obj(xyz=[0.5, float("nan"), 0.02])]), "xyz"),
    (plan(objects=[obj(yaw="0")]), "yaw"),
    (plan(objects=[obj(yaw=True)]), "yaw"),
    (plan(objects=["x"]), "object 1"),
    (plan(keyframes=[]), "1 to 40"),
    (plan(keyframes=[kf()] * 41), "1 to 40"),
    (plan(keyframes="x"), "keyframes"),
    (plan(keyframes=["x"]), "keyframe 1"),
    (plan(keyframes=[kf(xyz=[0.5, 0.1])]), "xyz"),
    (plan(keyframes=[kf(xyz=[0.5, 0.1, float("inf")])]), "xyz"),
    (plan(keyframes=[kf(yaw=None)]), "yaw"),
    (plan(keyframes=[kf(gripper="half")]), "gripper"),
    (plan(keyframes=[kf(holding=1)]), "holding"),
    (plan(keyframes=[kf(note=None)]), "note"),
    (plan(summary=5), "summary"),
])
def test_bad_fields_raise(bad, word):
    with pytest.raises(ReplyError, match=word):
        parse_plan(bad)


def test_missing_fields_raise():
    for key in ("objects", "keyframes", "summary"):
        body = plan()
        del body[key]
        with pytest.raises(ReplyError, match=key):
            parse_plan(body)
    body = plan(keyframes=[{k: v for k, v in kf().items() if k != "gripper"}])
    with pytest.raises(ReplyError, match="gripper"):
        parse_plan(body)


def test_schema_is_strict_and_matches_the_limits():
    s = PLAN_REPLY_SCHEMA
    assert s["additionalProperties"] is False and set(s["required"]) == {"objects", "keyframes", "summary"}
    o, k = s["properties"]["objects"], s["properties"]["keyframes"]
    assert o["maxItems"] == 12 and "minItems" not in k and k["maxItems"] == 40  # a done re-plan can have 0
    assert o["items"]["additionalProperties"] is False and k["items"]["additionalProperties"] is False
    assert set(o["items"]["required"]) == {"name", "shape", "size", "xyz", "yaw"}
    assert set(k["items"]["required"]) == {"xyz", "yaw", "gripper", "holding", "note"}
    assert o["items"]["properties"]["shape"]["enum"] == ["box", "cylinder"]
    assert k["items"]["properties"]["gripper"]["enum"] == ["open", "closed"]


def test_plan_system_shares_the_facts_of_system():
    for fact in ("The robot base is at x=0, y=0. The table surface is at z=0.",
                 "+y points to the right in both images.",
                 "The gripper position is the point between the two fingertips.",
                 "The arm can reach x from 0.25 to 0.75, y from -0.40 to 0.40, z from 0.01 to 0.50.",
                 "at 0 the fingers close along the x axis",
                 "The gripper is 0.08 m wide when open."):
        assert fact in SYSTEM and fact in PLAN_SYSTEM
    assert "keyframes" in PLAN_SYSTEM and "keyframes" not in SYSTEM
    assert "Use at most 8 actions" not in PLAN_SYSTEM


@pytest.mark.parametrize("names", [["red_cube", "red_cube"], ["red cube", "Red_Cube"]])
def test_duplicate_object_names_raise(names):
    with pytest.raises(ReplyError, match="twice"):
        parse_plan(plan(objects=[obj(name=n) for n in names]))


def test_plan_system_has_no_step_message():
    assert "step message" not in PLAN_SYSTEM
    assert "the step message lists them with their sizes" in SYSTEM


# ---- v1.11: the predicted "moves" of a keyframe ----
def two_objects():
    return [obj(), obj(name="blue cube", xyz=[0.55, 0.1, 0.02])]


def mv(**kw):
    return {"object": "Blue_Cube", "xyz": [0.55, 0.15, 0.02], "yaw": 0, **kw}


def test_a_good_move_maps_to_the_believed_name():
    p = parse_plan(plan(objects=two_objects(), keyframes=[kf(moves=[mv(yaw=10)]), kf()]))
    assert p.keyframes[0].moves == (Move("blue cube", (0.55, 0.15, 0.02), 10.0),)
    assert p.keyframes[1].moves == ()
    assert p.keyframes[0].to_dict()["moves"] == [{"name": "blue cube", "xyz": [0.55, 0.15, 0.02], "yaw": 10.0}]
    assert Keyframe.from_dict(p.keyframes[0].to_dict()) == p.keyframes[0]


@pytest.mark.parametrize("yaw, want", [(190, -170.0), (-180, 180.0), (540, 180.0), (-200, 160.0), (45, 45.0)])
def test_the_move_yaw_wraps_to_plus_minus_180(yaw, want):
    p = parse_plan(plan(objects=two_objects(), keyframes=[kf(moves=[mv(yaw=yaw)])]))
    assert p.keyframes[0].moves[0].yaw == pytest.approx(want)


@pytest.mark.parametrize("moves, word", [
    ([mv(object="green cube")], "not one of the 'objects'"),
    ([mv(), mv(object="blue cube")], "twice"),
    ([mv()] * 5, "at most 4"),
    ("x", "must be a list"),
    (["x"], "each move must be"),
    ([{"object": "blue cube", "xyz": [0.5, 0.1, 0.02]}], "each move must be"),
    ([mv(extra=1)], "each move must be"),
    ([mv(object=3)], "'object' must be a string"),
    ([mv(xyz=[0.5, 0.1])], "xyz"),
    ([mv(xyz=[0.5, "a", 0.1])], "xyz"),
    ([mv(yaw="0")], "yaw"),
    ([mv(yaw=True)], "yaw"),
])
def test_bad_moves_raise(moves, word):
    with pytest.raises(ReplyError, match=word):
        parse_plan(plan(objects=two_objects(), keyframes=[kf(moves=moves)]))


def test_a_move_of_the_held_object_raises():
    held = kf(gripper="closed", holding="blue cube", moves=[mv()])
    with pytest.raises(ReplyError, match="holds"):
        parse_plan(plan(objects=two_objects(), keyframes=[held]))
    other = kf(gripper="closed", holding="red cube", moves=[mv()])
    assert parse_plan(plan(objects=two_objects(), keyframes=[other])).keyframes[0].moves[0].name == "blue cube"


def test_the_schema_has_optional_strict_moves():
    k = PLAN_REPLY_SCHEMA["properties"]["keyframes"]["items"]
    m = k["properties"]["moves"]
    assert "moves" not in k["required"] and m["maxItems"] == 4
    assert m["items"]["additionalProperties"] is False and set(m["items"]["required"]) == {"object", "xyz", "yaw"}


def test_system_is_unchanged_and_plan_system_tells_about_moves():
    import hashlib
    assert hashlib.sha256(SYSTEM.encode()).hexdigest() == \
        "9dfe9f6753b2b90d980e24c17d1d5f13f595fbcfb7787926f2b0b329c5739792"
    assert '"moves"' in PLAN_SYSTEM and "pushes, slides or knocks" in PLAN_SYSTEM
    assert "end of the keyframe" in PLAN_SYSTEM and "\"moves\"" not in SYSTEM


def test_a_move_of_the_object_released_on_the_same_keyframe_raises():
    held = kf(gripper="closed", holding="blue cube")
    let_go = kf(gripper="open", holding="nothing", moves=[mv()])
    with pytest.raises(ReplyError, match="falls by itself: move it in a later keyframe"):
        parse_plan(plan(objects=two_objects(), keyframes=[held, let_go]))
    later = parse_plan(plan(objects=two_objects(), keyframes=[held, kf(), kf(moves=[mv()])]))
    assert later.keyframes[2].moves[0].name == "blue cube"
    swap = kf(gripper="closed", holding="red cube", moves=[mv()])  # it lets go of the blue cube, holds the red one
    with pytest.raises(ReplyError, match="falls by itself"):
        parse_plan(plan(objects=two_objects(), keyframes=[held, swap]))
