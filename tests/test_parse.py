import pytest

from plan_and_act.agent.parse import ReplyError, parse_reply
from plan_and_act.types import Action


def reply(actions, **kw):
    return {"observation": "o", "plan": "p", "actions": actions, "todo": ["a"], **kw}


def test_parse_valid_reply():
    r = parse_reply(reply([{"type": "move_to", "xyz": [0.5, -0.1, 0.15]}, {"type": "close"},
                           {"type": "wait", "seconds": 1}], notes="n"))
    assert r.actions == (Action("move_to", (0.5, -0.1, 0.15)), Action("close"), Action("wait", seconds=1.0))
    assert r.notes == "n" and r.todo == ("a",) and not r.truncated


def test_parse_notes_absent_is_none():
    assert parse_reply(reply([])).notes is None


def test_parse_truncates_long_plans():
    r = parse_reply(reply([{"type": "open"}] * 11))
    assert len(r.actions) == 8 and r.truncated


@pytest.mark.parametrize("xyz", [None, [0.5, 0.1], [0.5, "a", 0.1], [0.5, float("nan"), 0.1], [True, 0, 0], [0.5, 10**400, 0.1]])
def test_parse_rejects_bad_move_to(xyz):
    with pytest.raises(ReplyError, match="xyz"):
        parse_reply(reply([{"type": "move_to", "xyz": xyz}]))


def test_parse_rejects_unknown_type():
    with pytest.raises(ReplyError, match="unknown type"):
        parse_reply(reply([{"type": "jump"}]))


def test_parse_clamps_wait():
    assert parse_reply(reply([{"type": "wait", "seconds": 99}])).actions[0].seconds == 3.0
    assert parse_reply(reply([{"type": "wait", "seconds": -1}])).actions[0].seconds == 0.0


def test_parse_rejects_non_list_actions():
    with pytest.raises(ReplyError, match="actions"):
        parse_reply({"observation": "o", "plan": "p", "actions": "move"})


def test_parse_done_flag():
    assert parse_reply(reply([], done=True)).done is True
    assert parse_reply(reply([])).done is False


@pytest.mark.parametrize("done", ["yes", 1, None])
def test_parse_rejects_non_bool_done(done):
    with pytest.raises(ReplyError, match="done"):
        parse_reply(reply([], done=done))


def test_parse_move_to_with_yaw():
    r = parse_reply(reply([{"type": "move_to", "xyz": [0.5, 0, 0.1], "yaw": 30}, {"type": "move_to", "xyz": [0.5, 0, 0.1]}]))
    assert r.actions == (Action("move_to", (0.5, 0.0, 0.1), yaw=30.0), Action("move_to", (0.5, 0.0, 0.1)))
    assert r.actions[0].describe() == "move_to(0.500, 0.000, 0.100, yaw 30°)"
    assert r.actions[1].describe() == "move_to(0.500, 0.000, 0.100)"


@pytest.mark.parametrize("yaw", ["a", float("nan"), float("inf"), True, None, [1]])
def test_parse_rejects_bad_yaw(yaw):
    with pytest.raises(ReplyError, match="yaw"):
        parse_reply(reply([{"type": "move_to", "xyz": [0.5, 0, 0.1], "yaw": yaw}]))


def test_parse_lessons_are_stripped_cut_and_optional():
    assert parse_reply(reply([])).lessons == ()
    r = parse_reply(reply([], lessons=["  Closing 9 cm up slipped. 6 cm up held.  ", "   ", "x" * 300]))
    assert r.lessons == ("Closing 9 cm up slipped. 6 cm up held.", "x" * 200)


def test_parse_keeps_at_most_3_lessons():
    assert len(parse_reply(reply([], lessons=["a", "b", "c", "d"])).lessons) == 3


@pytest.mark.parametrize("lessons", ["one lesson", [1], [None], {"a": 1}])
def test_parse_rejects_bad_lessons(lessons):
    with pytest.raises(ReplyError, match="lessons"):
        parse_reply(reply([], lessons=lessons))
