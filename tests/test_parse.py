import pytest

from armmind.agent.parse import ReplyError, parse_reply
from armmind.types import Action


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
