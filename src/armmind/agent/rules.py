"""The stop rules: compare an action result with what the model wanted."""
from __future__ import annotations

from dataclasses import replace

from armmind.types import CLOSED_ON_NOTHING_M, Action, ActionResult, fmt


def _resolve(name: str, labels: dict[str, str]) -> str | None:
    """The object name for a name or a label, matched without case. "nothing" gives ""."""
    want = name.strip().lower()
    if want == "nothing":
        return ""
    for obj, label in labels.items():
        if want in {obj.lower(), obj.lower().replace("_", " "), label.lower(), label.split(",")[0].strip().lower()}:
            return obj
    return None


def check_result(action: Action, result: ActionResult, labels: dict[str, str]) -> tuple[ActionResult, str | None]:
    """Compare a result with what the model wanted. Gives the result, maybe with a note, and the reason to stop
    the action list, or None."""
    held = result.held_after if result.held_after is not None else tuple(c for c in result.contacts if c != "table")
    before = result.held_before or ()
    expect = _resolve(action.expect, labels) if action.expect is not None else None
    if action.expect is not None and expect is None:
        note = f"unknown object {action.expect!r} in expect, ignored"
        result = replace(result, note=f"{result.note}, {note}" if result.note else note)
    if action.type == "move_to" and result.status in ("blocked", "unreachable"):
        return result, f"the move to {fmt(action.xyz)} was {result.status}"
    if action.type == "close" and not held and result.gripper_width < CLOSED_ON_NOTHING_M and expect != "":
        return result, "the gripper closed on nothing"
    if action.type != "open" and (dropped := [n for n in before if n not in held]):
        why = f"dropped {', '.join(dropped)}"
        return replace(result, note=f"{result.note}, {why}" if result.note else why), why
    if expect is not None and (expect not in held if expect else bool(held)):
        return result, f"expected to hold {expect or 'nothing'}, holds {', '.join(held) or 'nothing'}"
    return result, None
