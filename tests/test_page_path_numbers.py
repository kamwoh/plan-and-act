"""v1.16 page: the path of a plan and its execution counts keyframes, not actions."""
import json
import subprocess

from tests.test_page_plan import WEB, _body, _web

# 7 keyframes with 3 gripper changes: 10 actions, as keyframe_actions gives them.
NODE = r"""
const O = require(process.argv[1]);
const go = (x) => ({ type: "move_to", xyz: [x, 0, 0.1], yaw: 0 });
const actions = [go(0.40), go(0.41), { type: "close", expect: "red cube" }, go(0.42), go(0.43),
                 { type: "open", expect: "nothing" }, go(0.44), go(0.45), { type: "close", expect: "nothing" }, go(0.46)];
const spec = { point: "move_to", marks: { open: "open", close: "closed" }, label: () => null };
const groups = O.pointGroups(actions, "move_to");
const items = O.grouped(O.items(actions, { xyz: [0.3, 0, 0.4], yaw: 0 }, spec), groups);
const circles = items.filter((it) => it.kind === "point").map((it) => it.index + 1);
const marks = items.filter((it) => it.kind === "mark").map((it) => it.index + 1);
const running = O.groupStates(O.states(10, { current: 5 }), groups);  // the open of keyframe 4 runs
const stopped = O.groupStates(O.states(10, { ran: 3, stoppedAt: 2 }), groups);  // the close of keyframe 2 stops
console.log(JSON.stringify({ groups, circles, marks, running, stopped }));
"""


def test_7_keyframes_with_3_gripper_changes_give_circles_1_to_7():
    out = json.loads(subprocess.run(["node", "-e", NODE, str(WEB / "overlay.js")], check=True, capture_output=True,
                                    text=True).stdout)
    assert out["groups"] == [0, 1, 1, 2, 3, 3, 4, 5, 5, 6]
    assert out["circles"] == [1, 2, 3, 4, 5, 6, 7]
    assert out["marks"] == [2, 4, 6]  # each bracket sits with its keyframe and takes no number
    assert out["running"] == ["done", "done", "done", "current", "later", "later", "later"]
    assert out["stopped"] == ["done", "stopped", "skipped", "skipped", "skipped", "skipped", "skipped"]


def test_only_a_plan_execution_numbers_by_keyframe():
    js = _web("app.js")
    assert "planView.track.start, ev.action_data, true);" in js
    handler = _body(js, "function onEvent(ev, replay = false)")
    reply = handler[handler.index('ev.kind === "reply"'):handler.index('ev.kind === "action"')]
    assert "ev.action_data);" in reply  # the step mode keeps its numbers by action
    assert "PathOverlay.groupStates(actionStates, stepPath.groups)" in _body(js, "function drawPath()")
