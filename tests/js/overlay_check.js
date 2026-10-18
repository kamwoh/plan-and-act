// The state logic and the layout of web/overlay.js. Usage: node overlay_check.js
"use strict";
const assert = require("assert");
const path = require("path");
const { states, layout } = require(path.join(__dirname, "..", "..", "web", "overlay.js"));

// A reply arrived and nothing runs yet: every action is later.
assert.deepStrictEqual(states(3, {}), ["later", "later", "later"]);
// Action 2 of 4 runs now.
assert.deepStrictEqual(states(4, { current: 1, ran: null }), ["done", "current", "later", "later"]);
// The step ended and every action ran.
assert.deepStrictEqual(states(3, { current: 2, ran: 3, stoppedAt: null }), ["done", "done", "done"]);
// Action 2 stopped the step: it turns red and the rest is skipped.
assert.deepStrictEqual(states(4, { current: 1, ran: 2, stoppedAt: 1 }), ["done", "stopped", "skipped", "skipped"]);
// The task was solved after 2 actions, or the user stopped the run: the rest is skipped.
assert.deepStrictEqual(states(4, { current: 1, ran: 2, stoppedAt: null }), ["done", "done", "skipped", "skipped"]);
// No actions.
assert.deepStrictEqual(states(0, { ran: 0 }), []);

// The layout on a camera that looks down from 1 m above the origin, with up along +x.
const cam = { pos: [0, 0, 1], forward: [0, 0, -1], up: [1, 0, 0], fovy: 90, width: 200, height: 200 };
const L = layout(cam, {
  start: { xyz: [0, 0, 0], yaw: 0 },
  items: [
    { index: 0, kind: "point", xyz: [0.5, 0, 0], yaw: 90 },
    { index: 1, kind: "mark", xyz: [0.5, 0, 0], shape: "closed", label: "→ nothing" },
    { index: 2, kind: "none" },
    { index: 3, kind: "point", xyz: [0.5, 0.5, 0], yaw: 0 },
  ],
});
const near = (a, b) => Math.abs(a[0] - b[0]) < 1e-9 && Math.abs(a[1] - b[1]) < 1e-9;
assert.strictEqual(L.pieces.length, 2);
assert.ok(near(L.pieces[0].a, [100, 100]) && near(L.pieces[0].b, [100, 50]));  // +x is up in the image
assert.ok(near(L.pieces[1].b, [50, 50]));  // +y is to the left
assert.deepStrictEqual(L.points.map((p) => p.index), [0, 3]);
assert.deepStrictEqual(L.marks.map((m) => [m.index, m.shape, m.label]), [[1, "closed", "→ nothing"]]);
// The yaw tick of point 1 (yaw 90) lies along world y, so it is level in the image.
const [t0, t1] = L.points[0].tick;
assert.ok(Math.abs(t0[1] - t1[1]) < 1e-9 && Math.abs(t0[0] - t1[0]) > 1);
// The tick of point 4 (yaw 0) lies along world x, so it is upright in the image.
const [u0, u1] = L.points[1].tick;
assert.ok(Math.abs(u0[0] - u1[0]) < 1e-9 && Math.abs(u0[1] - u1[1]) > 1);
console.log("ok");
