// The state logic and the layout of web/overlay.js. Usage: node overlay_check.js
"use strict";
const assert = require("assert");
const path = require("path");
const { items, ended, states, layout } = require(path.join(__dirname, "..", "..", "web", "overlay.js"));

// ---- items: the actions of a step as overlay items ----
const SPEC = { point: "go", marks: { release: "open", grip: "closed" },
               label: (a) => (a.expect ? `→ ${a.expect}` : null) };
const start = { xyz: [0.1, 0.2, 0.3], yaw: 15 };
const got = items([
  { type: "grip", expect: "nothing" },                     // no point before: at the start
  { type: "go", xyz: [0.5, 0, 0.2], yaw: 90 },
  { type: "go", xyz: [0.5, 0, 0.1] },                      // no yaw: keeps 90
  { type: "grip", expect: "a box" },                       // at the last point
  { type: "pause", seconds: 1 },                           // nothing to draw
  { type: "release" },
  { type: "go", xyz: [0.4, 0.1, 0.2], expect: "a box" },   // a point can have a label
], start, SPEC);
assert.deepStrictEqual(got, [
  { index: 0, kind: "mark", xyz: [0.1, 0.2, 0.3], shape: "closed", label: "→ nothing" },
  { index: 1, kind: "point", xyz: [0.5, 0, 0.2], yaw: 90, label: null },
  { index: 2, kind: "point", xyz: [0.5, 0, 0.1], yaw: 90, label: null },
  { index: 3, kind: "mark", xyz: [0.5, 0, 0.1], shape: "closed", label: "→ a box" },
  { index: 4, kind: "none" },
  { index: 5, kind: "mark", xyz: [0.5, 0, 0.1], shape: "open", label: null },
  { index: 6, kind: "point", xyz: [0.4, 0.1, 0.2], yaw: 90, label: "→ a box" },
]);
// The first point without a yaw takes the start yaw. Without a start, a mark has no place.
assert.strictEqual(items([{ type: "go", xyz: [0, 0, 0] }], start, SPEC)[0].yaw, 15);
assert.deepStrictEqual(items([{ type: "release" }], null, SPEC), [{ index: 0, kind: "none" }]);

// ---- ended: the progress when a step ends ----
const skipped = (line) => line.endsWith(": skipped");
// All 3 ran.
assert.deepStrictEqual(ended({ current: 2 }, ["a: ok", "b: ok", "c: ok"], null, skipped),
                       { current: 2, ran: 3, stoppedAt: null });
// The task was solved, or the user stopped the run, after 2 of 4: no lines for the rest.
assert.deepStrictEqual(ended({ current: 1 }, ["a: ok", "b: ok"], null, skipped), { current: 1, ran: 2, stoppedAt: null });
// Action 2 of 4 stopped the step: its lines and the skipped lines.
assert.deepStrictEqual(ended({ current: 1 }, ["a: ok", "b: bad", "c: skipped", "d: skipped"], { after: 2 }, skipped),
                       { current: 1, ran: 2, stoppedAt: 1 });
// A skipped line without a stop does not count as ran.
assert.deepStrictEqual(ended({}, ["a: ok", "b: skipped"], null, skipped), { ran: 1, stoppedAt: null });
assert.deepStrictEqual(states(4, ended({ current: 1 }, ["a: ok", "b: bad", "c: skipped", "d: skipped"], { after: 2 }, skipped)),
                       ["done", "stopped", "skipped", "skipped"]);

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
assert.deepStrictEqual(L.marks.map((m) => [m.index, m.shape, m.label, m.slot]), [[1, "closed", "→ nothing", 0]]);
// The yaw tick of point 1 (yaw 90) lies along world y, so it is level in the image.
const [t0, t1] = L.points[0].tick;
assert.ok(Math.abs(t0[1] - t1[1]) < 1e-9 && Math.abs(t0[0] - t1[0]) > 1);
// The tick of point 4 (yaw 0) lies along world x, so it is upright in the image.
const [u0, u1] = L.points[1].tick;
assert.ok(Math.abs(u0[0] - u1[0]) < 1e-9 && Math.abs(u0[1] - u1[1]) > 1);
console.log("ok");
