// Print the poses of web/plan3d.js for the times of a track case. The test compares them with the Python poses.
// Usage: node track_check.js <case.json>. The output is [{ name: { xyz, yaw } }], one item for each time.
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ctx = { window: {}, console };
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(__dirname, "..", "..", "web", "plan3d.js"), "utf8"), ctx);
const P = ctx.window.Plan3D;
const c = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const view = { track: c.track, believed: c.entities };
const who = P.carriers(view);
const out = c.times.map((t) => P.posesAt(c.track, c.entities, t, who.believed, who.believedMoves));
console.log(JSON.stringify(out));
