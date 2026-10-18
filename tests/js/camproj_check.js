// Project the points of a case file with web/camproj.js and compare with the expected pixels.
// Usage: node camproj_check.js <case.json>.
// The file is [{ camera, points: [[x, y, z]], pixels: [[u, v]], limit: the largest error in pixels }].
"use strict";
const fs = require("fs");
const path = require("path");
const { project } = require(path.join(__dirname, "..", "..", "web", "camproj.js"));

const cases = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
let worst = 0;
for (const c of cases) {
  c.points.forEach((p, i) => {
    const got = project(c.camera, p);
    const want = c.pixels[i];
    if (!got) throw new Error(`${c.camera.name}: point ${p} is behind the camera`);
    const err = Math.hypot(got[0] - want[0], got[1] - want[1]);
    worst = Math.max(worst, err);
    if (err > (c.limit ?? 2)) throw new Error(`${c.camera.name}: point ${p} at ${got}, the render says ${want}`);
  });
}
console.log(`ok: worst error ${worst.toFixed(4)} px`);
