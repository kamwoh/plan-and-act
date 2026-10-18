// A path overlay on a camera image: the planned points of one step, in order, with their states.
// All names and shapes come from the data, as in plan3d.js. The projection is CamProj.project.
//
// A path is { start: { xyz, yaw } | null, items }. Each item belongs to one action of the step:
//   { index, kind: "point", xyz, yaw, label }  a target, numbered index + 1, with a tick for the yaw
//   { index, kind: "mark", xyz, shape, label } a bracket at xyz: shape "open" or "closed"
//   { index, kind: "none" }                    an action with nothing to draw
// A label is a text or null. The yaw is in degrees about world +z. At yaw 0 the tick lies along world x.
(function (root) {
  "use strict";
  const project = (root.CamProj || (typeof require !== "undefined" ? require("./camproj.js") : null)).project;

  // The items of a list of actions. spec names the action types:
  //   { point: the type that moves to `xyz` (with an optional `yaw`),
  //     marks: { type: "open" | "closed" }, the types that change the state where the path is now,
  //     label(action): the label text of an action, or null }
  // A mark sits at the last point, or at the start when no point comes before it. A point without a yaw
  // keeps the yaw before it.
  function items(actions, start, spec) {
    let at = start ? start.xyz : null;
    let yaw = start && typeof start.yaw === "number" ? start.yaw : 0;
    const label = spec.label || (() => null);
    return actions.map((a, index) => {
      if (a.type === spec.point && a.xyz) {
        at = a.xyz;
        if (typeof a.yaw === "number") yaw = a.yaw;
        return { index, kind: "point", xyz: a.xyz, yaw, label: label(a) };
      }
      if (spec.marks[a.type] && at) return { index, kind: "mark", xyz: at, shape: spec.marks[a.type], label: label(a) };
      return { index, kind: "none" };
    });
  }

  // The progress when a step ends. results: one line for each action that ran or was skipped, in order.
  // stopped: { after: the number of the action that stopped the step } or null. isSkipped(line) is true
  // for a line of an action that did not run.
  function ended(progress, results, stopped, isSkipped) {
    const ran = stopped ? stopped.after : (results || []).filter((r) => !isSkipped(r)).length;
    return { ...progress, ran, stoppedAt: stopped ? stopped.after - 1 : null };
  }

  // The state of each action of a step: "done", "current", "later", "stopped" or "skipped".
  // progress: { current: the index of the action that runs now, or null,
  //             ran: the number of actions that ran when the step ended, or null while it runs,
  //             stoppedAt: the index of the action that stopped the step, or null }
  function states(n, progress) {
    const p = progress || {};
    const out = [];
    for (let i = 0; i < n; i++) {
      if (p.ran !== null && p.ran !== undefined) {
        if (p.stoppedAt !== null && p.stoppedAt !== undefined) {
          out.push(i < p.stoppedAt ? "done" : i === p.stoppedAt ? "stopped" : "skipped");
        } else {
          out.push(i < p.ran ? "done" : "skipped");
        }
      } else if (p.current !== null && p.current !== undefined) {
        out.push(i < p.current ? "done" : i === p.current ? "current" : "later");
      } else {
        out.push("later");
      }
    }
    return out;
  }

  // The sizes are in screen pixels. draw() scales them to image pixels.
  const STYLE = {
    later: { color: "#6f9bff", width: 2, dash: [6, 4], alpha: 1 },
    current: { color: "#ffd23f", width: 4, dash: [], alpha: 1 },
    done: { color: "#b4bac4", width: 2, dash: [], alpha: 0.9 },
    stopped: { color: "#ff4d4d", width: 3, dash: [], alpha: 1 },
    skipped: { color: "#6f9bff", width: 2, dash: [6, 4], alpha: 0.3 },
  };
  const ORDER = ["skipped", "done", "later", "stopped", "current"];  // the later ones draw on top
  const TICK_M = 0.04;  // half the length of the yaw tick, in metres

  // The pixel positions of the path on one camera: the line pieces, the points and the marks. A mark has a
  // slot: 0 for the first mark at a pixel, 1 for the next one, so that it can move to the side.
  function layout(cam, path) {
    const px = (p) => (p ? project(cam, p) : null);
    const pieces = [];
    const points = [];
    const marks = [];
    let last = path.start ? path.start.xyz : null;
    const atPoint = {};
    for (const it of path.items) {
      if (it.kind === "point") {
        const a = px(last), b = px(it.xyz);
        if (a && b) pieces.push({ index: it.index, a, b });
        if (b) {
          const yaw = ((it.yaw || 0) * Math.PI) / 180;
          const d = [Math.cos(yaw) * TICK_M, Math.sin(yaw) * TICK_M, 0];
          const t0 = px([it.xyz[0] - d[0], it.xyz[1] - d[1], it.xyz[2]]);
          const t1 = px([it.xyz[0] + d[0], it.xyz[1] + d[1], it.xyz[2]]);
          points.push({ index: it.index, at: b, tick: t0 && t1 ? [t0, t1] : null, label: it.label || null });
        }
        last = it.xyz;
      } else if (it.kind === "mark") {
        const at = px(it.xyz);
        if (!at) continue;
        const key = `${Math.round(at[0])},${Math.round(at[1])}`;
        const slot = atPoint[key] || 0;
        atPoint[key] = slot + 1;
        marks.push({ index: it.index, at, slot, shape: it.shape, label: it.label || null });
      }
    }
    return { pieces, points, marks };
  }

  // Draw the path on a canvas over an image of cam.width x cam.height pixels. scale: image pixels for one
  // screen pixel, so that the text and the lines keep their size when the image shows smaller. An empty
  // path clears the canvas.
  function draw(canvas, cam, path, actionStates, scale = 1) {
    const ctx = canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!path || !cam) return;
    const r = canvas.width / cam.width;
    const k = scale;
    ctx.setTransform(r, 0, 0, r, 0, 0);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const L = layout(cam, path);
    const stateOf = (i) => actionStates[i] || "later";
    const rank = (i) => ORDER.indexOf(stateOf(i));
    const byRank = (list) => [...list].sort((a, b) => rank(a.index) - rank(b.index));
    const use = (s, width = s.width, dash = s.dash) => {
      ctx.strokeStyle = s.color;
      ctx.fillStyle = s.color;
      ctx.lineWidth = width * k;
      ctx.setLineDash(dash.map((v) => v * k));
      ctx.globalAlpha = s.alpha;
    };
    const line = (a, b, s, width = s.width, dash = s.dash) => {  // a dark line under each line, for the contrast
      for (const dark of [true, false]) {
        use(dark ? { ...s, color: "rgba(0, 0, 0, 0.55)" } : s, dark ? width + 2 : width, dash);
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
      }
    };
    // A text with a small dark box behind it, so that it reads on the light table and on dark areas.
    const tag = (text, x, y, color) => {
      ctx.setLineDash([]);
      ctx.font = `600 ${12 * k}px Inter, sans-serif`;
      ctx.textBaseline = "middle";
      ctx.textAlign = "left";
      const w = ctx.measureText(text).width + 8 * k;
      ctx.fillStyle = "rgba(10, 12, 16, 0.8)";
      ctx.fillRect(x, y - 9 * k, w, 18 * k);
      ctx.fillStyle = color;
      ctx.fillText(text, x + 4 * k, y + 0.5 * k);
    };
    const bracket = (x, y, shape) => {
      const gap = (shape === "closed" ? 2 : 5.5) * k, h = 5.5 * k, foot = 3 * k;
      ctx.setLineDash([]);
      ctx.beginPath();
      for (const side of [-1, 1]) {
        ctx.moveTo(x + side * (gap - foot), y - h);
        ctx.lineTo(x + side * gap, y - h);
        ctx.lineTo(x + side * gap, y + h);
        ctx.lineTo(x + side * (gap - foot), y + h);
      }
      ctx.stroke();
    };

    for (const p of byRank(L.pieces)) line(p.a, p.b, STYLE[stateOf(p.index)]);
    for (const p of byRank(L.points)) {
      const state = stateOf(p.index), s = STYLE[state];
      if (p.tick) line(p.tick[0], p.tick[1], s, 2, []);
      use(s, state === "current" ? 2.5 : 1.8, []);
      ctx.beginPath();
      ctx.arc(p.at[0], p.at[1], 9 * k, 0, 2 * Math.PI);
      ctx.fillStyle = "rgba(10, 12, 16, 0.82)";
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = s.color;
      ctx.font = `700 ${11 * k}px Inter, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(p.index + 1), p.at[0], p.at[1] + 0.5 * k);
      if (p.label) tag(p.label, p.at[0] + 12 * k, p.at[1] - 14 * k, s.color);
    }
    for (const m of byRank(L.marks)) {
      const s = STYLE[stateOf(m.index)];
      const x = m.at[0] + (17 + 20 * m.slot) * k, y = m.at[1];
      ctx.globalAlpha = s.alpha;
      ctx.fillStyle = "rgba(10, 12, 16, 0.82)";
      ctx.fillRect(x - 9 * k, y - 9 * k, 18 * k, 18 * k);
      use(s, 2, []);
      bracket(x, y, m.shape);
      if (m.label) tag(m.label, x + 12 * k, y, s.color);
    }
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);
  }

  const api = { items, ended, states, layout, draw };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.PathOverlay = api;
})(typeof window !== "undefined" ? window : this);
