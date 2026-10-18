// A path overlay on a camera image: the planned points of one step, in order, with their states.
// All names and shapes come from the data, as in plan3d.js. The projection is CamProj.project.
//
// A path is { start: { xyz, yaw } | null, items }. Each item belongs to one action of the step:
//   { index, kind: "point", xyz, yaw }         a target, numbered index + 1, with a tick for the yaw
//   { index, kind: "mark", xyz, shape, label } a bracket at xyz: shape "open" or "closed", label text or null
//   { index, kind: "none" }                    an action with nothing to draw
// A point can also have a label. The yaw is in degrees about world +z. At yaw 0 the tick lies along world x.
(function (root) {
  "use strict";
  const project = (root.CamProj || (typeof require !== "undefined" ? require("./camproj.js") : null)).project;

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

  const STYLE = {
    later: { color: "#6f9bff", width: 2.5, dash: [7, 5], alpha: 1 },
    current: { color: "#ffd23f", width: 5, dash: [], alpha: 1 },
    done: { color: "#b4bac4", width: 2.5, dash: [], alpha: 0.9 },
    stopped: { color: "#ff4d4d", width: 3.5, dash: [], alpha: 1 },
    skipped: { color: "#6f9bff", width: 2.5, dash: [7, 5], alpha: 0.3 },
  };
  const ORDER = ["skipped", "done", "later", "stopped", "current"];  // the later ones draw on top
  const TICK_M = 0.04;  // half the length of the yaw tick, in metres

  // The pixel positions of the path on one camera: the line pieces, the points and the marks.
  function layout(cam, path) {
    const px = (p) => (p ? project(cam, p) : null);
    const pieces = [];
    const points = [];
    const marks = [];
    let last = path.start ? path.start.xyz : null;
    const atPoint = {};  // how many marks sit at a pixel already, so that the next one moves to the side
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
        const k = atPoint[key] || 0;
        atPoint[key] = k + 1;
        marks.push({ index: it.index, at, offset: 20 + 24 * k, shape: it.shape, label: it.label || null });
      }
    }
    return { pieces, points, marks };
  }

  function useStyle(ctx, s) {
    ctx.strokeStyle = s.color;
    ctx.fillStyle = s.color;
    ctx.lineWidth = s.width;
    ctx.setLineDash(s.dash);
    ctx.globalAlpha = s.alpha;
  }

  // A text with a small dark box behind it, so that it reads on the light table and on dark areas.
  function tag(ctx, text, x, y, color) {
    ctx.setLineDash([]);
    ctx.font = "600 14px Inter, sans-serif";
    ctx.textBaseline = "middle";
    const w = ctx.measureText(text).width + 10;
    ctx.fillStyle = "rgba(10, 12, 16, 0.8)";
    ctx.fillRect(x, y - 10, w, 20);
    ctx.fillStyle = color;
    ctx.fillText(text, x + 5, y + 0.5);
  }

  function bracket(ctx, x, y, shape) {
    const gap = shape === "closed" ? 2.5 : 7, h = 7, foot = 4;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(x - gap + foot, y - h);
    ctx.lineTo(x - gap, y - h);
    ctx.lineTo(x - gap, y + h);
    ctx.lineTo(x - gap + foot, y + h);
    ctx.moveTo(x + gap - foot, y - h);
    ctx.lineTo(x + gap, y - h);
    ctx.lineTo(x + gap, y + h);
    ctx.lineTo(x + gap - foot, y + h);
    ctx.stroke();
  }

  // Draw the path on a canvas over an image of cam.width x cam.height pixels. An empty path clears it.
  function draw(canvas, cam, path, actionStates) {
    const ctx = canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!path || !cam) return;
    const r = canvas.width / cam.width;
    ctx.setTransform(r, 0, 0, r, 0, 0);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const L = layout(cam, path);
    const stateOf = (i) => actionStates[i] || "later";
    const rank = (i) => ORDER.indexOf(stateOf(i));
    const outlined = (fn, s) => {  // a dark line under each line, for the contrast
      ctx.save();
      useStyle(ctx, { ...s, color: "rgba(0, 0, 0, 0.55)", width: s.width + 2.5 });
      fn();
      ctx.restore();
      useStyle(ctx, s);
      fn();
    };
    for (const p of [...L.pieces].sort((a, b) => rank(a.index) - rank(b.index))) {
      outlined(() => {
        ctx.beginPath();
        ctx.moveTo(p.a[0], p.a[1]);
        ctx.lineTo(p.b[0], p.b[1]);
        ctx.stroke();
      }, STYLE[stateOf(p.index)]);
    }
    for (const p of [...L.points].sort((a, b) => rank(a.index) - rank(b.index))) {
      const s = STYLE[stateOf(p.index)];
      if (p.tick) {
        outlined(() => {
          ctx.beginPath();
          ctx.moveTo(p.tick[0][0], p.tick[0][1]);
          ctx.lineTo(p.tick[1][0], p.tick[1][1]);
          ctx.stroke();
        }, { ...s, dash: [], width: 2.5 });
      }
      useStyle(ctx, { ...s, dash: [] });
      ctx.beginPath();
      ctx.arc(p.at[0], p.at[1], 10, 0, 2 * Math.PI);
      ctx.fillStyle = "rgba(10, 12, 16, 0.82)";
      ctx.fill();
      ctx.lineWidth = stateOf(p.index) === "current" ? 3.5 : 2.5;
      ctx.stroke();
      ctx.fillStyle = s.color;
      ctx.font = "700 13px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(p.index + 1), p.at[0], p.at[1] + 0.5);
      ctx.textAlign = "left";
      if (p.label) tag(ctx, p.label, p.at[0] + 14, p.at[1] - 16, s.color);
    }
    for (const m of [...L.marks].sort((a, b) => rank(a.index) - rank(b.index))) {
      const s = STYLE[stateOf(m.index)];
      const x = m.at[0] + m.offset, y = m.at[1];
      ctx.globalAlpha = s.alpha;
      ctx.fillStyle = "rgba(10, 12, 16, 0.82)";
      ctx.fillRect(x - 12, y - 11, 24, 22);
      useStyle(ctx, { ...s, dash: [], width: 2.5 });
      bracket(ctx, x, y, m.shape);
      if (m.label) tag(ctx, m.label, x + 14, y, s.color);
    }
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);
  }

  const api = { states, layout, draw };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.PathOverlay = api;
})(typeof window !== "undefined" ? window : this);
