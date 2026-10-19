// A generic player for a PlanView: fixed entities, the believed entities, an agent that follows a timed
// track, and one canvas for each camera of the environment. All names and shapes come from the view data.
(function () {
  "use strict";
  let T = null;  // window.THREE, read when a player is made: three.js loads async and can come late
  const RAD = Math.PI / 180;
  const BELIEF_COLOR = 0x9aa3b2;
  const PATH_COLOR = 0x5b8cff;
  const FAIL_COLOR = 0xff4d4d;
  const MOVE_COLOR = 0xffb020;  // a keyframe that moves an entity without carrying it
  const DEFAULT_COLOR = [0.78, 0.78, 0.78, 1];

  const clamp01 = (a) => Math.min(1, Math.max(0, a));
  const smooth = (a) => a * a * (3 - 2 * a);  // the same timing as the environment's motion
  const lerp = (a, b, s) => a + (b - a) * s;
  const lerp3 = (a, b, s) => [lerp(a[0], b[0], s), lerp(a[1], b[1], s), lerp(a[2], b[2], s)];
  const rgb = (c) => new T.Color((c || DEFAULT_COLOR)[0], (c || DEFAULT_COLOR)[1], (c || DEFAULT_COLOR)[2]);

  function unitGeometry(shape) {
    if (shape === "cylinder") {
      const g = new T.CylinderGeometry(0.5, 0.5, 1, 40);
      g.rotateX(Math.PI / 2);  // the axis along +z
      return g;
    }
    if (shape === "sphere") return new T.SphereGeometry(0.5, 32, 16);
    return new T.BoxGeometry(1, 1, 1);
  }

  // An entity as a group: a unit shape scaled to the size, at the centre, turned by the yaw.
  // style: "solid", or "ghost" (see-through grey with light edges).
  function entityGroup(e, style) {
    const group = new T.Group();
    const geo = unitGeometry(e.shape);
    const ghost = style === "ghost";
    const mat = new T.MeshLambertMaterial({
      color: ghost ? BELIEF_COLOR : rgb(e.color), transparent: ghost || (e.color && e.color[3] < 1),
      opacity: ghost ? 0.55 : (e.color ? e.color[3] : 1), depthWrite: !ghost,
    });
    group.add(new T.Mesh(geo, mat));
    if (ghost) group.add(new T.LineSegments(new T.EdgesGeometry(geo, 30), new T.LineBasicMaterial({ color: 0xdfe5ee })));
    group.scale.set(e.size[0], e.size[1], e.size[2]);
    group.position.set(e.xyz[0], e.xyz[1], e.xyz[2]);
    group.rotation.z = (e.yaw || 0) * RAD;
    group.userData.name = e.name;
    return group;
  }

  // ---- Time: the agent pose and the carried entities at time t ----
  function agentAt(track, t) {
    const kfs = track.keyframes;
    let prev = track.start;
    let t0 = 0;
    for (let k = 0; k < kfs.length; k++) {
      const kf = kfs[k];
      if (t < kf.t) {
        const s = smooth(clamp01((t - t0) / Math.max(1e-6, kf.t - t0)));
        return { xyz: lerp3(prev.xyz, kf.xyz, s), yaw: lerp(prev.yaw, kf.yaw, s), gap: prev.gap, index: k };
      }
      if (t < kf.t_done) {
        const s = kf.changes ? smooth(clamp01((t - kf.t) / Math.max(1e-6, kf.t_done - kf.t))) : 1;
        return { xyz: kf.xyz.slice(), yaw: kf.yaw, gap: lerp(prev.gap, kf.gap, s), index: k };
      }
      prev = kf;
      t0 = kf.t_done;
    }
    return { xyz: prev.xyz.slice(), yaw: prev.yaw, gap: prev.gap, index: Math.max(0, kfs.length - 1) };
  }

  // An entity pose in the frame of an agent pose, and back.
  function toLocal(agent, pose) {
    const c = Math.cos(-agent.yaw * RAD), s = Math.sin(-agent.yaw * RAD);
    const dx = pose.xyz[0] - agent.xyz[0], dy = pose.xyz[1] - agent.xyz[1];
    return { xyz: [c * dx - s * dy, s * dx + c * dy, pose.xyz[2] - agent.xyz[2]], yaw: pose.yaw - agent.yaw };
  }

  function toWorld(agent, local) {
    const c = Math.cos(agent.yaw * RAD), s = Math.sin(agent.yaw * RAD);
    return {
      xyz: [agent.xyz[0] + c * local.xyz[0] - s * local.xyz[1], agent.xyz[1] + s * local.xyz[0] + c * local.xyz[1],
            agent.xyz[2] + local.xyz[2]],
      yaw: agent.yaw + local.yaw,
    };
  }

  // ---- Boxes that turn only about z: [centre, full size, yaw in degrees] ----
  function corners(box) {
    const [[x, y], [w, d], yaw] = box;
    const c = Math.cos(yaw * RAD), s = Math.sin(yaw * RAD);
    return [[w / 2, d / 2], [-w / 2, d / 2], [-w / 2, -d / 2], [w / 2, -d / 2]].map(([u, v]) => [x + c * u - s * v, y + s * u + c * v]);
  }

  // True when the two boxes share a volume deeper than margin (a touch is not an overlap).
  function boxesOverlap(a, b, margin = 1e-4) {
    if (Math.abs(a[0][2] - b[0][2]) >= (a[1][2] + b[1][2]) / 2 - margin) return false;
    const ca = corners(a), cb = corners(b);
    for (const yaw of [a[2], a[2] + 90, b[2], b[2] + 90]) {  // the separating axis test in the plane
      const ax = [Math.cos(yaw * RAD), Math.sin(yaw * RAD)];
      const pa = ca.map((p) => p[0] * ax[0] + p[1] * ax[1]);
      const pb = cb.map((p) => p[0] * ax[0] + p[1] * ax[1]);
      if (Math.min(Math.max(...pa), Math.max(...pb)) - Math.max(Math.min(...pa), Math.min(...pb)) <= margin) return false;
    }
    return true;
  }

  function flatBox(e, pose) {  // the footprint as a tall box at z 0: round shapes do not turn
    const yaw = e.shape === "cylinder" || e.shape === "sphere" ? 0 : pose.yaw;
    return [[pose.xyz[0], pose.xyz[1], 0], [e.size[0], e.size[1], 1], yaw];
  }

  // A released entity falls onto the table (z 0) or onto the highest entity below it. It never rises.
  // An entity without a size stays where it is. src/armmind/plan/track.py has the same rule in drop_pose.
  function dropPose(name, byName, poses) {
    const e = byName[name], p = poses[name];
    if (!e || !e.size) return p;
    const flat = flatBox(e, p);
    let support = 0;
    for (const [other, o] of Object.entries(byName)) {
      if (other === name || !o.size || !poses[other]) continue;
      const top = poses[other].xyz[2] + o.size[2] / 2;
      if (top <= p.xyz[2] && boxesOverlap(flat, flatBox(o, poses[other]))) support = Math.max(support, top);
    }
    return { xyz: [p.xyz[0], p.xyz[1], Math.min(p.xyz[2], support + e.size[2] / 2)], yaw: p.yaw };
  }

  // The angle in (-180, 180].
  function wrapDeg(a) {
    const w = (((a + 180) % 360) + 360) % 360 - 180;
    return w === -180 ? 180 : w;
  }

  // A pose part of the way to a move target. The yaw takes the short way round.
  function movePose(pose, m, s) {
    const target = pose.yaw + wrapDeg(m.yaw - pose.yaw);
    return { xyz: lerp3(pose.xyz, m.xyz, s), yaw: lerp(pose.yaw, target, s) };
  }

  // The poses of a set of entities at time t. carriedAt(k, keyframe) names the entity of the set that the
  // agent carries after keyframe k, or gives null. carriedAt(-1, track.start) names the entity in the hand at
  // the start. movesAt(k, keyframe) gives the predicted moves of keyframe
  // k: [{ name, xyz, yaw }]. On the segment into keyframe k, each moved entity goes to its new pose with the
  // timing of the agent. A carried entity does not move. A release drops the entity (see dropPose).
  // src/armmind/plan/track.py has the same rules.
  function posesAt(track, entities, t, carriedAt, movesAt) {
    const poses = {};
    const byName = {};
    for (const e of entities) {
      poses[e.name] = { xyz: e.xyz.slice(), yaw: e.yaw || 0 };
      byName[e.name] = e;
    }
    let carried = null;
    const first = carriedAt(-1, track.start) || null;  // a round that continues can start with it in the hand
    if (first && poses[first]) carried = { name: first, local: toLocal(track.start, poses[first]), from: -1 };
    let t0 = 0;
    const kfs = track.keyframes;
    const free = (name) => poses[name] && !(carried && carried.name === name);
    for (let k = 0; k < kfs.length; k++) {
      const kf = kfs[k];
      const moves = (movesAt && movesAt(k, kf)) || [];
      if (t < kf.t) {
        const s = smooth(clamp01((t - t0) / Math.max(1e-6, kf.t - t0)));
        for (const m of moves) if (free(m.name)) poses[m.name] = movePose(poses[m.name], m, s);
        break;
      }
      const at = { xyz: kf.xyz, yaw: kf.yaw };
      const want = carriedAt(k, kf) || null;
      if (carried && carried.name !== want) {  // a release: the entity falls
        poses[carried.name] = toWorld(at, carried.local);
        poses[carried.name] = dropPose(carried.name, byName, poses);
        carried = null;
      }
      for (const m of moves) if (free(m.name)) poses[m.name] = movePose(poses[m.name], m, 1);
      if (want && !carried && poses[want]) carried = { name: want, local: toLocal(at, poses[want]), from: k };
      t0 = kf.t_done;
    }
    if (carried) poses[carried.name] = toWorld(agentAt(track, t), carried.local);
    return poses;
  }

  // Who carries and moves what: the belief follows the track, with its predicted moves.
  function carriers(view) {
    return {
      believed: (k, kf) => kf.carrying,
      believedMoves: (k, kf) => kf.moves || [],
    };
  }

  // ---- One camera view ----
  class CameraView {
    constructor(slot, cam, view) {
      this.cam = cam;
      this.canvas = slot.canvas;
      this.overlay = slot.overlay;
      this.renderer = slot.renderer;
      this.scene = new T.Scene();
      this.scene.background = new T.Color(0x000000);
      const W = cam.width, H = cam.height;
      this.camera = new T.PerspectiveCamera(cam.fovy, W / H, 0.01, 30);
      this.scene.add(this.camera);
      this.scene.add(new T.AmbientLight(0xffffff, 0.5));
      const sun = new T.DirectionalLight(0xffffff, 0.5);
      sun.position.set(0, 0, 1.5);
      this.scene.add(sun);
      const head = new T.DirectionalLight(0xffffff, 0.4);  // a light at the camera, as a headlight
      head.position.set(0, 0, 1);
      this.camera.add(head);
      this.resetCamera();
      this.build(view);
      this.bindOrbit();
    }

    resetCamera() {
      const c = this.cam;
      this.camera.position.set(c.pos[0], c.pos[1], c.pos[2]);
      this.camera.up.set(c.up[0], c.up[1], c.up[2]);
      this.camera.lookAt(c.pos[0] + c.forward[0], c.pos[1] + c.forward[1], c.pos[2] + c.forward[2]);
      const down = c.forward[2] < -0.05 ? -c.pos[2] / c.forward[2] : 1;  // where the view axis meets the floor
      this.target = new T.Vector3(c.pos[0] + c.forward[0] * down, c.pos[1] + c.forward[1] * down,
                                  c.pos[2] + c.forward[2] * down);
      this.orbiting = false;
      this.dirty = true;
    }

    build(view) {
      for (const f of view.fixtures) this.scene.add(entityGroup(f, "solid"));
      if (this.cam.grid && view.grid) this.addGrid(view.grid);
      this.believed = {};
      for (const e of view.believed) this.scene.add(this.believed[e.name] = entityGroup(e, "ghost"));
      this.addAgent(view.agent);
      this.addPath(view);
    }

    addGrid(grid) {
      const pts = [];
      const z = (grid.z || 0) + 0.0008;
      const x0 = grid.x[0], x1 = grid.x[grid.x.length - 1], y0 = grid.y[0], y1 = grid.y[grid.y.length - 1];
      for (const x of grid.x) pts.push(new T.Vector3(x, y0, z), new T.Vector3(x, y1, z));
      for (const y of grid.y) pts.push(new T.Vector3(x0, y, z), new T.Vector3(x1, y, z));
      const geo = new T.BufferGeometry().setFromPoints(pts);
      this.scene.add(new T.LineSegments(geo, new T.LineBasicMaterial({ color: 0x1e1e1e })));
      this.grid = grid;
    }

    addAgent(agent) {
      this.agent = new T.Group();
      this.slideAxis = agent.slide_axis || [1, 0, 0];
      this.parts = agent.parts.map((p) => {
        const g = entityGroup({ ...p, yaw: 0, color: agent.color }, "solid");
        g.userData.base = p.xyz;
        g.userData.slide = p.slide || 0;
        this.agent.add(g);
        return g;
      });
      this.scene.add(this.agent);
    }

    addPath(view) {
      const track = view.track;
      const pts = [track.start.xyz, ...track.keyframes.map((k) => k.xyz)].map((p) => new T.Vector3(p[0], p[1], p[2]));
      const line = new T.Line(new T.BufferGeometry().setFromPoints(pts),
                              new T.LineBasicMaterial({ color: PATH_COLOR, depthTest: false, transparent: true, opacity: 0.85 }));
      line.renderOrder = 10;
      this.scene.add(line);
      const marks = view.marks || [];
      track.keyframes.forEach((k, i) => {
        const bad = marks[i] && marks[i].ok === false;
        const moves = (k.moves || []).length > 0;
        const color = bad ? FAIL_COLOR : moves ? MOVE_COLOR : PATH_COLOR;
        const dot = new T.Mesh(new T.SphereGeometry(bad ? 0.009 : moves ? 0.008 : 0.005, 16, 8),
                               new T.MeshBasicMaterial({ color, depthTest: false, transparent: true }));
        dot.position.set(k.xyz[0], k.xyz[1], k.xyz[2]);
        dot.renderOrder = 11;
        this.scene.add(dot);
      });
    }

    pose(groups, poses) {
      for (const [name, p] of Object.entries(poses)) {
        const g = groups[name];
        if (!g) continue;
        g.position.set(p.xyz[0], p.xyz[1], p.xyz[2]);
        g.rotation.z = p.yaw * RAD;
      }
    }

    update(agentPose, believedPoses) {
      this.agent.position.set(agentPose.xyz[0], agentPose.xyz[1], agentPose.xyz[2]);
      this.agent.rotation.z = agentPose.yaw * RAD;
      const a = this.slideAxis;
      for (const g of this.parts) {
        const d = (g.userData.slide * agentPose.gap) / 2;
        const b = g.userData.base;
        g.position.set(b[0] + a[0] * d, b[1] + a[1] * d, b[2] + a[2] * d);
      }
      this.pose(this.believed, believedPoses);
      this.dirty = true;
    }

    render() {
      if (!this.dirty) return;
      this.dirty = false;
      this.renderer.render(this.scene, this.camera);
      this.drawLabels();
    }

    // The pose of the three.js camera now, in the form of CamProj. It moves when the user turns the view.
    cameraNow() {
      const c = this.camera;
      const forward = new T.Vector3();
      c.getWorldDirection(forward);
      const up = new T.Vector3(0, 1, 0).applyQuaternion(c.quaternion);
      return { pos: c.position.toArray(), forward: forward.toArray(), up: up.toArray(), fovy: this.cam.fovy,
               width: this.cam.width, height: this.cam.height };
    }

    // The grid labels at the same pixel offsets as the labels of the real image.
    drawLabels() {
      const ctx = this.overlay.getContext("2d");
      const W = this.cam.width, H = this.cam.height;
      const r = this.overlay.width / W;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
      if (!this.grid) return;
      ctx.setTransform(r, 0, 0, r, 0, 0);
      ctx.fillStyle = "#000";
      ctx.font = "10px sans-serif";
      ctx.textBaseline = "hanging";
      const fmt = (v) => (Math.abs(v) < 1e-9 ? 0 : v).toFixed(1);
      const cam = this.cameraNow();
      const px = (x, y) => window.CamProj.project(cam, [x, y, this.grid.z || 0]);
      const g = this.grid;
      for (const x of g.x) {
        const a = px(x, g.y[0]);
        if (a) ctx.fillText(`x=${fmt(x)}`, a[0] - 34, a[1] - 9);
      }
      for (const y of g.y) {
        const a = px(g.x[0], y);
        if (a) ctx.fillText(`y=${fmt(y)}`, a[0] - 14, a[1] - 17);
      }
    }

    // Drag to turn the view about the point it looks at. The wheel zooms. A double click goes back.
    bindOrbit() {
      const c = this.canvas;
      let last = null;
      this.onDown = (e) => {
        last = [e.clientX, e.clientY];
        c.setPointerCapture(e.pointerId);
      };
      this.onMove = (e) => {
        if (!last) return;
        const dx = e.clientX - last[0], dy = e.clientY - last[1];
        last = [e.clientX, e.clientY];
        this.orbit(-dx * 0.008, dy * 0.008, 1);
      };
      this.onUp = () => { last = null; };
      this.onWheel = (e) => {
        e.preventDefault();
        this.orbit(0, 0, e.deltaY > 0 ? 1.1 : 1 / 1.1);
      };
      this.onDbl = () => this.resetCamera();
      c.addEventListener("pointerdown", this.onDown);
      c.addEventListener("pointermove", this.onMove);
      c.addEventListener("pointerup", this.onUp);
      c.addEventListener("wheel", this.onWheel, { passive: false });
      c.addEventListener("dblclick", this.onDbl);
    }

    orbit(dAz, dEl, zoom) {
      const off = this.camera.position.clone().sub(this.target);
      const r = off.length() * zoom;
      let az = Math.atan2(off.y, off.x) + dAz;
      let el = Math.asin(Math.max(-1, Math.min(1, off.z / off.length()))) + dEl;
      el = Math.max(5 * RAD, Math.min(88 * RAD, el));
      if (!this.orbiting && Math.abs(off.z / off.length()) > 0.999) az = Math.atan2(-this.cam.up[1], -this.cam.up[0]);
      this.orbiting = true;
      this.camera.up.set(0, 0, 1);
      this.camera.position.set(this.target.x + r * Math.cos(el) * Math.cos(az), this.target.y + r * Math.cos(el) * Math.sin(az),
                               this.target.z + r * Math.sin(el));
      this.camera.lookAt(this.target);
      this.dirty = true;
    }

    dispose() {
      const c = this.canvas;
      c.removeEventListener("pointerdown", this.onDown);
      c.removeEventListener("pointermove", this.onMove);
      c.removeEventListener("pointerup", this.onUp);
      c.removeEventListener("wheel", this.onWheel);
      c.removeEventListener("dblclick", this.onDbl);
      this.scene.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) o.material.dispose();
      });
    }
  }

  // ---- The player: one clock for all views ----
  class Player {
    // slots: [{ canvas, overlay, camera: the camera name of the view }]
    constructor(slots) {
      this.slots = slots.map((s) => ({ ...s, renderer: new T.WebGLRenderer({ canvas: s.canvas, antialias: true }) }));
      this.views = [];
      this.view = null;
      this.time = 0;
      this.playing = false;
      this.mode = "imagined";
      this.listeners = [];
      this.raf = null;
      this.lastNow = null;
    }

    load(view) {
      this.unload();
      this.view = view;
      const byName = Object.fromEntries(view.cameras.map((c) => [c.name, c]));
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      for (const slot of this.slots) {
        const cam = byName[slot.camera];
        if (!cam) continue;
        slot.renderer.setPixelRatio(dpr);
        slot.renderer.setSize(cam.width, cam.height, false);
        slot.overlay.width = Math.round(cam.width * dpr);
        slot.overlay.height = Math.round(cam.height * dpr);
        this.views.push(new CameraView(slot, cam, view));
      }
      this.carry = carriers(view);
      this.setMode(this.mode);
      this.seek(0);
      this.loop();
    }

    get duration() {
      return this.view ? this.view.track.duration : 0;
    }

    setMode(mode) {
      this.mode = mode;
    }

    seek(t) {
      if (!this.view) return;
      this.time = Math.min(Math.max(0, t), this.duration);
      const track = this.view.track;
      const agent = agentAt(track, this.time);
      const believed = posesAt(track, this.view.believed, this.time, this.carry.believed, this.carry.believedMoves);
      for (const v of this.views) v.update(agent, believed);
      const kf = track.keyframes[agent.index];
      for (const fn of this.listeners) fn({ time: this.time, duration: this.duration, index: agent.index, keyframe: kf, playing: this.playing });
    }

    play() {
      if (!this.view) return;
      if (this.time >= this.duration) this.seek(0);
      this.playing = true;
      this.lastNow = null;
      this.seek(this.time);
    }

    pause() {
      this.playing = false;
      this.seek(this.time);
    }

    toggle() {
      if (this.playing) this.pause();
      else this.play();
    }

    onTick(fn) {
      this.listeners.push(fn);
    }

    resetViews() {
      for (const v of this.views) v.resetCamera();
    }

    // Stop the clock and the drawing while the views are hidden. wake() draws again.
    sleep() {
      this.playing = false;
      if (this.raf !== null) cancelAnimationFrame(this.raf);
      this.raf = null;
      if (this.view) this.seek(this.time);  // the tick listeners show the pause
    }

    wake() {
      if (!this.view || this.raf !== null) return;
      for (const v of this.views) v.dirty = true;
      this.seek(this.time);
      this.loop();
    }

    loop() {
      const step = (now) => {
        if (!this.view) return;
        if (this.playing) {
          if (this.lastNow !== null) {
            const t = this.time + (now - this.lastNow) / 1000;  // real time
            if (t >= this.duration) {
              this.playing = false;
              this.seek(this.duration);
            } else {
              this.seek(t);
            }
          }
          this.lastNow = now;
        }
        for (const v of this.views) v.render();
        this.raf = requestAnimationFrame(step);
      };
      this.raf = requestAnimationFrame(step);
    }

    unload() {
      if (this.raf !== null) cancelAnimationFrame(this.raf);
      this.raf = null;
      for (const v of this.views) v.dispose();
      this.views = [];
      this.view = null;
      this.playing = false;
      this.time = 0;
    }
  }

  window.Plan3D = {
    available() {
      try {
        T = window.THREE || null;
        const c = document.createElement("canvas");
        return Boolean(T && (c.getContext("webgl2") || c.getContext("webgl")));
      } catch (err) {
        return false;
      }
    },
    loaded() {
      return Boolean(window.THREE);
    },
    player(slots) {
      T = window.THREE;
      return new Player(slots);
    },
    agentAt,
    posesAt,
    carriers,
    wrapDeg,
  };
})();
