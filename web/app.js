const $ = (id) => document.getElementById(id);
let tasks = [];       // the tasks of the chosen scene
let scenes = [];
let sceneName = "cubes";
const DEFAULT_CAMERAS = { front: "Front", top: "Top · grid" };
let cameras = { ...DEFAULT_CAMERAS };  // the image labels of the chosen scene
let runCameras = { ...DEFAULT_CAMERAS };  // the image labels of the scene of the last run, for its step details

// Plan first: the camera boxes show either the camera images or the imagined video of a plan.
const PLAN_BADGES = { imagined: "Imagined" };
const XY_LIMIT_CM = 2;     // the plan check marks a larger error red
const YAW_LIMIT_DEG = 15;
const PRED_LIMIT_CM = 3;    // the Prediction entry marks a larger displacement error red
let viewMode = "camera";   // "camera", or the plan mode "imagined"
let player = null;         // the Plan3D player, made on the first plan
let playerFailed = false;  // no WebGL: the page says so and keeps the camera images
let planRunId = null;      // the id of the plan run on the page, for Execute and Re-plan
let planRound = 1;         // the round of the plan on the page: Re-plan works up to MAX_ROUND
let planMode = null;       // the mode of the plan on the page: "continue", "start", or null in round 1
let planInstruction = "";  // the newest instruction of the chain on the page: another one in the box is new
let ownStart = false;      // this tab posted a plan or a re-plan: its run_start keeps the instruction box
let planExecuted = false;  // the plan on the page has an execution: its Plan from here is after the execution
let planDone = false;      // Claude's re-plan on the page says that the task is done
let planImpossible = "";  // Claude's plan on the page says that the task is not possible: the reason
let parentCheck = null;    // the auto-check of the execution before this re-plan: true, false or null
let execError = false;     // the last execution of the plan on the page ended with an error: it has no end state
let execPartial = false;   // the last execution ran only its first keyframes: a re-plan always continues
const MAX_ROUND = 30;
let autoLoop = false;      // a live Auto loop: the page stays busy from round to round until loop_end
// One chain of rounds is one chat. The step maps and the entries of the chat are keyed by round and step.
let chatRound = 1;         // the round of the events that the chat gets now
let roundFeedback = {};    // round -> the end images of the last round that its re-plan sent, as server URLs
let runFrames = {};       // run id -> the frames of its last execution that went to Claude
let roundFrames = {};     // round -> the frames of the last round that its re-plan sent
let lastImages = null;     // the newest camera images of the chat, for a saved run
let replayShowsPlan = true; // false while loadRun replays an older round of a chain: only the last video loads

function cameraLabels(name) {
  const scene = scenes.find((s) => s.name === name);
  return { ...DEFAULT_CAMERAS, ...((scene && scene.cameras) || {}) };
}

function el(tag, text = "", cls = "") {
  const e = document.createElement(tag);
  if (text) e.textContent = text;
  if (cls) e.className = cls;
  return e;
}

function li(text, cls = "") {
  return el("li", text, cls);
}

// The pill classes: "" (idle), "thinking", "moving", "success", "bad".
function setStatus(text, cls = "") {
  $("status").textContent = text;
  $("status").className = `pill ${cls}`.trim();
}

function fmtTok(n) {
  return n >= 10000 ? `${(n / 1000).toFixed(0)}k` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
}

// ---- Examples and the auto-check line ----
function matchTask() {
  const text = $("instruction").value.trim();
  return tasks.find((x) => x.instruction === text);
}

function showMode() {
  const t = matchTask();
  $("mode").textContent = t ? `Auto-check: ${t.name}` : "No auto-check: Claude says when it is done";
  $("mode").classList.toggle("check", Boolean(t));
  for (const chip of $("examples").children) chip.classList.toggle("active", t ? chip.dataset.name === t.name : chip.dataset.name === "");
  showReplanLabels();
}

function chip(label, name, instruction) {
  const b = el("button", label, "chip");
  b.type = "button";
  b.dataset.name = name;
  b.addEventListener("click", () => {
    $("instruction").value = instruction;
    showMode();
    $("instruction").focus();
  });
  return b;
}

// ---- Scenes: the chosen scene sets the examples, the auto-check and the preview ----
const SCENE_KEY = "armmind.scene";

function readScene() {
  try {
    return localStorage.getItem(SCENE_KEY);
  } catch (err) {
    return null;
  }
}

function writeScene(name) {
  try {
    localStorage.setItem(SCENE_KEY, name);
  } catch (err) {
    // storage is blocked: the choice lasts until the page closes
  }
}

// keepText: true when a run starts, so that its instruction stays in the box.
// save: false when the page shows an earlier run, so that the chosen scene stays in storage.
function chooseScene(name, keepText = false, save = true) {
  const scene = scenes.find((s) => s.name === name) || scenes[0];
  if (!scene) return;
  const text = $("instruction").value.trim();
  if (!keepText && tasks.some((t) => t.instruction === text)) $("instruction").value = "";  // an example of the old scene
  sceneName = scene.name;
  tasks = scene.tasks;
  if (save) writeScene(sceneName);
  for (const c of $("scenes").children) c.classList.toggle("active", c.dataset.name === sceneName);
  $("examples").replaceChildren(chip("Your own", "", ""), ...tasks.map((t) => chip(t.name, t.name, t.instruction)));
  cameras = cameraLabels(scene.name);
  for (const cam of ["front", "top"]) $(`cam_${cam}`).querySelector(".cam-label").textContent = cameras[cam];
  showMode();
  loadLessons();
}

function sceneChip(s) {
  const b = el("button", s.title, "chip");
  b.type = "button";
  b.dataset.name = s.name;
  b.title = s.description;
  b.disabled = running;
  b.addEventListener("click", () => {
    if (running || s.name === sceneName) return;
    chooseScene(s.name);
    loadPreview();
  });
  return b;
}

async function loadScenes() {
  try {
    const r = await fetch("/api/scenes");
    if (!r.ok) throw new Error(r.statusText);
    scenes = await r.json();
  } catch (err) {
    $("mode").textContent = "could not load scenes";
    $("mode").classList.remove("check");
    return;
  }
  $("scenes").replaceChildren(...scenes.map(sceneChip));
  chooseScene(readScene() || "cubes");
}

$("instruction").addEventListener("input", showMode);
// Enter in the instruction submits the form, so it runs Plan. Shift+Enter starts a new line.
$("instruction").addEventListener("keydown", (e) => {
  if (e.key !== "Enter" || e.shiftKey || e.isComposing) return;
  e.preventDefault();
  $("run").requestSubmit();
});

let running = false;

// The scene chips are off while a run is active: the run keeps its scene.
function setRunning(value) {
  running = value;
  for (const c of $("scenes").children) c.disabled = value;
  $("shuffle").disabled = value;
  $("reset").disabled = value;
  for (const b of document.querySelectorAll(".exec-btn")) {
    b.disabled = value || !planRunId || b.dataset.run !== planRunId;  // the buttons of an older round stay off
  }
}
let previewSeq = 0;

// ---- Run totals ----
let stepImages = {};   // step number -> { front, top }: object URLs of the PNG images of the observation events

// A live event has the PNG as base64 data. A saved run has a URL of the server for it.
function imageSrc(value) {
  if (!value) return "";
  return value.startsWith("/api/runs/") ? value : pngUrl(value);
}

function pngUrl(b64) {  // one Blob and one object URL for each image, shared by the chat and the step dialog
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return URL.createObjectURL(new Blob([bytes], { type: "image/png" }));
}
let runSystemPrompt = "";
let totals;

function resetTotals() {
  totals = { steps: 0, input: 0, out: 0, cost: 0, time: 0 };
  showTotals();
}

function addTotals(u) {
  totals.steps += 1;
  if (u) {
    totals.input += (u.in || 0) + (u.cache_write || 0) + (u.cache_read || 0);
    totals.out += u.out || 0;
    totals.cost += u.cost_usd || 0;
    totals.time += u.latency_s || 0;
  }
  showTotals();
}

function stat(value, label) {
  const s = el("span", "", "stat");
  s.append(el("b", value), el("span", label));
  return s;
}

function showTotals() {
  const avg = totals.steps ? `${(totals.time / totals.steps).toFixed(1)}s` : "–";
  $("totals").replaceChildren(
    stat(String(totals.steps), "steps"),
    stat(fmtTok(totals.input), "in"),
    stat(fmtTok(totals.out), "out"),
    stat(`$${totals.cost.toFixed(3)}`, "cost"),
    stat(avg, "per step"),
  );
}
resetTotals();

// ---- Camera badges ----
// "preview" | "live" | "ended" | "run" (the images of an earlier run, with the label)
let camState = "preview";
let runKey = 0;     // a new run, a replay or a preview adds 1
let imageKey = -1;  // the runKey when the camera images last changed
let camLabel = "";
let outcome = null;  // the badge of the last execution, until the next Plan, Execute, Reset, scene or seed

function setCamState(state, label = "") {
  camState = state;
  camLabel = label;
  if (state === "preview" || state === "live") outcome = null;
  paintBadges();
}

// The keyframe of a stop: { k, n }, or null. N is the keyframe count of the plan. A run from before v1.16 has
// no keyframe in "stopped": its stop frame gives it. Older runs have neither.
function stopKeyframe(ev) {
  const s = ev.stopped;
  if (!s) return null;
  const frame = (Array.isArray(ev.frames) ? ev.frames : []).find((f) => f.at === "stop");
  if (typeof s.keyframe === "number") return { k: s.keyframe, n: ev.keyframes || (frame && frame.of) || s.keyframe };
  return frame ? { k: frame.keyframe, n: frame.of } : null;
}

// "Stopped in keyframe 6 of 7 (action 9 of 10): <reason>", or by action alone for an old run.
function stopText(ev) {
  const s = ev.stopped, kf = stopKeyframe(ev);
  return kf ? `Stopped in keyframe ${kf.k} of ${kf.n} (action ${s.after} of ${s.of}): ${s.reason}`
    : `Stopped after action ${s.after} of ${s.of}: ${s.reason}`;
}

// The outcome of an execution for the camera badge and the Execution entry: { text, cls, title, edge }.
// cls is "bad", "ok", "muted" or "plain". edge: a red edge on both camera frames.
function outcomeBadge(ev) {
  const s = ev.stopped;
  if (ev.status === "stopped") {
    const kf = stopKeyframe(ev);
    const text = kf ? `■ STOPPED · keyframe ${kf.k}/${kf.n}` : s ? `■ STOPPED · action ${s.after}/${s.of}` : "■ STOPPED";
    return { text, cls: "bad", edge: true, title: s ? `${stopText(ev)}.` : "The plan stopped." };
  }
  if (ev.status === "partial") {
    return { text: `◐ RAN ${ev.ran_keyframes}/${ev.keyframes} KEYFRAMES`, cls: "info", edge: false,
             title: `Run per round: the first ${ev.ran_keyframes} of ${ev.keyframes} keyframes ran. A re-plan continues from here.` };
  }
  if (ev.status === "cancelled") return { text: "■ CANCELLED", cls: "muted", edge: false, title: "The person stopped the execution." };
  if (ev.status === "success") return { text: "✓ SOLVED", cls: "ok", edge: false, title: "The auto-check says the task is solved." };
  if (ev.status === "done") return { text: "DONE", cls: "plain", edge: false, title: "All actions ran. There is no auto-check." };
  if (ev.status === "failed") return { text: "✗ NOT SOLVED", cls: "bad", edge: false, title: "All actions ran. The auto-check says not solved." };
  return { text: "ERROR", cls: "bad", edge: false, title: ev.message || "The execution failed with an error." };
}

// The badge says what the boxes show: the plan video, or the camera images. The Camera mode shows the outcome.
function paintBadges() {
  const plan = viewMode !== "camera";
  const out = !plan && outcome && (camState === "ended" || camState === "run") ? outcome : null;
  const text = plan ? PLAN_BADGES[viewMode]
    : out ? out.text : ({ live: "Live", ended: "Last frame", run: camLabel }[camState] || "Preview");
  for (const id of ["cam_front", "cam_top"]) {
    $(id).classList.toggle("live", !plan && camState === "live");
    $(id).classList.toggle("imagined", plan);
    $(id).classList.toggle("stop-edge", Boolean(out && out.edge));
    const badge = $(id).querySelector(".cam-badge");
    badge.textContent = text;
    badge.className = `cam-badge${out ? ` out-${out.cls}` : ""}`;
    badge.title = out ? out.title : "";
  }
}

function showImage(id, src) {
  const was = $(id).hidden, wasKey = imageKey;
  $(id).src = src;
  imageKey = runKey;
  $(id).hidden = viewMode !== "camera";  // the image waits behind the plan video
  if ($(id).hidden !== was || imageKey !== wasKey) drawPath();  // the path shows only over an image of its run
}

function clearPreviewError() {  // a good preview: drop the text of an earlier error
  for (const fig of ["cam_front", "cam_top"]) $(fig).querySelector(".placeholder").textContent = "";
}

function showPreviewError(message) {  // hide the images and put the message where they were
  for (const [fig, img] of [["cam_front", "live"], ["cam_top", "top"]]) {
    $(img).hidden = true;
    $(fig).querySelector(".placeholder").textContent = message;
  }
  drawPath();
}

// ---- Step details ----
function kvTiles(rows) {
  const box = el("div", "", "tiles");
  for (const [k, v] of rows) {
    const t = el("div", "", "tile");
    t.append(el("span", k), el("b", String(v)));
    box.append(t);
  }
  return box;
}

function usageLine(u) {
  if (!u) return "no usage reported";
  return `in ${u.in ?? "?"} · cache write ${u.cache_write ?? "?"} · cache read ${u.cache_read ?? "?"} · ` +
    `out ${u.out ?? "?"} · $${(u.cost_usd ?? 0).toFixed(4)} · ${(u.latency_s ?? 0).toFixed(1)}s`;
}

function openStepDetails(ev, key) {
  const kindText = { step: "", plan: " · plan", bad_reply: " · bad reply" }[ev.kind] ?? " · model error";
  const round = Number(key.split(":")[0]);
  $("step_title").textContent = `${round > 1 ? `Round ${round} · step` : "Step"} ${ev.step}${kindText}`;
  const u = stepUsage(ev);
  const items = [el("h2", "Tokens (exact, from the API)")];
  if (u) {
    const total = (u.in || 0) + (u.cache_write || 0) + (u.cache_read || 0);
    const imageCount = stepImageList(key).length;
    const imageTokens = imageCount * Math.round((640 * 480) / 750);
    const imgShare = total ? Math.min(100, (imageTokens / total) * 100) : 0;
    const bar = el("div", "", "bar");
    const bi = el("i", "", "b-img");
    bi.style.width = `${imgShare}%`;
    const bt = el("i", "", "b-txt");
    bt.style.width = `${100 - imgShare}%`;
    bar.append(bi, bt);
    const legend = el("div", "", "legend");
    legend.append(el("span", `images ≈ ${imageTokens}`, "l-img"), el("span", `text ≈ ${Math.max(0, total - imageTokens)}`, "l-txt"));
    items.push(
      kvTiles([["input, uncached", u.in ?? "?"], ["cache write", u.cache_write ?? "?"], ["cache read", u.cache_read ?? "?"],
               ["output", u.out ?? "?"], ["cost", `$${(u.cost_usd ?? 0).toFixed(4)}`], ["time", `${(u.latency_s ?? 0).toFixed(1)} s`]]),
      el("h2", "Estimated split of the input (not exact)"),
      bar,
      legend,
      el("p", `images = ${imageCount} × round(640×480/750) = ${imageCount} × ${Math.round((640 * 480) / 750)} = ${imageTokens}; text = total input (${total}) − images`, "muted"),
    );
  } else {
    items.push(el("p", "no usage reported", "muted"));
  }
  const sys = el("details");
  sys.append(el("summary", "System prompt"), el("pre", runSystemPrompt));
  items.push(sys, el("h2", "User message"), el("pre", ev.prompt ?? ""));
  const imgs = stepImageList(key);
  if (imgs.length) {
    const pair = el("div", "", "pair");
    for (const [name, data] of imgs.map(([label, src], i) => [`Image ${i + 1} · ${label}`, src])) {
      const fig = el("figure");
      const img = document.createElement("img");
      img.src = data;
      img.alt = name;
      fig.append(img, el("figcaption", name));
      pair.append(fig);
    }
    items.push(pair);
  }
  items.push(el("h2", "Model calls"));
  (ev.attempts || []).forEach((a, i) => {
    const box = el("div", "", "attempt");
    box.append(el("strong", `Call ${i + 1} · ${a.images} image${a.images === 1 ? "" : "s"} sent`), el("pre", a.message));
    if (a.reply !== null && a.reply !== undefined) box.append(el("strong", "Reply"), el("pre", a.reply));
    if (a.error) box.append(el("strong", "Error", "bad"), el("pre", a.error));
    box.append(el("div", usageLine(a.usage), "muted"));
    items.push(box);
  });
  $("step_body").replaceChildren(...items);
  $("step_body").scrollTop = 0;
  $("step_dialog").showModal();
}

$("step_close").addEventListener("click", () => $("step_dialog").close());

// ---- Chat ----
let stepEvents = {};     // step key -> the step, bad_reply or llm_error event, for the details dialog
let stepSent = {};       // step key -> the count of images that the first message of the step sent
let stepPrompts = {};    // step key -> the first message text, until the step event comes
let thinking = null;     // { item, timer, start } while Claude thinks
let runLessons = [];
let replaying = false;   // true while loadRun sends the events of a saved run

function atBottom() {
  const c = $("chat");
  return c.scrollHeight - c.scrollTop - c.clientHeight < 24;
}

function scrollChat(force = false) {
  const c = $("chat");
  if (force) c.scrollTop = c.scrollHeight;
  $("chat_latest").hidden = atBottom();
}

// The key of a step in the step maps: "round:step". Each round of a chain has its own step 1.
function stepKey(step) {
  return `${chatRound}:${step}`;
}

// Add an entry. Stay at the bottom when the reader is there. Otherwise do not move, and show "↓ latest".
function addEntry(item) {
  const empty = $("chat").querySelector(".chat-empty");
  if (empty) empty.remove();
  if (!item.classList.contains("round-div")) item.dataset.round = String(chatRound);
  if (replaying) return $("chat").appendChild(item);  // no layout read for each entry: loadRun scrolls once
  const stick = atBottom();
  $("chat").append(item);
  if (stick) scrollChat(true);
  else $("chat_latest").hidden = false;
  return item;
}

function entry(cls, title, step = null) {
  const item = el("li", "", `entry ${cls}`);
  const head = el("div", "", "entry-head");
  head.append(el("span", title, "who"));
  if (step !== null) head.append(el("span", `step ${step}`, "entry-step"));
  item.append(head);
  return item;
}

function openStep(key) {
  const step = Number(key.split(":")[1]);
  openStepDetails(stepEvents[key] || { kind: "step", step, prompt: stepPrompts[key] ?? "", attempts: [], usage: null }, key);
}

// The entry opens the details of a step. `key` is its step key.
function stepLink(item, key) {
  item.classList.add("link");
  item.tabIndex = 0;
  item.setAttribute("role", "button");
  item.addEventListener("click", (e) => {
    if (e.target.closest("button, summary")) return;  // "show all" and folds work on their own
    openStep(key);
  });
  item.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && e.target === item) {
      e.preventDefault();  // the focus moves to the dialog: the key must not press its Close button
      openStep(key);
    }
  });
  return item;
}

// A text block that shows its first 4 lines, with a "show all" button when it is longer.
function folded(text) {
  const box = el("div", "", "fold");
  const pre = el("pre", text, "msg");
  box.append(pre);
  if (text.split("\n").length > 4) {
    pre.classList.add("short");
    const more = el("button", "show all", "more");
    more.type = "button";
    more.addEventListener("click", () => {
      const short = pre.classList.toggle("short");
      more.textContent = short ? "show all" : "show less";
    });
    box.append(more);
  }
  return box;
}

// The images of a step message. A re-plan from the start also sends the end images of the last round.
function stepImageList(key) {
  const imgs = stepImages[key];
  const list = imgs ? [[runCameras.front, imgs.front], [runCameras.top, imgs.top]] : [];
  const [round, step] = key.split(":");
  const end = roundFeedback[round];
  if (step === "1") {  // the frames go before the end images, as in the message
    for (const f of roundFrames[round] || []) list.push([`${runCameras.front} · ${frameLabel(f)} of the last round`, f.src]);
  }
  if (step === "1" && end) {
    list.push([`${runCameras.front} · end of the last round`, end.front],
              [`${runCameras.top} · end of the last round`, end.top]);
  }
  // A chat that has the start images gets a re-plan without them: "Image 1" of the message is then the first image here.
  const sent = stepSent[key];
  if (typeof sent === "number" && sent === list.length - 2) list.splice(0, 2);
  return list;
}

function thumbs(key) {
  const step = key.split(":")[1];
  const row = el("div", "", "thumbs");
  for (const [label, data] of stepImageList(key)) {
    const img = document.createElement("img");
    img.loading = "lazy";  // a long run loads only the images on screen
    img.src = data;
    img.alt = `${label} image of step ${step}`;
    img.title = label;
    row.append(img);
  }
  return row;
}

function startThinking(step) {
  endThinking();
  const item = entry("thinking", "Claude is thinking…", step);
  const clock = el("span", "0 s", "clock");
  item.querySelector(".entry-head").append(clock);
  const start = performance.now();
  thinking = { item, start, timer: setInterval(() => {
    clock.textContent = `${Math.floor((performance.now() - start) / 1000)} s`;
  }, 250) };
  addEntry(item);
}

function endThinking() {
  if (!thinking) return;
  clearInterval(thinking.timer);
  thinking.item.remove();
  thinking = null;
}

function usageText(u) {
  if (!u) return "";
  const total = (u.in || 0) + (u.cache_write || 0) + (u.cache_read || 0);
  return `${fmtTok(total)} in · ${fmtTok(u.out || 0)} out · $${(u.cost_usd ?? 0).toFixed(4)} · ${(u.latency_s ?? 0).toFixed(1)} s`;
}

function field(label, node) {
  const row = el("div", "", "kv");
  row.append(el("span", label, "k"), node);
  return row;
}

function lines(items, cls = "") {
  const ul = el("ul", "", `lines ${cls}`.trim());
  ul.append(...items.map((t) => li(t)));
  return ul;
}

function onAsk(ev) {
  // The last message with images sets the image list of the step. A retry sends text only.
  if (typeof ev.images === "number" && ev.images > 0) stepSent[stepKey(ev.step)] = ev.images;
  if (ev.attempt === 1) {
    stepPrompts[stepKey(ev.step)] = ev.prompt;
    const item = entry("sent", "→ Sent", ev.step);
    item.append(folded(ev.prompt), thumbs(stepKey(ev.step)));
    addEntry(stepLink(item, stepKey(ev.step)));
  } else {
    endThinking();
    const item = entry("retry", "↻ Retry: the reply was not valid", ev.step);
    item.append(folded(ev.prompt));
    addEntry(stepLink(item, stepKey(ev.step)));
  }
  startThinking(ev.step);
}

function onReply(ev) {
  endThinking();
  const item = entry("claude", "← Claude", ev.step);
  if (ev.observation) item.append(field("saw", el("span", ev.observation)));
  item.append(field("plan", el("span", ev.plan || "(no plan)", "plan")));
  item.append(field("actions", ev.actions.length ? lines(ev.actions, "mono") : el("span", "none", "muted")));
  if (ev.notes) item.append(field("notes", el("span", ev.notes)));
  if (ev.todo && ev.todo.length) item.append(field("todo", lines(ev.todo)));
  if (ev.lessons && ev.lessons.length) item.append(field("lessons", lines(ev.lessons, "lesson-lines")));
  if (ev.usage) item.append(el("div", usageText(ev.usage), "entry-meta"));
  addEntry(stepLink(item, stepKey(ev.step)));
}

// "move_to(0.5, 0.1, 0.2): reached, gripper at …" -> action, status word, rest
function resultLine(text) {
  const row = el("li");
  const i = text.indexOf(": ");
  if (i < 0) {
    row.textContent = text;
    return row;
  }
  const rest = text.slice(i + 2);
  const word = rest.split(/[,\s]/)[0];
  row.append(el("span", text.slice(0, i), "act"), document.createTextNode(" "),
             el("span", word, `st-${word}`), document.createTextNode(rest.slice(word.length)));
  row.title = text;
  return row;
}

function onStepResults(ev) {
  const item = entry("res", "⚙ Results", ev.step);
  if (ev.results.length) {
    const ul = el("ul", "", "results");
    ul.append(...ev.results.map(resultLine));
    item.append(ul);
  } else {
    item.append(el("div", "no actions ran", "muted"));
  }
  if (ev.success) item.append(el("span", "Task solved", "badge success"));
  else if (ev.done) item.append(el("span", "Claude says done", "badge done"));
  if (ev.truncated) item.append(el("span", "Plan cut to 8 actions", "badge warn"));
  addEntry(stepLink(item, stepKey(ev.step)));
  if (ev.stopped) {
    const warn = entry("stopped", `⚠ Stopped after action ${ev.stopped.after} of ${ev.stopped.of}`, ev.step);
    warn.append(el("div", ev.stopped.reason));
    addEntry(warn);
  }
}

function errorEntry(ev, title, text) {
  endThinking();
  const item = entry("error", title, ev.step);
  item.append(el("div", text));
  if (ev.usage) item.append(el("div", usageText(ev.usage), "entry-meta"));
  addEntry(stepLink(item, stepKey(ev.step)));
}

const SAME_CHAT = "the same chat: Claude remembers the rounds above";
const NEW_CHAT_TEXT = "a new chat: Claude gets a summary of the last round";
let roundChatText = null;  // the chat text of the Plan started entry of this round, while it says "the same chat"

// chained: a later round of the chain in the chat. It continues the chat, so it has the same system prompt.
function onRunStart(ev, chained = false) {
  roundChatText = null;
  const item = entry("system", ev.mode === "plan" ? `Plan started${roundText(ev.round)}` : "Run started");
  const sys = el("details");
  sys.append(el("summary", "System prompt"), el("pre", ev.system_prompt || ""));
  if (!chained) item.append(field("task", el("span", ev.instruction)));
  if (ev.parent) {
    const how = ev.new_instruction
      ? "new instruction: the scene goes on from where that run left it, and Claude plans the new task"
      : ev.round_mode === "continue"
      ? "continue: the scene goes on from the end of that run, and Claude plans what is left"
      : "from start: the scene starts again, and Claude also sees the end images";
    // A round without a session to resume starts a new chat. A chat_lost event can also say so later.
    const same = chained && Boolean(ev.session_id);
    const chat = same ? SAME_CHAT : chained ? NEW_CHAT_TEXT : "Claude sees that plan and its result";
    const text = el("span", chat);
    if (same) roundChatText = text;
    const row = el("span");
    row.append(document.createTextNode(`of run ${ev.parent} · `), text, document.createTextNode(` · ${how}`));
    item.append(field("re-plan", row));
  }
  if (ev.auto_rounds && !chained) item.append(field("auto", el("span", `the Auto loop runs at most ${ev.auto_rounds} round${ev.auto_rounds === 1 ? "" : "s"}`)));
  if (!chained) item.append(sys);
  const saved = ev.saved_lessons || [];
  if (ev.use_lessons === false) item.append(el("div", "Saved lessons: off for this run", "muted"));
  else if (saved.length) {
    const d = el("details");
    d.append(el("summary", `Saved lessons given to Claude (${saved.length})`), lines(saved, "lesson-lines"));
    item.append(d);
  }
  addEntry(item);
}

function onRunEnd(ev) {
  endThinking();
  const ok = ev.status === "success" || ev.status === "done" || ev.status === "planned";
  const item = entry(ok ? "end ok" : "end", `■ ${ev.status === "planned" ? "Plan ready" : `Run ended: ${ev.status}`}`);
  if (ev.message) item.append(el("div", ev.message));
  addEntry(item);
}

function resetChat() {
  endThinking();
  for (const imgs of Object.values(stepImages)) for (const url of Object.values(imgs)) URL.revokeObjectURL(url);
  stepImages = {};
  stepEvents = {};
  stepPrompts = {};
  stepSent = {};
  stepStarts = {};
  roundFeedback = {};
  runFrames = {};
  roundFrames = {};
  lastImages = null;
  chatRound = 1;
  $("chat").replaceChildren();
  $("chat_latest").hidden = true;
}

// "── Round 2 · continue ──": the first line of a round in the chat. A click folds the round to this line.
// A round with a new instruction says "· new instruction", and the next line shows that instruction.
function roundDivider(round, mode, instruction = null) {
  const item = el("li", "", "round-div");
  item.dataset.round = String(round);
  const b = el("button", "", "round-btn");
  b.type = "button";
  b.setAttribute("aria-expanded", "true");
  b.append(el("span", `── Round ${round}${instruction ? " · new instruction" : modeText(mode)} ──`), el("span", "", "round-count"));
  b.addEventListener("click", () => foldRound(round, b.getAttribute("aria-expanded") === "true"));
  item.append(b);
  if (instruction) item.append(el("div", instruction, "round-instr"));
  return item;
}

function foldRound(round, fold) {
  const items = $("chat").querySelectorAll(`.entry[data-round="${round}"]`);
  for (const e of items) e.hidden = fold;
  const div = $("chat").querySelector(`.round-div[data-round="${round}"]`);
  if (!div) return;
  div.classList.toggle("folded", fold);
  div.querySelector("button").setAttribute("aria-expanded", String(!fold));
  div.querySelector(".round-count").textContent = fold ? `${items.length} entr${items.length === 1 ? "y" : "ies"} · show` : "";
}

// A later round of the chain on the page: the older rounds fold to one line each, and a divider starts this one.
function startRound(mode, instruction = null) {
  const shown = new Set([...$("chat").querySelectorAll(".entry[data-round]")].map((e) => e.dataset.round));
  for (const r of shown) {
    if (!$("chat").querySelector(`.round-div[data-round="${r}"]`)) {  // the first round of the chat has no divider yet
      $("chat").querySelector(`.entry[data-round="${r}"]`).before(roundDivider(Number(r), null));
    }
    foldRound(Number(r), true);
  }
  addEntry(roundDivider(chatRound, mode, instruction));
}

$("chat").addEventListener("scroll", () => {
  $("chat_latest").hidden = atBottom();
});
$("chat_latest").addEventListener("click", () => scrollChat(true));

// ---- Lessons ----
const LESSONS_KEY = "armmind.useLessons";
let savedLessons = [];

function readUseLessons() {
  try {
    return localStorage.getItem(LESSONS_KEY) !== "false";
  } catch (err) {
    return true;
  }
}

function writeUseLessons(on) {
  try {
    localStorage.setItem(LESSONS_KEY, on ? "true" : "false");
  } catch (err) {
    // storage is blocked: the choice lasts until the page closes
  }
}

$("use_lessons").checked = readUseLessons();
$("use_lessons").addEventListener("change", () => {
  writeUseLessons($("use_lessons").checked);
  showLessons();
});

function showLessons() {
  $("lessons").replaceChildren(...(runLessons.length ? runLessons.map((t) => li(t)) : [li("None in this run", "empty")]));
  const n = savedLessons.length;
  $("lessons_tag").textContent = String(n);
  $("lessons_saved").textContent = `${n} saved for this scene${$("use_lessons").checked ? "" : " · off"}`;
}

async function loadLessons() {
  try {
    const r = await fetch(`/api/lessons?scene=${encodeURIComponent(sceneName)}`);
    savedLessons = r.ok ? await r.json() : [];
  } catch (err) {
    savedLessons = [];
  }
  showLessons();
  if ($("lessons_dialog").open) fillLessonsDialog();
}

function fillLessonsDialog() {
  const scene = scenes.find((x) => x.name === sceneName);
  $("lessons_title").textContent = `Saved lessons · ${scene ? scene.title : sceneName}`;
  $("lessons_clear").disabled = !savedLessons.length;
  if (!savedLessons.length) {
    $("lessons_list").replaceChildren(li("No saved lessons for this scene", "empty"));
    return;
  }
  $("lessons_list").replaceChildren(...savedLessons.map((l, i) => {
    const row = el("li", "", "saved");
    const text = el("div", "", "saved-text");
    const when = l.t ? new Date(l.t * 1000).toLocaleString() : "";
    text.append(el("span", l.text), el("small", [l.outcome, when].filter(Boolean).join(" · "), "muted"));
    const del = el("button", "Delete", "btn ghost small");
    del.type = "button";
    del.setAttribute("aria-label", `Delete lesson ${i + 1}`);
    del.addEventListener("click", () => deleteLessons(i));
    row.append(text, del);
    return row;
  }));
}

// The text goes with the index: if the list changed in the meantime, the server refuses with 409.
async function deleteLessons(index = null) {
  let q = `scene=${encodeURIComponent(sceneName)}`;
  if (index !== null) q += `&index=${index}&text=${encodeURIComponent(savedLessons[index].text)}`;
  try {
    const r = await fetch(`/api/lessons?${q}`, { method: "DELETE" });
    $("lessons_msg").textContent = r.ok ? "" : r.status === 409 ? "The list changed or a run is active. Look again." : `error ${r.status}`;
  } catch (err) {
    $("lessons_msg").textContent = `error: ${err.message}`;
  }
  await loadLessons();
}

$("lessons_btn").addEventListener("click", () => {
  $("lessons_msg").textContent = "";
  fillLessonsDialog();
  $("lessons_dialog").showModal();
  loadLessons();
});
$("lessons_close").addEventListener("click", () => $("lessons_dialog").close());
$("lessons_clear").addEventListener("click", () => {
  if (confirm(`Delete all ${savedLessons.length} saved lessons of this scene?`)) deleteLessons(null);
});

function emptyMemo() {
  runLessons = [];
  showLessons();
}

// ---- Prompt editor ----
const PROMPT_KEY = "armmind.systemPrompt";
let defaultPrompt = "";
let savedPrompt = null;   // an edited prompt, or null for the default

function readSaved() {
  try {
    return localStorage.getItem(PROMPT_KEY);
  } catch (err) {
    return null;
  }
}

function writeSaved(text) {
  try {
    if (text === null) localStorage.removeItem(PROMPT_KEY);
    else localStorage.setItem(PROMPT_KEY, text);
  } catch (err) {
    // storage is blocked: the edit lasts until the page closes
  }
}

function showPromptTag() {
  $("prompt_tag").textContent = savedPrompt === null ? "default" : "edited";
  $("prompt_tag").classList.toggle("edited", savedPrompt !== null);
}

async function loadPrompt() {
  savedPrompt = readSaved();
  showPromptTag();
  try {
    const p = await (await fetch("/api/prompt")).json();
    defaultPrompt = p.system;
    $("prompt_example").textContent = p.step_example;
  } catch (err) {
    $("prompt_msg").textContent = "could not load the default prompt";
  }
  $("prompt_text").value = savedPrompt ?? defaultPrompt;
}

$("prompt_btn").addEventListener("click", () => {
  $("prompt_text").value = savedPrompt ?? defaultPrompt;
  $("prompt_msg").textContent = "";
  $("prompt_dialog").showModal();
});
$("prompt_close").addEventListener("click", () => $("prompt_dialog").close());
$("prompt_save").addEventListener("click", () => {
  const text = $("prompt_text").value;
  savedPrompt = text.trim() === "" || text === defaultPrompt ? null : text;
  writeSaved(savedPrompt);
  showPromptTag();
  $("prompt_msg").textContent = savedPrompt === null ? "Saved: using the default" : "Saved: only a step-by-step run of the API uses this prompt";
});
$("prompt_reset").addEventListener("click", () => {
  savedPrompt = null;
  writeSaved(null);
  $("prompt_text").value = defaultPrompt;
  showPromptTag();
  $("prompt_msg").textContent = "Reset to default";
});

// ---- Scene preview: the starting scene of the seed, unless a run is active ----
function setLoading(on) {
  for (const id of ["cam_front", "cam_top"]) $(id).classList.toggle("loading", on);
}

async function loadPreview() {
  runKey++;
  clearPlan();
  viewSeq++;  // drop an earlier run that is still on its way
  const seq = ++previewSeq;
  setLoading(true);
  try {
    await renderPreview(seq);
  } finally {
    if (seq === previewSeq) setLoading(false);  // a newer preview owns the spinner now
  }
}

async function renderPreview(seq) {
  try {
    const state = await (await fetch("/api/state")).json();
    autoLoop = Boolean(state.auto);
    setRunning(state.running);
    if (running) return;
    const q = `scene=${encodeURIComponent(sceneName)}&seed=${encodeURIComponent($("seed").value || 0)}`;
    const r = await fetch(`/api/preview?${q}`);
    if (seq !== previewSeq || running) return;
    if (r.status === 503) {  // files are missing: say which, and how to get them
      showPreviewError((await r.json()).detail);
      return;
    }
    if (!r.ok) return;
    const p = await r.json();
    if (seq !== previewSeq || running) return;
    clearPreviewError();
    showImage("live", `data:image/jpeg;base64,${p.front}`);
    showImage("top", `data:image/png;base64,${p.top}`);
    setCamState("preview");
  } catch (err) {
    // no preview is not an error: the images stay hidden
  }
}

let previewTimer = null;
$("shuffle").addEventListener("click", () => {
  if (running) return;
  $("seed").value = Math.floor(Math.random() * 100000);
  loadPreview();
});

// Reset: show the start layout of this scene and seed again, and clear the last run from the page.
// Saved runs and saved lessons stay.
$("reset").addEventListener("click", () => {
  if (running) return;
  resetChat();
  $("chat").replaceChildren(el("li", "Press Plan. Each message to Claude and each reply shows here.", "chat-empty"));
  emptyMemo();
  resetTotals();
  setStatus("idle");
  loadPreview();
});

$("seed").addEventListener("input", () => {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(loadPreview, 300);
});

// ---- Start and Stop ----
// Clear the page for a new run or plan. Reset before the POST: its events can arrive before the response.
function beginRun(text) {
  writeScene(sceneName);  // the scene of an earlier run on the page becomes the chosen scene
  viewSeq++;
  previewSeq++;  // drop a preview that is still on its way
  setLoading(false);
  clearPlan();
  runSystemPrompt = "";
  resetTotals();
  resetChat();
  emptyMemo();
  setStatus(text, "thinking");
}

async function postJson(url, body) {  // true when the server took it; else the status pill shows the error
  try {
    const r = await fetch(url, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined,
    });
    if (!r.ok) {
      const { detail } = await r.json().catch(() => ({}));
      const text = Array.isArray(detail) ? detail.map((d) => d.msg).join(", ") : (detail || r.statusText);
      setStatus(`error: ${text}`, "bad");
      return false;
    }
  } catch (err) {
    setStatus(`error: ${err.message}`, "bad");
    return false;
  }
  return true;
}

// The Plan button submits the form, and so does Enter in the instruction.
$("run").addEventListener("submit", (e) => {
  e.preventDefault();
  startPlan();
});

// Plan: the instruction, scene, seed and lessons switch. With Auto, the server loops over the rounds.
// A plan never gets the true positions.
async function startPlan() {
  if (running) {
    setStatus("a run is already active", "bad");
    return;
  }
  if (!$("run").reportValidity()) return;
  const body = {
    scene: sceneName,
    instruction: $("instruction").value.trim(),
    seed: Number($("seed").value),
    use_lessons: $("use_lessons").checked,
    auto: $("auto").checked,
    rounds: roundsValue($("rounds").value),
    mode: roundMode(),
    max_keyframes: perRound(),
  };
  beginRun("planning");
  ownStart = true;
  if (await postJson("/api/plan", body)) setRunning(true);
  else ownStart = false;
}

// The Rounds field as a whole number from 1 to 30 (MAX_ROUND). An empty or bad field gives 3.
function roundsValue(text) {
  const n = Math.round(Number(text));
  if (String(text).trim() === "" || !Number.isFinite(n)) return 3;
  return Math.min(MAX_ROUND, Math.max(1, n));
}

// ---- The Auto switch ----
const AUTO_KEY = "armmind.auto";

function readAuto() {
  try {
    return localStorage.getItem(AUTO_KEY) === "true";
  } catch (err) {
    return false;
  }
}

function writeAuto(on) {
  try {
    localStorage.setItem(AUTO_KEY, on ? "true" : "false");
  } catch (err) {
    // storage is blocked: the choice lasts until the page closes
  }
}

$("auto").checked = readAuto();
$("auto").addEventListener("change", () => writeAuto($("auto").checked));

// ---- The Continue switch: the mode of Re-plan and of the Auto loop after round 1 ----
const CONTINUE_KEY = "armmind.continue";

function readContinue() {
  try {
    return localStorage.getItem(CONTINUE_KEY) !== "false";  // on by default
  } catch (err) {
    return true;
  }
}

function writeContinue(on) {
  try {
    localStorage.setItem(CONTINUE_KEY, on ? "true" : "false");
  } catch (err) {
    // storage is blocked: the choice lasts until the page closes
  }
}

// "continue": a re-plan goes on from the end state of the last execution. "start": the scene starts again.
function roundMode() {
  return $("continue").checked ? "continue" : "start";
}

$("continue").checked = readContinue();
$("continue").addEventListener("change", () => writeContinue($("continue").checked));

// ---- Run per round: Execute runs only the first K keyframes, and each re-plan continues from there ----
function perRound() {
  const n = Math.round(Number($("per_round").value));
  return Number.isFinite(n) ? Math.min(MAX_ROUND, Math.max(0, n)) : 0;
}

const CONTINUE_TITLE = document.querySelector(".continue-switch").title;

// With K > 0, a re-plan must continue from where the K keyframes ended: Continue is on and locked.
function lockContinue() {
  const locked = perRound() > 0;
  if (locked) $("continue").checked = true;
  else $("continue").checked = readContinue();
  $("continue").disabled = locked;
  document.querySelector(".continue-switch").title = locked
    ? "Run per round is on: each re-plan continues from where the first keyframes ended. Set Run per round to 0 to choose."
    : CONTINUE_TITLE;
}

$("per_round").addEventListener("input", lockContinue);
lockContinue();

$("stop").addEventListener("click", () => fetch("/api/stop", { method: "POST" }));

// ---- Events from the server ----
// Old runs have a step event with cost_usd and latency_s, but no usage.
function stepUsage(ev) {
  if (ev.usage) return ev.usage;
  return typeof ev.cost_usd === "number" ? { cost_usd: ev.cost_usd, latency_s: ev.latency_s || 0 } : null;
}

// replay: true when the event comes from a saved run (see loadRun). It then changes no run state.
function onEvent(ev, replay = false) {
  if (ev.kind === "run_start") {
    outcome = null;
    // A re-plan of the plan on the page continues its chain: the same chat and the same totals.
    const chained = ev.mode === "plan" && Boolean(ev.parent) && ev.parent === planRunId;
    // A run of this tab, or a later round of its Auto loop, keeps the box. Any other run shows its instruction.
    const own = !replay && (ownStart || (autoLoop && chained));
    if (!replay) ownStart = false;
    runSystemPrompt = ev.system_prompt || "";
    runCameras = cameraLabels(ev.scene);
    if (!replay) {
      viewSeq++;  // a live run takes over the page from an earlier run
      setRunning(true);
    }
    if (ev.scene && ev.scene !== sceneName) chooseScene(ev.scene, true, !replay);
    runKey++;
    clearPlan();
    planRunId = ev.mode === "plan" ? ev.run_id : null;
    planRound = ev.round || 1;
    planMode = ev.round_mode || null;
    planInstruction = ev.instruction || "";
    if (!own) showInstruction();
    planExecuted = false;
    planDone = false;
    planImpossible = "";
    parentCheck = typeof ev.parent_check === "boolean" ? ev.parent_check : null;
    execError = false;
    execPartial = false;
    if (!replay) autoLoop = Boolean(ev.auto_rounds);
    runCams = ev.cameras || null;
    if (!chained) {
      resetChat();
      resetTotals();
    }
    chatRound = ev.mode === "plan" ? planRound : 1;
    if (chained) startRound(planMode, ev.new_instruction ? ev.instruction : null);
    // A round that continues sends no end images: its own images show the end of the last round.
    roundFeedback[chatRound] = ev.parent && ev.end_step && planMode !== "continue" ? endImages(ev.parent, ev.end_step) : null;
    roundFrames[chatRound] = (ev.parent && runFrames[ev.parent]) || null;
    emptyMemo();
    onRunStart(ev, chained);
    showReplanLabels();
    if (ev.mode === "plan") setStatus(`planning${roundText(planRound)} · Claude is thinking`, "thinking");
  } else if (ev.kind === "observation") {
    const key = stepKey(ev.step);
    const old = stepImages[key];
    if (old) for (const url of Object.values(old)) URL.revokeObjectURL(url);  // a server URL needs nothing
    stepImages[key] = lastImages = { front: imageSrc(ev.front_png), top: imageSrc(ev.top_png) };  // for the chat and the step details
    if (Array.isArray(ev.tcp)) stepStarts[key] = { xyz: ev.tcp, yaw: ev.yaw || 0 };
    setStatus(`step ${ev.step} · thinking`, "thinking");
  } else if (ev.kind === "ask") {
    onAsk(ev);
  } else if (ev.kind === "reply") {
    setStatus(`step ${ev.step} · moving`, "moving");
    onReply(ev);
    setPath(ev.step, runCams, stepStarts[stepKey(ev.step)] || null, ev.action_data);
  } else if (ev.kind === "action") {
    onPathAction(ev);
  } else if (ev.kind === "step") {
    stepEvents[stepKey(ev.step)] = ev;
    addTotals(stepUsage(ev));
    onStepResults(ev);
    onPathStep(ev);
    for (const t of ev.lessons || []) if (!runLessons.includes(t)) runLessons.push(t);
    showLessons();
  } else if (ev.kind === "plan") {
    roundChatText = null;  // the plan is in: a later note of a verdict does not change what the plan saw
    stepEvents[stepKey(ev.step)] = ev;
    addTotals(ev.usage);
    onPlan(ev);
    for (const t of ev.lessons || []) if (!runLessons.includes(t)) runLessons.push(t);
    showLessons();
    planView = ev.view || null;
    planDone = Boolean(ev.done);
    planImpossible = ev.impossible || "";
    addEntry(planCheckEntry(ev.scores || {}, planImpossible ? [] : ev.keyframes || [], Boolean(ev.done)));
    if (planImpossible) addEntry(impossibleEntry(planImpossible));
    if (ev.view && (!replay || replayShowsPlan)) showPlan(ev.view);
  } else if (ev.kind === "exec_start") {
    outcome = null;
    const own = planView && ev.run_id === planRunId;  // the cameras and the start pose of the plan on the page
    runCameras = cameraLabels(ev.scene);
    if (ev.scene && ev.scene !== sceneName) chooseScene(ev.scene, true, !replay);
    if (!replay) {
      viewSeq++;
      setRunning(true);
      if (ev.run_id !== planRunId) {  // another tab runs a plan that this page does not show
        resetChat();
        resetTotals();
        runKey++;
        clearPath();
        planRunId = ev.run_id;
        planRound = ev.round || 1;
        planMode = ev.round_mode || null;
        planInstruction = ev.instruction || "";
        showInstruction();
        chatRound = planRound;
      }
      setStatus("running the plan", "moving");
      setViewMode("camera");  // the live frames; a saved run keeps the imagined video
    }
    if (ev.run_id === planRunId) planExecuted = true;
    showReplanLabels();
    onExecStart(ev);
    if (own) setPath(ev.step, planView.cameras, planView.track && planView.track.start, ev.action_data, true);
  } else if (ev.kind === "exec_end") {
    if (!replay && !autoLoop && !ev.verdict_next) setRunning(false);  // a verdict call can follow
    if (!replay && $("history_dialog").open) fillHistory();
    if (ev.front_png && ev.top_png) {
      const key = stepKey(ev.step);
      const old = stepImages[key];  // a second Execute of the same plan has the same step
      if (old) for (const url of Object.values(old)) URL.revokeObjectURL(url);  // a server URL needs nothing
      stepImages[key] = lastImages = { front: imageSrc(ev.front_png), top: imageSrc(ev.top_png) };
    }
    outcome = outcomeBadge(ev);
    setCamState("ended");
    execError = ev.status === "error";
    execPartial = ev.status === "partial";
    const endItem = onExecEnd(ev);
    const pred = ev.prediction ? predictionEntry(ev.prediction, ev) : null;
    if (planRunId && planRound < MAX_ROUND) (pred || endItem).append(replanButton());
    if (pred) addEntry(pred);
    setStatus(execText(ev), ev.success ? "success" : ev.status === "error" ? "bad" : "");
    if (!replay && ev.verdict_next) setStatus(`${execText(ev)} · Claude judges the end images`, "thinking");
  } else if (ev.kind === "verdict") {
    if (!replay && !autoLoop) setRunning(false);
    addTotals(ev.usage);
    addEntry(verdictEntry(ev));
    if (!autoLoop) setStatus(verdictText(ev), ev.done ? "success" : "");
  } else if (ev.kind === "chat_lost") {
    if (roundChatText) roundChatText.textContent = NEW_CHAT_TEXT;  // the round does not continue the chat after all
    const item = entry("stopped", "⚠ New chat");
    item.append(el("div", ev.message || "This round starts a new chat with a summary."));
    if (ev.error) item.append(el("div", ev.error, "muted"));
    addEntry(item);
    if (thinking) addEntry(thinking.item);  // Claude still thinks: its entry moves below the note
  } else if (ev.kind === "loop_end") {
    if (!replay) {
      autoLoop = false;
      setRunning(false);
      if ($("history_dialog").open) fillHistory();
    }
    addEntry(loopEndEntry(ev));
    const last = ev.round ?? ev.rounds_run;  // the absolute round. A loop_end before v1.17 has only rounds_run
    const solved = ev.judge === "claude" ? `Claude says done in round ${last}` : `solved in round ${last}`;
    if (ev.impossible) setStatus("not possible", "bad");
    else setStatus(ev.success ? `Auto loop · ${solved}` : `Auto loop ended · ${ev.reason}`, ev.success ? "success" : "");
  } else if (ev.kind === "bad_reply") {
    stepEvents[stepKey(ev.step)] = ev;
    addTotals(ev.usage);
    errorEntry(ev, "✕ Bad reply: step skipped", ev.error);
  } else if (ev.kind === "llm_error") {
    stepEvents[stepKey(ev.step)] = ev;
    setStatus(`model error: ${ev.message}`, "bad");
    addTotals(ev.usage);
    errorEntry(ev, `✕ Model error (${ev.error_kind})`, ev.message);
  } else if (ev.kind === "run_end") {
    if (!replay && !autoLoop) setRunning(false);
    if (!replay && $("history_dialog").open) fillHistory();  // the rows can open again
    setCamState("ended");
    onRunEnd(ev);
    loadLessons();
    if (ev.status === "planned" && planImpossible) setStatus("not possible", "bad");
    else if (ev.status === "planned" && planDone) setStatus("Claude says the task is done", "success");
    else if (ev.status === "planned" && autoLoop) setStatus("plan ready · the Auto loop executes it next", "success");
    else if (ev.status === "planned") setStatus("plan ready · read the plan check, then execute", "success");
    else if (ev.status === "success") setStatus("success · task solved", "success");
    else if (ev.status === "done") setStatus("done · Claude says finished", "success");
    else setStatus(`ended: ${ev.status}${ev.message ? ` (${ev.message})` : ""}`, ev.status === "error" || ev.status === "llm_error" ? "bad" : "");
  }
}

// ---- Path overlay: the planned actions of a step on the camera images ----
const PATH_KEY = "armmind.showPath";
let runCams = null;    // the camera data of the run on the page, or null for an old run without it
let planView = null;   // the view of the plan on the page: its cameras and start pose serve Execute
let stepStarts = {};   // step number -> { xyz, yaw }: the gripper pose of the observation
let stepPath = null;   // { step, cameras, path, n, progress, groups } of the last reply or execution, or null

function readShowPath() {
  try {
    return localStorage.getItem(PATH_KEY) !== "false";
  } catch (err) {
    return true;
  }
}

function writeShowPath(on) {
  try {
    localStorage.setItem(PATH_KEY, on ? "true" : "false");
  } catch (err) {
    // storage is blocked: the choice lasts until the page closes
  }
}

$("show_path").checked = readShowPath();
$("show_path").addEventListener("change", () => {
  writeShowPath($("show_path").checked);
  drawPath();
});

const expectText = (e) => `→ ${String(e).replace(/_/g, " ")}`;

// The action types of the arm for the overlay: a move_to is a numbered point, an open or a close is a mark.
const PATH_SPEC = { point: "move_to", marks: { open: "open", close: "closed" },
                    label: (a) => (a.expect ? expectText(a.expect) : null) };

// key: the run that the path belongs to. The path shows only over an image of the same run.
// byKeyframe: a plan numbers its path by keyframe. The step mode numbers it by action.
function setPath(step, cams, start, actions, byKeyframe = false) {
  if (!(cams && actions && window.PathOverlay)) {
    stepPath = null;
    drawPath();
    return;
  }
  const groups = byKeyframe ? PathOverlay.pointGroups(actions, PATH_SPEC.point) : null;
  const list = PathOverlay.items(actions, start, PATH_SPEC);
  stepPath = { step, cameras: cams, n: actions.length, progress: {}, key: runKey, groups,
               path: { start, items: groups ? PathOverlay.grouped(list, groups) : list } };
  drawPath();
}

function clearPath() {
  stepPath = null;
  drawPath();
}

function onPathAction(ev) {
  if (!stepPath || stepPath.step !== ev.step) return;
  stepPath.progress.current = ev.index;
  drawPath();
}

// The step ended: the actions that ran are done, a stop turns its action red, and the rest is skipped.
function onPathStep(ev) {
  if (!stepPath || stepPath.step !== ev.step) return;
  stepPath.progress = PathOverlay.ended(stepPath.progress, ev.results, ev.stopped, (r) => r.endsWith(": skipped"));
  drawPath();
}

function drawPath() {
  const show = Boolean(stepPath && $("show_path").checked && viewMode === "camera" && imageKey === stepPath.key);
  let actionStates = show ? PathOverlay.states(stepPath.n, stepPath.progress) : [];
  if (show && stepPath.groups) actionStates = PathOverlay.groupStates(actionStates, stepPath.groups);
  for (const [name, img] of [["front", "live"], ["top", "top"]]) {
    const canvas = $(`path_${name}`);
    const cam = show && stepPath.cameras.find((c) => c.name === name);
    canvas.hidden = !cam || $(img).hidden;
    if (canvas.hidden) continue;
    // object-fit shows the image smaller than its pixels: scale the text and the lines back to screen size
    const box = canvas.getBoundingClientRect();
    const shown = Math.min(box.width, (box.height * cam.width) / cam.height);
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.round(cam.width * dpr), h = Math.round(cam.height * dpr);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    PathOverlay.draw(canvas, cam, stepPath.path, actionStates, shown > 0 ? cam.width / shown : 1);
  }
}

if (window.ResizeObserver) {  // a new box size changes the scale of the text
  const resized = new ResizeObserver(() => {
    if (stepPath) drawPath();
  });
  for (const id of ["cam_front", "cam_top"]) resized.observe($(id).querySelector(".frame"));
}

// ---- Plan first: the imagined video, the plan entries and Execute ----
function ensurePlayer() {
  if (player || playerFailed) return player;
  if (window.Plan3D && !Plan3D.loaded()) return null;  // three.js still loads: try again on the next plan
  try {
    if (!window.Plan3D || !Plan3D.available()) throw new Error("no WebGL");
    player = Plan3D.player([
      { canvas: $("plan_front"), overlay: $("plan_front_labels"), camera: "front" },
      { canvas: $("plan_top"), overlay: $("plan_top_labels"), camera: "top" },
    ]);
    player.onTick(showPlayerTime);
  } catch (err) {
    playerFailed = true;
    player = null;
  }
  return player;
}

function showPlan(view) {
  const p = ensurePlayer();
  if (!p) {  // no bar with buttons that do nothing: the chat says why
    $("player").hidden = true;
    const item = entry("system", "No imagined video");
    item.append(el("div", "The 3D view needs WebGL and three.js in this browser. The boxes keep the camera images."));
    addEntry(item);
    return;
  }
  $("player").hidden = false;
  p.load(view);
  $("play_slider").max = String(view.track.duration || 0);
  setViewMode("imagined");
  p.play();
}

// A new run, preview, scene or seed drops the plan. Its old Plan check entry can no longer execute.
function clearPlan() {
  if (player) player.unload();
  planRunId = null;
  planView = null;
  clearPath();
  for (const b of document.querySelectorAll(".exec-btn")) b.disabled = true;
  $("player").hidden = true;
  setViewMode("camera");
  showReplanLabels();
}

function setViewMode(mode) {
  viewMode = mode in PLAN_BADGES && player && player.view ? mode : "camera";
  const plan = viewMode !== "camera";
  for (const cam of ["front", "top"]) {
    $(`plan_${cam}`).hidden = !plan;
    $(`plan_${cam}_labels`).hidden = !plan;
  }
  for (const img of ["live", "top"]) $(img).hidden = plan || !$(img).getAttribute("src");
  if (plan) {
    player.setMode(viewMode);
    player.wake();
  } else if (player) {
    player.sleep();  // the hidden video stops: no clock and no drawing
  }
  for (const b of $("player").querySelectorAll(".seg button")) b.classList.toggle("active", b.dataset.mode === viewMode);
  paintBadges();
  drawPath();
}

function showPlayerTime({ time, duration, index, keyframe, playing }) {
  $("play_slider").value = String(time);
  $("play_time").textContent = `${time.toFixed(1)} / ${duration.toFixed(1)} s`;
  const n = player && player.view ? player.view.track.keyframes.length : 0;
  const moved = keyframe && (keyframe.moves || []).length ? ` · moves ${keyframe.moves.map((m) => m.name).join(", ")}` : "";
  $("play_kf").textContent = keyframe ? `${index + 1}/${n} · ${keyframe.note || "(no note)"}${moved}` : "";
  $("play_kf").title = $("play_kf").textContent;
  $("play_btn").textContent = playing ? "❚❚" : "▶";
}

$("play_btn").addEventListener("click", () => {
  if (!player || !player.view) return;
  if (viewMode === "camera") setViewMode("imagined");
  player.toggle();
});
$("play_slider").addEventListener("input", () => {
  if (!player || !player.view) return;
  const t = Number($("play_slider").value);  // read it first: pause() writes the old time back to the slider
  if (viewMode === "camera") setViewMode("imagined");
  player.pause();
  player.seek(t);
});
for (const b of $("player").querySelectorAll(".seg button")) b.addEventListener("click", () => setViewMode(b.dataset.mode));

const xyzText = (p) => `(${p.map((v) => v.toFixed(2)).join(", ")})`;
const signed = (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}`;

function onPlan(ev) {
  endThinking();
  const item = entry("claude", `← Claude · plan${roundText(planRound)}${modeText(planMode)}`, ev.step);
  item.append(field("plan", el("span", ev.summary || "(no summary)", "plan")));
  const objects = (ev.objects || []).map((o) => `${o.name}: ${o.shape} ${o.size.map((v) => (v * 100).toFixed(1)).join("×")} cm, ` +
    `centre ${xyzText(o.xyz)}, yaw ${Math.round(o.yaw)}°`);
  item.append(field("sees", objects.length ? lines(objects, "mono") : el("span", "no objects", "muted")));
  const keys = el("ol", "", "kf-lines");
  (ev.keyframes || []).forEach((k, i) => {
    const row = li("");
    const holds = k.carrying ? ` · holds ${k.carrying}` : "";
    const moves = (k.moves || []).length ? ` · moves ${movesText(k.moves)}` : "";
    row.append(el("span", String(i + 1), "n"), document.createTextNode(`${k.note || "(no note)"} · ${xyzText(k.xyz)} yaw ${Math.round(k.yaw)}° · ${k.state}${holds}${moves}`));
    keys.append(row);
  });
  const none = ev.impossible ? "none: Claude says the task is not possible" : "none: Claude says the task is done";
  item.append(field("keys", (ev.keyframes || []).length ? keys : el("span", none, "muted")));
  if (ev.warnings && ev.warnings.length) item.append(lines(ev.warnings, "mono"));
  if (ev.lessons && ev.lessons.length) item.append(field("lessons", lines(ev.lessons, "lesson-lines")));
  if (ev.usage) item.append(el("div", usageText(ev.usage), "entry-meta"));
  addEntry(stepLink(item, stepKey(ev.step)));
}

// The predicted moves of a keyframe in short form, for example "bowl → (0.62, 0.10)".
function movesText(moves) {
  return (moves || []).map((m) => `${m.name} → (${m.xyz[0].toFixed(2)}, ${m.xyz[1].toFixed(2)})`).join(", ");
}

function planCheckEntry(scores, keyframes = [], done = false) {
  const item = entry("plan", "◎ Plan check");
  if (done) {
    const last = parentCheck === null ? "" : ` The last auto-check says ${parentCheck ? "solved" : "not solved"}.`;
    item.append(el("div", `Claude says the task is done.${last}`, "done-note"));
  }
  const rows = scores.perception || [];
  if (rows.length) {
    const table = el("table", "", "ptable");
    const head = el("tr");
    for (const h of ["object", "xy", "z", "size (x y z)", "yaw"]) head.append(el("th", h));
    table.append(head);
    for (const r of rows) {
      const tr = el("tr");
      const sizeMax = Math.max(...r.size_cm.map(Math.abs));
      const cells = [
        [r.name, false],
        [`${r.xy_cm.toFixed(1)} cm`, r.xy_cm > XY_LIMIT_CM],
        [`${signed(r.z_cm)} cm`, Math.abs(r.z_cm) > XY_LIMIT_CM],
        [`${r.size_cm.map(signed).join(" ")} cm`, sizeMax > XY_LIMIT_CM],
        [r.yaw_deg === null ? "round" : `${r.yaw_deg.toFixed(0)}°`, r.yaw_deg !== null && r.yaw_deg > YAW_LIMIT_DEG],
      ];
      for (const [text, over] of cells) tr.append(el("td", text, over ? "over" : ""));
      tr.firstChild.title = `true object: ${r.true_name}`;
      table.append(tr);
    }
    item.append(table);
  } else {
    item.append(el("div", "No object of the belief matches a true object.", "muted"));
  }
  const s = scores.summary || {};
  item.append(el("div", `xy error: mean ${(s.mean_xy_cm ?? 0).toFixed(1)} cm, max ${(s.max_xy_cm ?? 0).toFixed(1)} cm · ` +
    `red: more than ${XY_LIMIT_CM} cm or ${YAW_LIMIT_DEG}°`, "muted"));
  if ((scores.missing || []).length) item.append(field("missing", el("span", scores.missing.join(", "))));
  if ((scores.invented || []).length) item.append(field("invented", el("span", scores.invented.join(", "))));
  const marks = el("ol", "", "kf-lines");
  let shownPredicted = new Set();  // say "at its predicted pose" once, on the first keyframe that uses it
  (scores.marks || []).forEach((m, i) => {
    const row = li("", m.ok ? "" : "bad");
    const why = Object.entries(m.checks || {}).filter(([, v]) => v !== "ok").map(([k, v]) => `${k}: ${v}`);
    row.append(el("span", m.ok ? "✓" : "✕", m.ok ? "mark-ok" : "mark-bad"), el("span", ` ${i + 1}`, "n"),
               document.createTextNode([...why, m.info].filter(Boolean).join("; ") || "ok"));
    const fresh = (m.predicted || []).filter((n) => !shownPredicted.has(n));
    shownPredicted = new Set(m.predicted || []);
    if (fresh.length) row.append(el("span", ` · from here, ${fresh.join(", ")} at the predicted pose`, "kf-muted"));
    marks.append(row);
  });
  if ((scores.marks || []).length) item.append(field("checks", marks));
  const moved = keyframes.map((k, i) => [i, k.moves || []]).filter(([, m]) => m.length)
    .map(([i, m]) => `${i + 1} · moves ${movesText(m)}`);
  if (moved.length) item.append(field("moves", lines(moved, "mono")));
  const fromHere = planRunId && planRound < MAX_ROUND ? replanButton(true) : null;
  if (fromHere && !keyframes.length) item.append(fromHere);
  if (!keyframes.length) return item;  // a done plan without keyframes has nothing to execute
  const exec = el("button", "▶ Execute plan", "btn exec-btn");
  exec.type = "button";
  exec.dataset.run = planRunId || "";
  exec.disabled = running;
  exec.title = planMode === "continue"
    ? "Run the keyframes once in the physics, from where the last round ended. No model is called."
    : "Run the keyframes once in the physics, from a fresh start of the same scene and seed. No model is called.";
  exec.addEventListener("click", executePlan);
  item.append(exec);
  if (fromHere) item.append(fromHere);
  return item;
}

async function executePlan() {
  if (running || !planRunId) return;
  setStatus("starting the plan", "moving");
  if (await postJson(`/api/plan/${encodeURIComponent(planRunId)}/execute`, { max_keyframes: perRound() })) setRunning(true);
}

function onExecStart(ev) {
  const item = entry("system", "▶ Executing the plan");
  const from = ev.round_mode === "continue" ? "from where the last round ended" : `from a fresh start of seed ${ev.seed}`;
  const part = ev.plan_keyframes && ev.plan_keyframes > ev.keyframes ? `The first ${ev.keyframes} of ${ev.plan_keyframes}` : String(ev.keyframes);
  item.append(el("div", `${part} keyframe${ev.keyframes === 1 && part === "1" ? "" : "s"} ${from}. ` +
    (ev.check ? `Auto-check: ${ev.check}.` : "No auto-check.")));
  addEntry(item);
}

function execText(ev) {
  if (ev.status === "success") return "plan executed · task solved";
  if (ev.status === "stopped" && ev.stopped) {
    const kf = stopKeyframe(ev);
    return kf ? `plan stopped in keyframe ${kf.k} of ${kf.n}` : `plan stopped at action ${ev.stopped.after} of ${ev.stopped.of}`;
  }
  if (ev.status === "failed") return "plan executed · the auto-check says not solved";
  if (ev.status === "cancelled") return "plan execution stopped";
  if (ev.status === "partial") return `ran ${ev.ran_keyframes} of ${ev.keyframes} keyframes · re-plan the rest`;
  if (ev.status === "error") return `error: ${ev.message || "the execution failed"}`;
  return "plan executed";
}

function onExecEnd(ev) {
  const item = entry(ev.success || ev.status === "done" ? "end ok" : "end", `■ Execution ended: ${ev.status}`);
  const tag = outcomeBadge(ev);
  const t = el("span", tag.text, `out-tag out-${tag.cls}`);
  t.title = tag.title;
  item.querySelector(".entry-head").append(t);
  item.append(el("div", ev.stopped ? `${stopText(ev)}.`
    : ev.success ? "The task is solved." : execText(ev)));
  const frames = (Array.isArray(ev.frames) ? ev.frames : []).map((f) => ({ ...f, src: frameSrc(f) }));
  if (planRunId) runFrames[planRunId] = frames.filter((f) => f.sent);
  if (frames.length) item.append(frameStrip(frames));
  return addEntry(item);
}

// ---- The frames of an execution: one front camera image at the end of each keyframe that ran ----
// A saved run gives the URL. A live event gives only the file name in the folder of the plan.
function frameSrc(f) {
  return f.url || `/api/runs/${encodeURIComponent(planRunId || "")}/images/${encodeURIComponent(f.file || "")}`;
}

function frameLabel(f) {  // "kf 2/5" or "stop 3/5"
  return `${f.at === "stop" ? "stop" : "kf"} ${f.keyframe}/${f.of}`;
}

function frameTitle(f) {
  return f.at === "stop" ? `at the stop in keyframe ${f.keyframe} of ${f.of}` : `after keyframe ${f.keyframe} of ${f.of}`;
}

// A small image for each frame, with its label. The images load only when the entry shows (a folded round hides it).
function frameStrip(frames) {
  const box = el("div", "", "kf-box");
  const row = el("div", "", "kf-strip");
  for (const f of frames) {
    const b = el("button", "", `kf-thumb${f.sent ? " sent" : ""}`);
    b.type = "button";
    b.title = `${runCameras.front} ${frameTitle(f)}${f.sent ? " · sent to Claude" : ""}`;
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = f.src;
    img.alt = b.title;
    b.append(img, el("span", frameLabel(f)));
    b.addEventListener("click", () => openFrame(f));
    row.append(b);
  }
  box.append(row);
  if (frames.some((f) => f.sent)) box.append(el("div", "A blue edge marks a frame that went to Claude.", "muted kf-note"));
  return box;
}

function openFrame(f) {
  $("frame_title").textContent = `${runCameras.front} camera ${frameTitle(f)}`;
  $("frame_img").src = f.src;
  $("frame_img").alt = $("frame_title").textContent;
  $("frame_note").textContent = `${f.note ? `Note: ${f.note} · ` : ""}${f.sent ? "sent to Claude" : "not sent to Claude"}`;
  $("frame_dialog").showModal();
}

$("frame_close").addEventListener("click", () => $("frame_dialog").close());

// " · round 2" for a later round, "" for round 1.
function roundText(round) {
  return round > 1 ? ` · round ${round}` : "";
}

// " · continue" or " · from start" for a re-plan, "" for round 1.
function modeText(mode) {
  return mode === "continue" ? " · continue" : mode === "start" ? " · from start" : "";
}

// The server URLs of the end images of an executed plan.
function endImages(runId, step) {
  const base = `/api/runs/${encodeURIComponent(runId)}/images/step${String(step).padStart(3, "0")}`;
  return { front: `${base}_front.png`, top: `${base}_top.png` };
}

// Re-plan: the same rules as Execute disable it (see clearPlan and setRunning).
const REPLAN_TITLE = "Claude sees this plan and its result. With Continue on, it plans the rest from where the " +
  "execution ended. With Continue off, it sees the end images and plans the whole task again from a fresh start.";
const FROM_HERE_TITLE = "Claude gets the new instruction in the same chat, and the arm goes on from where it is now";

// The text of the instruction box when it differs from the newest instruction of the chain on the page, else "".
function newInstruction() {
  const text = $("instruction").value.trim();
  return planRunId && text && text !== planInstruction ? text : "";
}

// "↻ Re-plan", or "▶ Plan from here" when the box has a new instruction. Only the buttons of the plan on the page.
// The button of a plan check (onlyNew) shows only for a new instruction, while the plan has no execution.
function replanLabel(b) {
  const fromHere = b.dataset.run === planRunId && Boolean(newInstruction());
  b.textContent = fromHere ? "▶ Plan from here" : "↻ Re-plan";
  b.title = fromHere ? FROM_HERE_TITLE : REPLAN_TITLE;
  if (b.dataset.onlyNew) b.hidden = !fromHere || planExecuted;
}

// The instruction of the run on the page into the box: a History load or a run of another tab.
function showInstruction() {
  $("instruction").value = planInstruction;
  showMode();
}

function showReplanLabels() {
  for (const b of document.querySelectorAll(".replan-btn")) replanLabel(b);
}

// onlyNew: the button of a plan check, for a new instruction before an execution (see replanLabel).
function replanButton(onlyNew = false) {
  const b = el("button", "↻ Re-plan", "btn exec-btn replan-btn");
  b.type = "button";
  b.dataset.run = planRunId || "";
  b.disabled = running || !planRunId;
  if (onlyNew) {
    b.dataset.onlyNew = "true";
    b.addEventListener("click", () => replan());
    replanLabel(b);
    return b;
  }
  replanLabel(b);
  const fromStart = execError;  // the mode when the button is made: a later Execute must not change it
  const partial = execPartial;  // a re-plan after a partial run always continues
  b.addEventListener("click", () => replan(fromStart, partial));
  if (!fromStart) return b;
  // An execution that failed with an error saved no end state, so a re-plan cannot continue from it.
  const box = el("div", "", "replan-box");
  box.append(b, el("div", "The execution failed, so this re-plan starts from the start, also with Continue on.", "muted"));
  return box;
}

// Plan from here: a new instruction in the box always continues, and with Auto on it starts the Auto loop.
async function replan(fromStart = false, partial = false) {
  if (running || !planRunId) return;
  const text = newInstruction();
  // No end state after an error: see replanButton. The server then refuses a new instruction.
  const mode = text ? "continue" : fromStart ? "start" : partial ? "continue" : roundMode();
  setStatus(text ? "planning from here · new instruction" : `re-planning · ${mode === "continue" ? "continue" : "from start"}`, "thinking");
  const body = { mode, max_keyframes: mode === "continue" ? perRound() : 0 };
  if (text) Object.assign(body, { instruction: text, auto: $("auto").checked, rounds: roundsValue($("rounds").value) });
  ownStart = true;
  if (await postJson(`/api/plan/${encodeURIComponent(planRunId)}/replan`, body)) setRunning(true);
  else ownStart = false;
}

// "Claude: done" or "Claude: not done", and the auto-check when there is one.
function verdictText(ev) {
  const claude = ev.done === true ? "Claude: done" : ev.done === false ? "Claude: not done" : "Claude: no verdict";
  if (ev.check === true || ev.check === false) return `${claude} · auto-check: ${ev.check ? "solved" : "not solved"}`;
  return claude;
}

// Claude's verdict on the end images of an execution, next to the auto-check result.
function verdictEntry(ev) {
  const agree = (ev.check === true || ev.check === false) && typeof ev.done === "boolean";
  const ok = ev.done === true && ev.check !== false;
  const item = entry(ok ? "end ok" : "end", `◎ ${verdictText(ev)}`);
  if (ev.reason) item.append(field("reason", el("span", ev.reason)));
  if (agree) item.append(el("div", ev.done === ev.check ? "Claude agrees with the auto-check." : "Claude and the auto-check disagree.", "muted"));
  if (ev.error) item.append(el("div", `The verdict call failed: ${ev.error}`, "muted"));
  if (ev.usage) item.append(el("div", usageText(ev.usage), "entry-meta"));
  return item;
}

// Claude stops the task: a red entry with the reason. Nothing executes.
function impossibleEntry(reason) {
  const item = entry("impossible", "✕ Claude stopped: not possible");
  item.append(el("div", reason));
  return item;
}

function loopEndEntry(ev) {
  const solved = ev.judge === "claude" ? "■ Auto loop: Claude says done" : "■ Auto loop: task solved";
  const gaveUp = ev.impossible && typeof ev.impossible === "object" ? ev.impossible : null;
  const item = entry(gaveUp ? "impossible" : ev.success ? "end ok" : "end",
    gaveUp ? "■ Auto loop: not possible" : ev.success ? solved : "■ Auto loop ended");
  item.append(el("div", gaveUp ? `Claude gave up in round ${gaveUp.round}: ${gaveUp.reason}`
    : `${ev.rounds_run} of ${ev.rounds_max} round${ev.rounds_max === 1 ? "" : "s"} run · ${ev.reason}.`));
  if (ev.judge) item.append(field("verdict", el("span", ev.judge === "claude" ? "Claude's own done verdict (no auto-check)" : "the auto-check")));
  const rounds = (ev.verdicts || []).map((v) => `round ${v.round} · Claude: ${v.claude ? "done" : "not done"}` +
    (v.check === true || v.check === false ? ` · auto-check: ${v.check ? "solved" : "not solved"}` : ""));
  if (rounds.length) item.append(field("judged", lines(rounds, "mono")));
  if (ev.compared) item.append(el("div", `Claude agrees with the auto-check in ${ev.agree} of ${ev.compared} execution${ev.compared === 1 ? "" : "s"}.`, "muted"));
  return item;
}

// The predicted moves of the plan against the physics. Each move starts from its own start pose, so the
// error of the belief stays out of the displacement error.
function predictionEntry(p, ev = {}) {
  const item = entry("plan", "◎ Prediction");
  const s = p.summary || {};
  if (ev.status === "partial") {
    item.append(el("div", `Run per round: keyframes 1 to ${ev.ran_keyframes} of ${ev.keyframes} ran by design. This compares them.`, "muted"));
  } else if (s.partial) {
    item.append(el("div", s.ran ? `The execution stopped early: this compares keyframes 1 to ${s.ran} of ${s.of}.`
      : "The execution stopped before the first keyframe ended: nothing to compare.", "muted"));
  }
  const rows = p.rows || [];
  if (rows.length || (p.unmatched || []).length) {
    const table = el("table", "", "ptable");
    const head = el("tr");
    for (const h of ["object", "predicted move", "real move", "error"]) head.append(el("th", h));
    table.append(head);
    for (const r of rows) {
      const tr = el("tr");
      const cells = [
        [r.name, false],
        [`${r.predicted_cm.map(signed).join(" ")} cm`, false],
        [`${r.real_cm.map(signed).join(" ")} cm`, r.agree === false],
        [`${r.disp_cm.toFixed(1)} cm`, r.disp_cm > PRED_LIMIT_CM],
      ];
      for (const [text, over] of cells) tr.append(el("td", text, over ? "over" : ""));
      tr.title = `true object: ${r.true_name} · final position error: xy ${r.final_xy_cm.toFixed(1)} cm, z ${signed(r.final_z_cm)} cm`;
      table.append(tr);
    }
    for (const u of p.unmatched || []) {
      const tr = el("tr");
      const td = el("td", `not in the belief: ${u.true_name} moved ${u.moved_cm.toFixed(1)} cm`, "over");
      td.colSpan = 4;
      tr.append(td);
      table.append(tr);
    }
    item.append(table);
    item.append(el("div", `moves are x y z · error: the xy distance between the two moves · red: more than ${PRED_LIMIT_CM} cm`, "muted"));
    item.append(el("div", `moved or not (xy more than 2 cm): ${s.agree} of ${s.count} agree · ` +
      `final position error: mean xy ${(s.mean_final_xy_cm ?? 0).toFixed(1)} cm`));
  } else if (s.ran) {
    item.append(el("div", "No object of the belief matches a true object.", "muted"));
  }
  const keyLines = (p.keyframes || []).map((k) => `${k.keyframe} · ${k.name}: ` +
    (k.disp_cm === null ? "no true object" : `${k.disp_cm.toFixed(1)} cm`));
  if (keyLines.length) item.append(field("moves", lines(keyLines, "mono")));
  return item;
}

// ---- History: read an earlier run on the page ----
let viewSeq = 0;  // a newer load, a preview, a Start or a live run drops an older load

function runTime(t) {
  if (!t) return "?";
  return new Date(t * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function sceneTitle(name) {
  const scene = scenes.find((s) => s.name === name);
  return scene ? scene.title : name;
}

function historyRow(run) {
  const active = run.active && running;  // after run_end the server can list the run as active for a moment
  const status = active ? "running" : run.status;
  const row = el("li");
  const b = el("button", "", "run-row");
  b.type = "button";
  b.append(
    el("span", runTime(run.started), "run-when"),
    el("span", sceneTitle(run.scene), "run-scene"),
    el("span", run.instruction || "(no instruction)", "run-text"),
    el("span", `${run.steps} step${run.steps === 1 ? "" : "s"}`, "run-num"),
    el("span", `$${run.cost_usd.toFixed(3)}`, "run-num"),
    el("span", (run.exec_status ? `run: ${run.exec_status}` : status).replace("_", " "), `run-badge s-${run.exec_status || status}`),
  );
  if (run.mode === "plan") b.querySelector(".run-text").prepend(el("span", "plan", "run-mode"));
  if (run.mode === "plan" && run.round > 1) {  // the parent id "20261018-183553" shows as "after 18:35:53"
    const round = el("span", `round ${run.round}`, "run-mode run-round");
    b.querySelector(".run-mode").after(round);
    const t = /^\d{8}-(\d\d)(\d\d)(\d\d)/.exec(run.parent || "");
    if (t) round.after(el("span", `after ${t[1]}:${t[2]}:${t[3]} `, "run-parent"));
  }
  b.title = run.parent ? `${run.instruction} · re-plan of run ${run.parent}` : run.instruction;
  if (active) {
    b.disabled = true;
    b.title = "This run is active. The page shows it live.";
  } else if (running) {
    b.disabled = true;
    b.title = "A run is active. Stop it to open an earlier run.";
  }
  b.addEventListener("click", () => {
    $("history_dialog").close();
    loadRun(run);
  });
  row.append(b);
  return row;
}

async function openHistory() {
  $("history_list").replaceChildren();
  $("history_dialog").showModal();
  await fillHistory();
}

async function fillHistory() {
  $("history_msg").textContent = "Loading…";
  try {
    const r = await fetch("/api/runs");
    if (!r.ok) throw new Error(`error ${r.status}`);
    const runs = await r.json();
    $("history_list").replaceChildren(...runs.map(historyRow));
    $("history_msg").textContent = !runs.length ? "No saved runs yet" : running ? "A run is active. Stop it to open an earlier run." : "";
  } catch (err) {
    $("history_msg").textContent = `could not load the runs: ${err.message}`;
  }
}

// Clear the page as Reset does, then send each saved event through the live handler.
async function loadRun(run) {
  if (running) return;
  const seq = ++viewSeq;
  const when = runTime(run.started);
  setStatus(`loading run ${when}`, "thinking");
  let runs;  // the event lists of the chain of the run, from round 1 to this run
  try {
    const c = await fetch(`/api/runs/${encodeURIComponent(run.id)}/chain`);
    const ids = c.ok ? await c.json() : [run.id];
    runs = await Promise.all(ids.map(async (id) => {
      const r = await fetch(`/api/runs/${encodeURIComponent(id)}`);
      if (!r.ok) throw new Error(`error ${r.status}`);
      return r.json();
    }));
  } catch (err) {
    if (seq === viewSeq) setStatus(`could not load the run: ${err.message}`, "bad");
    return;
  }
  if (seq !== viewSeq || running) return;
  previewSeq++;  // drop a preview that is still on its way
  setLoading(false);
  runSystemPrompt = "";
  resetChat();
  emptyMemo();
  resetTotals();
  let skipped = 0;  // a corrupt event is left out, and the rest of the run still shows
  replaying = true;
  try {
    runs.forEach((events, i) => {
      replayShowsPlan = i === runs.length - 1;  // the imagined video of the run that was opened
      for (const ev of events) {
        try {
          onEvent(ev, true);
        } catch (err) {
          skipped += 1;
        }
      }
    });
  } finally {
    replaying = false;
    replayShowsPlan = true;
  }
  endThinking();  // an unfinished run can end on a question
  scrollChat(true);
  if (lastImages) {
    clearPreviewError();
    showImage("live", lastImages.front);
    showImage("top", lastImages.top);
  }
  setCamState("run", `Run ${when}`);
  const ok = run.status === "success" || run.status === "done" || run.status === "planned";
  const note = skipped ? ` · ${skipped} bad event${skipped > 1 ? "s" : ""} left out` : "";
  setStatus(`viewing run ${when} · ${run.status}${note}`, ok && !skipped ? "success" : run.status === "error" || run.status === "llm_error" || skipped ? "bad" : "");
}

$("history_btn").addEventListener("click", openHistory);
$("history_close").addEventListener("click", () => $("history_dialog").close());

// Read the busy state and the Auto loop from the server, for example after the socket connects again.
async function syncState() {
  try {
    const state = await (await fetch("/api/state")).json();
    autoLoop = Boolean(state.auto);
    setRunning(state.running);
  } catch (err) {
    // the server is not there: the next connect tries again
  }
}

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => syncState();
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.type === "frame") {
      showImage(msg.cam === "top" ? "top" : "live", `data:image/jpeg;base64,${msg.jpg}`);
      if (running) setCamState("live");
    } else if (msg.type === "event") {
      onEvent(msg);
    }
  };
  ws.onclose = () => setTimeout(connect, 1000);
}

loadScenes().finally(loadPreview);  // the preview needs the chosen scene
loadPrompt();
connect();
