const $ = (id) => document.getElementById(id);
let tasks = [];       // the tasks of the chosen scene
let scenes = [];
let sceneName = "cubes";
const DEFAULT_CAMERAS = { front: "Front", top: "Top · grid" };
let cameras = { ...DEFAULT_CAMERAS };  // the image labels of the chosen scene
let runCameras = { ...DEFAULT_CAMERAS };  // the image labels of the scene of the last run, for its step details

// Plan first: the camera boxes show either the camera images or the imagined video of a plan.
const PLAN_BADGES = { imagined: "Imagined", real: "Real start", both: "Imagined + real" };
const XY_LIMIT_CM = 2;     // the plan check marks a larger error red
const YAW_LIMIT_DEG = 15;
let viewMode = "camera";   // "camera", or a plan mode: "imagined", "real" or "both"
let player = null;         // the Plan3D player, made on the first plan
let playerFailed = false;  // no WebGL: the page says so and keeps the camera images
let planRunId = null;      // the id of the plan run on the page, for Execute

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

let running = false;

// The scene chips are off while a run is active: the run keeps its scene.
function setRunning(value) {
  running = value;
  for (const c of $("scenes").children) c.disabled = value;
  $("shuffle").disabled = value;
  $("reset").disabled = value;
  for (const b of document.querySelectorAll(".exec-btn")) b.disabled = value || !planRunId;  // an old entry stays off
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

function setCamState(state, label = "") {
  camState = state;
  camLabel = label;
  paintBadges();
}

// The badge says what the boxes show: the plan video, or the camera images.
function paintBadges() {
  const plan = viewMode !== "camera";
  const text = plan ? PLAN_BADGES[viewMode] : ({ live: "Live", ended: "Last frame", run: camLabel }[camState] || "Preview");
  for (const id of ["cam_front", "cam_top"]) {
    $(id).classList.toggle("live", !plan && camState === "live");
    $(id).classList.toggle("imagined", plan);
    $(id).querySelector(".cam-badge").textContent = text;
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

function openStepDetails(ev) {
  const kindText = { step: "", plan: " · plan", bad_reply: " · bad reply" }[ev.kind] ?? " · model error";
  $("step_title").textContent = `Step ${ev.step}${kindText}`;
  const u = stepUsage(ev);
  const items = [el("h2", "Tokens (exact, from the API)")];
  if (u) {
    const total = (u.in || 0) + (u.cache_write || 0) + (u.cache_read || 0);
    const imageTokens = 2 * Math.round((640 * 480) / 750);
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
      el("p", `images = 2 × round(640×480/750) = 2 × ${imageTokens / 2} = ${imageTokens}; text = total input (${total}) − images`, "muted"),
    );
  } else {
    items.push(el("p", "no usage reported", "muted"));
  }
  const sys = el("details");
  sys.append(el("summary", "System prompt"), el("pre", runSystemPrompt));
  items.push(sys, el("h2", "User message"), el("pre", ev.prompt ?? ""));
  const imgs = stepImages[ev.step];
  if (imgs) {
    const pair = el("div", "", "pair");
    for (const [name, data] of [[`Image 1 · ${runCameras.front}`, imgs.front], [`Image 2 · ${runCameras.top}`, imgs.top]]) {
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
let stepEvents = {};     // step number -> the step, bad_reply or llm_error event, for the details dialog
let stepPrompts = {};    // step number -> the first message text, until the step event comes
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

// Add an entry. Stay at the bottom when the reader is there. Otherwise do not move, and show "↓ latest".
function addEntry(item) {
  const empty = $("chat").querySelector(".chat-empty");
  if (empty) empty.remove();
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

function openStep(step) {
  openStepDetails(stepEvents[step] || { kind: "step", step, prompt: stepPrompts[step] ?? "", attempts: [], usage: null });
}

function stepLink(item, step) {
  item.classList.add("link");
  item.tabIndex = 0;
  item.setAttribute("role", "button");
  item.addEventListener("click", (e) => {
    if (e.target.closest("button, summary")) return;  // "show all" and folds work on their own
    openStep(step);
  });
  item.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && e.target === item) {
      e.preventDefault();  // the focus moves to the dialog: the key must not press its Close button
      openStep(step);
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

function thumbs(step) {
  const imgs = stepImages[step];
  const row = el("div", "", "thumbs");
  if (!imgs) return row;
  for (const [label, data] of [[runCameras.front, imgs.front], [runCameras.top, imgs.top]]) {
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
  if (ev.attempt === 1) {
    stepPrompts[ev.step] = ev.prompt;
    const item = entry("sent", "→ Sent", ev.step);
    item.append(folded(ev.prompt), thumbs(ev.step));
    addEntry(stepLink(item, ev.step));
  } else {
    endThinking();
    const item = entry("retry", "↻ Retry: the reply was not valid", ev.step);
    item.append(folded(ev.prompt));
    addEntry(stepLink(item, ev.step));
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
  addEntry(stepLink(item, ev.step));
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
  addEntry(stepLink(item, ev.step));
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
  addEntry(stepLink(item, ev.step));
}

function onRunStart(ev) {
  const item = entry("system", ev.mode === "plan" ? "Plan started" : "Run started");
  const sys = el("details");
  sys.append(el("summary", "System prompt"), el("pre", ev.system_prompt || ""));
  item.append(field("task", el("span", ev.instruction)), sys);
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
  stepStarts = {};
  $("chat").replaceChildren();
  $("chat_latest").hidden = true;
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

// The plan and the to-do list of the first reply stay at the top of the To-do column, folded to 2 lines.
function showFirstPlan(ev) {
  const box = $("first_plan");
  const body = el("div", "", "fp-body");
  body.append(el("p", ev.plan || "(no plan)", "fp-text"));
  if (ev.todo && ev.todo.length) {
    const ol = el("ol");
    ol.append(...ev.todo.map((t) => li(t)));
    body.append(ol);
  }
  box.replaceChildren(el("span", "First plan", "fp-label"), body);
  box.classList.remove("open");
  box.setAttribute("aria-expanded", "false");
  box.hidden = false;
  markClipped();
}

// A fade shows the folded text. A new box width can change it.
function markClipped() {
  const box = $("first_plan");
  const body = box.querySelector(".fp-body");
  if (!box.hidden && body) box.classList.toggle("clipped", body.scrollHeight > body.clientHeight + 1);
}

function toggleFirstPlan() {
  const open = $("first_plan").classList.toggle("open");
  $("first_plan").setAttribute("aria-expanded", String(open));
}

$("first_plan").addEventListener("click", toggleFirstPlan);
$("first_plan").addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    toggleFirstPlan();
  }
});

function emptyMemo() {
  $("first_plan").hidden = true;
  $("first_plan").replaceChildren();
  $("todo").replaceChildren(li("Claude writes its plan here", "empty"));
  $("notes").replaceChildren(el("span", "Nothing yet", "empty"));
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
  $("prompt_msg").textContent = savedPrompt === null ? "Saved: using the default" : "Saved: Start will use this prompt";
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
  $("chat").replaceChildren(el("li", "Press Start. Each message to Claude and each reply shows here.", "chat-empty"));
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

$("run").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (running) {  // keep the steps of the active run on screen
    setStatus("a run is already active", "bad");
    return;
  }
  const body = {
    scene: sceneName,
    instruction: $("instruction").value.trim(),
    seed: Number($("seed").value),
    max_steps: Number($("max_steps").value),
    show_positions: $("show_positions").checked,
    use_lessons: $("use_lessons").checked,
  };
  if (savedPrompt !== null) body.system_prompt = savedPrompt;
  beginRun("starting");
  if (await postJson("/api/run", body)) setRunning(true);
});

// Plan first: the same instruction, scene, seed and lessons switch as Start. Never the true positions.
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
  };
  beginRun("planning");
  if (await postJson("/api/plan", body)) setRunning(true);
}

$("plan_btn").addEventListener("click", startPlan);

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
    runCams = ev.cameras || null;
    resetChat();
    emptyMemo();
    resetTotals();
    onRunStart(ev);
    if (ev.mode === "plan") setStatus("planning · Claude is thinking", "thinking");
  } else if (ev.kind === "observation") {
    const old = stepImages[ev.step];
    if (old) for (const url of Object.values(old)) URL.revokeObjectURL(url);  // a server URL needs nothing
    stepImages[ev.step] = { front: imageSrc(ev.front_png), top: imageSrc(ev.top_png) };  // for the chat and the step details
    if (Array.isArray(ev.tcp)) stepStarts[ev.step] = { xyz: ev.tcp, yaw: ev.yaw || 0 };
    setStatus(`step ${ev.step} · thinking`, "thinking");
  } else if (ev.kind === "ask") {
    onAsk(ev);
  } else if (ev.kind === "reply") {
    setStatus(`step ${ev.step} · moving`, "moving");
    onReply(ev);
    if ($("first_plan").hidden) showFirstPlan(ev);  // the first reply of the run
    setPath(ev.step, runCams, stepStarts[ev.step] || null, ev.action_data);
  } else if (ev.kind === "action") {
    onPathAction(ev);
  } else if (ev.kind === "step") {
    stepEvents[ev.step] = ev;
    addTotals(stepUsage(ev));
    onStepResults(ev);
    onPathStep(ev);
    $("todo").replaceChildren(...(ev.todo.length ? ev.todo.map((t) => li(t)) : [li("Claude writes its plan here", "empty")]));
    $("notes").replaceChildren(ev.notes ? document.createTextNode(ev.notes) : el("span", "Nothing yet", "empty"));
    for (const t of ev.lessons || []) if (!runLessons.includes(t)) runLessons.push(t);
    showLessons();
  } else if (ev.kind === "plan") {
    stepEvents[ev.step] = ev;
    addTotals(ev.usage);
    onPlan(ev);
    planView = ev.view || null;
    addEntry(planCheckEntry(ev.scores || {}, ev.keyframes || []));
    if (ev.view) showPlan(ev.view);
  } else if (ev.kind === "exec_start") {
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
      }
      setStatus("running the plan", "moving");
      setViewMode("camera");  // the live frames; a saved run keeps the imagined video
    }
    onExecStart(ev);
    if (own) setPath(ev.step, planView.cameras, planView.track && planView.track.start, ev.action_data);
  } else if (ev.kind === "exec_end") {
    if (!replay) setRunning(false);
    if (!replay && $("history_dialog").open) fillHistory();
    setCamState("ended");
    onExecEnd(ev);
    setStatus(execText(ev), ev.success ? "success" : ev.status === "error" ? "bad" : "");
  } else if (ev.kind === "bad_reply") {
    stepEvents[ev.step] = ev;
    addTotals(ev.usage);
    errorEntry(ev, "✕ Bad reply: step skipped", ev.error);
  } else if (ev.kind === "llm_error") {
    stepEvents[ev.step] = ev;
    setStatus(`model error: ${ev.message}`, "bad");
    addTotals(ev.usage);
    errorEntry(ev, `✕ Model error (${ev.error_kind})`, ev.message);
  } else if (ev.kind === "run_end") {
    if (!replay) setRunning(false);
    if (!replay && $("history_dialog").open) fillHistory();  // the rows can open again
    setCamState("ended");
    onRunEnd(ev);
    loadLessons();
    if (ev.status === "planned") setStatus("plan ready · read the plan check, then execute", "success");
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
let stepPath = null;   // { step, cameras, path, n, progress } of the last reply, or null

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
function setPath(step, cams, start, actions) {
  stepPath = cams && actions && window.PathOverlay ? {
    step, cameras: cams, n: actions.length, progress: {}, key: runKey,
    path: { start, items: PathOverlay.items(actions, start, PATH_SPEC) },
  } : null;
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
  const actionStates = show ? PathOverlay.states(stepPath.n, stepPath.progress) : [];
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
  new ResizeObserver(markClipped).observe($("first_plan"));
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
  const item = entry("claude", "← Claude · plan", ev.step);
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
  item.append(field("keys", keys));
  if (ev.warnings && ev.warnings.length) item.append(lines(ev.warnings, "mono"));
  if (ev.usage) item.append(el("div", usageText(ev.usage), "entry-meta"));
  addEntry(stepLink(item, ev.step));
}

// The predicted moves of a keyframe in short form, for example "bowl → (0.62, 0.10)".
function movesText(moves) {
  return (moves || []).map((m) => `${m.name} → (${m.xyz[0].toFixed(2)}, ${m.xyz[1].toFixed(2)})`).join(", ");
}

function planCheckEntry(scores, keyframes = []) {
  const item = entry("plan", "◎ Plan check");
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
  (scores.marks || []).forEach((m, i) => {
    const row = li("", m.ok ? "" : "bad");
    const why = Object.entries(m.checks || {}).filter(([, v]) => v !== "ok").map(([k, v]) => `${k}: ${v}`);
    row.append(el("span", m.ok ? "✓" : "✕", m.ok ? "mark-ok" : "mark-bad"), el("span", ` ${i + 1}`, "n"),
               document.createTextNode([...why, m.info].filter(Boolean).join("; ") || "ok"));
    marks.append(row);
  });
  item.append(field("checks", marks));
  const moved = keyframes.map((k, i) => [i, k.moves || []]).filter(([, m]) => m.length)
    .map(([i, m]) => `${i + 1} · moves ${movesText(m)}`);
  if (moved.length) item.append(field("moves", lines(moved, "mono")));
  const exec = el("button", "▶ Execute plan", "btn exec-btn");
  exec.type = "button";
  exec.disabled = running;
  exec.title = "Run the keyframes once in the physics, from a fresh start of the same scene and seed. No model is called.";
  exec.addEventListener("click", executePlan);
  item.append(exec);
  return item;
}

async function executePlan() {
  if (running || !planRunId) return;
  setStatus("starting the plan", "moving");
  if (await postJson(`/api/plan/${encodeURIComponent(planRunId)}/execute`)) setRunning(true);
}

function onExecStart(ev) {
  const item = entry("system", "▶ Executing the plan");
  item.append(el("div", `${ev.keyframes} keyframes from a fresh start of seed ${ev.seed}. ` +
    (ev.check ? `Auto-check: ${ev.check}.` : "No auto-check.")));
  addEntry(item);
}

function execText(ev) {
  if (ev.status === "success") return "plan executed · task solved";
  if (ev.status === "stopped" && ev.stopped) return `plan stopped at action ${ev.stopped.after} of ${ev.stopped.of}`;
  if (ev.status === "failed") return "plan executed · the auto-check says not solved";
  if (ev.status === "cancelled") return "plan execution stopped";
  if (ev.status === "error") return `error: ${ev.message || "the execution failed"}`;
  return "plan executed";
}

function onExecEnd(ev) {
  const item = entry(ev.success || ev.status === "done" ? "end ok" : "end", `■ Execution ended: ${ev.status}`);
  item.append(el("div", ev.stopped ? `Stopped after action ${ev.stopped.after} of ${ev.stopped.of}: ${ev.stopped.reason}.`
    : ev.success ? "The task is solved." : execText(ev)));
  addEntry(item);
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
  b.title = run.instruction;
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
  let events;
  try {
    const r = await fetch(`/api/runs/${encodeURIComponent(run.id)}`);
    if (!r.ok) throw new Error(`error ${r.status}`);
    events = await r.json();
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
    for (const ev of events) {
      try {
        onEvent(ev, true);
      } catch (err) {
        skipped += 1;
      }
    }
  } finally {
    replaying = false;
  }
  endThinking();  // an unfinished run can end on a question
  scrollChat(true);
  const last = Math.max(...Object.keys(stepImages).map(Number));
  if (stepImages[last]) {
    clearPreviewError();
    showImage("live", stepImages[last].front);
    showImage("top", stepImages[last].top);
  }
  setCamState("run", `Run ${when}`);
  const ok = run.status === "success" || run.status === "done" || run.status === "planned";
  const note = skipped ? ` · ${skipped} bad event${skipped > 1 ? "s" : ""} left out` : "";
  setStatus(`viewing run ${when} · ${run.status}${note}`, ok && !skipped ? "success" : run.status === "error" || run.status === "llm_error" || skipped ? "bad" : "");
}

$("history_btn").addEventListener("click", openHistory);
$("history_close").addEventListener("click", () => $("history_dialog").close());

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
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
