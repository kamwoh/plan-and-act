const $ = (id) => document.getElementById(id);
let tasks = [];       // the tasks of the chosen scene
let scenes = [];
let sceneName = "cubes";
const DEFAULT_CAMERAS = { front: "Front", top: "Top · grid" };
let cameras = { ...DEFAULT_CAMERAS };  // the image labels of the chosen scene
let runCameras = { ...DEFAULT_CAMERAS };  // the image labels of the scene of the last run, for its step details

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

// keepText: true when a run starts, so that its instruction stays in the box
function chooseScene(name, keepText = false) {
  const scene = scenes.find((s) => s.name === name) || scenes[0];
  if (!scene) return;
  const text = $("instruction").value.trim();
  if (!keepText && tasks.some((t) => t.instruction === text)) $("instruction").value = "";  // an example of the old scene
  sceneName = scene.name;
  tasks = scene.tasks;
  writeScene(sceneName);
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
}
let previewSeq = 0;

// ---- Run totals ----
let stepImages = {};   // step number -> { front, top }: object URLs of the PNG images of the observation events

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
function setCamState(state) {  // "preview" | "live" | "ended"
  for (const id of ["cam_front", "cam_top"]) {
    $(id).classList.toggle("live", state === "live");
    $(id).querySelector(".cam-badge").textContent = state === "live" ? "Live" : state === "ended" ? "Last frame" : "Preview";
  }
}

function showImage(id, src) {
  $(id).src = src;
  $(id).hidden = false;
}

function clearPreviewError() {  // a good preview: drop the text of an earlier error
  for (const fig of ["cam_front", "cam_top"]) $(fig).querySelector(".placeholder").textContent = "";
}

function showPreviewError(message) {  // hide the images and put the message where they were
  for (const [fig, img] of [["cam_front", "live"], ["cam_top", "top"]]) {
    $(img).hidden = true;
    $(fig).querySelector(".placeholder").textContent = message;
  }
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
  const kindText = ev.kind === "step" ? "" : ev.kind === "bad_reply" ? " · bad reply" : " · model error";
  $("step_title").textContent = `Step ${ev.step}${kindText}`;
  const u = ev.usage;
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
  const stick = atBottom();
  const empty = $("chat").querySelector(".chat-empty");
  if (empty) empty.remove();
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
  const item = entry("system", "Run started");
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
  const ok = ev.status === "success" || ev.status === "done";
  const item = entry(ok ? "end ok" : "end", `■ Run ended: ${ev.status}`);
  if (ev.message) item.append(el("div", ev.message));
  addEntry(item);
}

function resetChat() {
  endThinking();
  for (const imgs of Object.values(stepImages)) for (const url of Object.values(imgs)) URL.revokeObjectURL(url);
  stepImages = {};
  stepEvents = {};
  stepPrompts = {};
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

function emptyMemo() {
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
  previewSeq++;  // drop a preview that is still on its way
  setLoading(false);
  // Reset before the POST: events of the new run can arrive before the response.
  runSystemPrompt = "";
  resetTotals();
  resetChat();
  emptyMemo();
  setStatus("starting", "thinking");
  try {
    const r = await fetch("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    if (!r.ok) {
      const { detail } = await r.json().catch(() => ({}));
      const text = Array.isArray(detail) ? detail.map((d) => d.msg).join(", ") : (detail || r.statusText);
      setStatus(`error: ${text}`, "bad");
      return;
    }
  } catch (err) {
    setStatus(`error: ${err.message}`, "bad");
    return;
  }
  setRunning(true);
});

$("stop").addEventListener("click", () => fetch("/api/stop", { method: "POST" }));

// ---- Events from the server ----
function onEvent(ev) {
  if (ev.kind === "run_start") {
    runSystemPrompt = ev.system_prompt || "";
    runCameras = cameraLabels(ev.scene);
    setRunning(true);
    if (ev.scene && ev.scene !== sceneName) chooseScene(ev.scene, true);
    resetChat();
    onRunStart(ev);
  } else if (ev.kind === "observation") {
    const old = stepImages[ev.step];
    if (old) for (const url of Object.values(old)) URL.revokeObjectURL(url);
    stepImages[ev.step] = { front: pngUrl(ev.front_png), top: pngUrl(ev.top_png) };  // for the chat and the step details
    setStatus(`step ${ev.step} · thinking`, "thinking");
  } else if (ev.kind === "ask") {
    onAsk(ev);
  } else if (ev.kind === "reply") {
    setStatus(`step ${ev.step} · moving`, "moving");
    onReply(ev);
  } else if (ev.kind === "step") {
    stepEvents[ev.step] = ev;
    addTotals(ev.usage);
    onStepResults(ev);
    $("todo").replaceChildren(...(ev.todo.length ? ev.todo.map((t) => li(t)) : [li("Claude writes its plan here", "empty")]));
    $("notes").replaceChildren(ev.notes ? document.createTextNode(ev.notes) : el("span", "Nothing yet", "empty"));
    for (const t of ev.lessons || []) if (!runLessons.includes(t)) runLessons.push(t);
    showLessons();
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
    setRunning(false);
    setCamState("ended");
    onRunEnd(ev);
    loadLessons();
    if (ev.status === "success") setStatus("success · task solved", "success");
    else if (ev.status === "done") setStatus("done · Claude says finished", "success");
    else setStatus(`ended: ${ev.status}${ev.message ? ` (${ev.message})` : ""}`, ev.status === "error" || ev.status === "llm_error" ? "bad" : "");
  }
}

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
