const $ = (id) => document.getElementById(id);
let tasks = [];       // the tasks of the chosen scene
let scenes = [];
let sceneName = "cubes";
const DEFAULT_CAMERAS = { front: "Front", top: "Top · grid" };
let cameras = { ...DEFAULT_CAMERAS };  // the image labels of the chosen scene

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
  cameras = { ...DEFAULT_CAMERAS, ...(scene.cameras || {}) };
  for (const cam of ["front", "top"]) $(`cam_${cam}`).querySelector(".cam-label").textContent = cameras[cam];
  showMode();
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
}
let previewSeq = 0;

// ---- Run totals ----
let stepImages = {};   // step number -> { front, top } (base64 PNG), from the observation events
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
    for (const [name, data] of [[`Image 1 · ${cameras.front}`, imgs.front], [`Image 2 · ${cameras.top}`, imgs.top]]) {
      const fig = el("figure");
      const img = document.createElement("img");
      img.src = `data:image/png;base64,${data}`;
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

function clickable(item, ev) {
  item.tabIndex = 0;
  item.setAttribute("role", "button");
  item.addEventListener("click", () => openStepDetails(ev));
  item.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && e.target === item) {
      e.preventDefault();  // the focus moves to the dialog: the key must not press its Close button
      openStepDetails(ev);
    }
  });
  return item;
}

$("step_close").addEventListener("click", () => $("step_dialog").close());

// ---- Step cards ----
function stepMeta(u) {
  const meta = el("span", "", "meta");
  if (u) {
    const total = (u.in || 0) + (u.cache_write || 0) + (u.cache_read || 0) + (u.out || 0);
    meta.append(el("span", `${fmtTok(total)} tok`), el("span", `${(u.latency_s ?? 0).toFixed(0)}s`));
  }
  return meta;
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

function stepCard(ev) {
  const card = el("li", "", `step ${ev.success || ev.done ? "ok" : ""}`.trim());
  const head = el("div", "", "step-head");
  head.append(el("span", String(ev.step), "num"), el("span", ev.plan || "(no plan)", "plan"), stepMeta(ev.usage));
  card.append(head);
  if (ev.observation) card.append(el("div", ev.observation, "saw"));
  if (ev.results.length) {
    const ul = el("ul", "", "results");
    ul.append(...ev.results.map(resultLine));
    card.append(ul);
  }
  if (ev.success) card.append(el("span", "Task solved", "badge success"));
  else if (ev.done) card.append(el("span", "Claude says done", "badge done"));
  if (ev.truncated) card.append(el("span", "Plan cut to 8 actions", "badge warn"));
  return clickable(card, ev);
}

function errorCard(ev, text) {
  const card = el("li", "", "step err");
  const head = el("div", "", "step-head");
  head.append(el("span", String(ev.step), "num"), el("span", text, "plan errline"), stepMeta(ev.usage));
  card.append(head);
  return clickable(card, ev);
}

function emptyMemo() {
  $("todo").replaceChildren(li("Claude writes its plan here", "empty"));
  $("notes").replaceChildren(el("span", "Nothing yet", "empty"));
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
    if (!r.ok || seq !== previewSeq || running) return;
    const p = await r.json();
    if (seq !== previewSeq || running) return;
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
  };
  if (savedPrompt !== null) body.system_prompt = savedPrompt;
  previewSeq++;  // drop a preview that is still on its way
  setLoading(false);
  // Reset before the POST: events of the new run can arrive before the response.
  stepImages = {};
  runSystemPrompt = "";
  resetTotals();
  $("steps").replaceChildren();
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
    setRunning(true);
    if (ev.scene && ev.scene !== sceneName) chooseScene(ev.scene, true);
  } else if (ev.kind === "observation") {
    stepImages[ev.step] = { front: ev.front_png, top: ev.top_png };  // shown in the step details
    setStatus(`step ${ev.step} · thinking`, "thinking");
  } else if (ev.kind === "reply") {
    setStatus(`step ${ev.step} · moving`, "moving");
  } else if (ev.kind === "step") {
    addTotals(ev.usage);
    $("steps").prepend(stepCard(ev));
    $("todo").replaceChildren(...(ev.todo.length ? ev.todo.map((t) => li(t)) : [li("Claude writes its plan here", "empty")]));
    $("notes").replaceChildren(ev.notes ? document.createTextNode(ev.notes) : el("span", "Nothing yet", "empty"));
  } else if (ev.kind === "bad_reply") {
    addTotals(ev.usage);
    $("steps").prepend(errorCard(ev, `Bad reply: ${ev.error}`));
  } else if (ev.kind === "llm_error") {
    setStatus(`model error: ${ev.message}`, "bad");
    addTotals(ev.usage);
    $("steps").prepend(errorCard(ev, `Model error (${ev.error_kind}): ${ev.message}`));
  } else if (ev.kind === "run_end") {
    setRunning(false);
    setCamState("ended");
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
