const $ = (id) => document.getElementById(id);
let tasks = [];

function setStatus(text, cls = "") {
  $("status").textContent = text;
  $("status").className = cls;
}

function li(text, cls = "") {
  const el = document.createElement("li");
  el.textContent = text;
  if (cls) el.className = cls;
  return el;
}

function showMode() {
  const text = $("instruction").value.trim();
  const t = tasks.find((x) => x.instruction === text);
  $("mode").textContent = t ? `Auto-check: ${t.name}` : "No auto-check: Claude says when it is done";
}

async function loadTasks() {
  tasks = await (await fetch("/api/tasks")).json();
  $("examples").append(...tasks.map((t) => new Option(t.name, t.instruction)));
  showMode();
}

$("examples").addEventListener("change", () => {
  if ($("examples").value) $("instruction").value = $("examples").value;
  showMode();
});
$("instruction").addEventListener("input", () => {
  const match = tasks.find((x) => x.instruction === $("instruction").value.trim());
  $("examples").value = match ? match.instruction : "";
  showMode();
});

let running = false;
let previewSeq = 0;

// ---- Run totals and per-step details ----
let stepImages = {};   // step number -> { front, top } (base64 PNG), from the observation events
let runSystemPrompt = "";
let totals;

function resetTotals() {
  totals = { steps: 0, input: 0, out: 0, cost: 0, time: 0 };
  showTotals();
}

function addTotals(u) {
  totals.steps += 1;
  if (!u) return;
  totals.input += (u.in || 0) + (u.cache_write || 0) + (u.cache_read || 0);
  totals.out += u.out || 0;
  totals.cost += u.cost_usd || 0;
  totals.time += u.latency_s || 0;
  showTotals();
}

function showTotals() {
  const avg = totals.steps ? `${(totals.time / totals.steps).toFixed(1)}s` : "-";
  $("totals").textContent =
    `steps ${totals.steps} · in ${totals.input} tok · out ${totals.out} tok · $${totals.cost.toFixed(4)} · avg ${avg}/step`;
}
resetTotals();

function el(tag, text = "", cls = "") {
  const e = document.createElement(tag);
  if (text) e.textContent = text;
  if (cls) e.className = cls;
  return e;
}

function kv(rows) {
  const dl = el("dl", "", "kv");
  for (const [k, v] of rows) dl.append(el("dt", k), el("dd", String(v)));
  return dl;
}

function usageLine(u) {
  if (!u) return "no usage reported";
  return `in ${u.in ?? "?"} · cache write ${u.cache_write ?? "?"} · cache read ${u.cache_read ?? "?"} · ` +
    `out ${u.out ?? "?"} · $${(u.cost_usd ?? 0).toFixed(4)} · ${(u.latency_s ?? 0).toFixed(1)}s`;
}

function pre(text) {
  return el("pre", text);
}

function openStepDetails(ev) {
  const body = $("step_body");
  const title = ev.kind === "step" ? `Step ${ev.step}` : `Step ${ev.step} (${ev.kind === "bad_reply" ? "bad reply" : "model error"})`;
  $("step_title").textContent = title;
  const u = ev.usage;
  const items = [el("h2", "Tokens")];
  if (u) {
    const total = (u.in || 0) + (u.cache_write || 0) + (u.cache_read || 0);
    const imageTokens = 2 * Math.round((640 * 480) / 750);
    items.push(
      kv([["input (uncached)", u.in ?? "?"], ["cache write", u.cache_write ?? "?"], ["cache read", u.cache_read ?? "?"],
          ["output", u.out ?? "?"], ["cost", `$${(u.cost_usd ?? 0).toFixed(4)}`], ["time", `${(u.latency_s ?? 0).toFixed(1)} s`]]),
      el("h2", "Estimated split (not exact)"),
      kv([["images", `${imageTokens} tokens`], ["text", `${Math.max(0, total - imageTokens)} tokens`]]),
      el("p", `images = 2 × round(640×480/750) = 2 × ${imageTokens / 2} = ${imageTokens}; text = total input (${total}) - images, at least 0`, "muted"),
    );
  } else {
    items.push(el("p", "no usage reported", "muted"));
  }
  const sys = el("details");
  sys.append(el("summary", "System prompt"), pre(runSystemPrompt));
  items.push(sys, el("h2", "User message"), pre(ev.prompt ?? ""));
  const imgs = stepImages[ev.step];
  if (imgs) {
    const pair = el("div", "", "pair");
    for (const [name, data] of [["Front camera", imgs.front], ["Top camera", imgs.top]]) {
      const img = document.createElement("img");
      img.src = `data:image/png;base64,${data}`;
      img.alt = name;
      pair.append(img);
    }
    items.push(pair);
  }
  items.push(el("h2", "Attempts"));
  (ev.attempts || []).forEach((a, i) => {
    const box = el("div", "", "attempt");
    box.append(el("strong", `Attempt ${i + 1}: ${a.images} image${a.images === 1 ? "" : "s"} sent`), pre(a.message));
    if (a.reply !== null && a.reply !== undefined) box.append(el("strong", "Reply"), pre(a.reply));
    if (a.error) box.append(el("strong", "Error", "bad"), pre(a.error));
    box.append(el("div", usageLine(a.usage), "muted"));
    items.push(box);
  });
  body.replaceChildren(...items);
  body.scrollTop = 0;
  $("step_dialog").showModal();
}

function clickable(item, ev) {
  item.classList.add("clickable");
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

// Show the starting scene of the seed, unless a run is active.
async function loadPreview() {
  const seq = ++previewSeq;
  try {
    const state = await (await fetch("/api/state")).json();
    running = state.running;
    if (running) return;
    const r = await fetch(`/api/preview?seed=${encodeURIComponent($("seed").value || 0)}`);
    if (!r.ok || seq !== previewSeq || running) return;
    const p = await r.json();
    if (seq !== previewSeq || running) return;
    showImage("live", `data:image/jpeg;base64,${p.front}`);
    showImage("top", `data:image/png;base64,${p.top}`);
  } catch (err) {
    // no preview is not an error: the images stay hidden
  }
}

function showImage(id, src) {
  $(id).src = src;
  $(id).hidden = false;
}

let previewTimer = null;
$("seed").addEventListener("input", () => {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(loadPreview, 300);
});

$("run").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = {
    instruction: $("instruction").value.trim(),
    seed: Number($("seed").value),
    max_steps: Number($("max_steps").value),
    show_positions: $("show_positions").checked,
  };
  if (savedPrompt !== null) body.system_prompt = savedPrompt;
  previewSeq++;  // drop a preview that is still on its way
  // Reset before the POST: events of the new run can arrive before the response.
  stepImages = {};
  runSystemPrompt = "";
  resetTotals();
  $("steps").replaceChildren();
  $("todo").replaceChildren();
  $("notes").textContent = "";
  setStatus("starting");
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
  running = true;
});

$("stop").addEventListener("click", () => fetch("/api/stop", { method: "POST" }));

function onEvent(ev) {
  if (ev.kind === "run_start") {
    runSystemPrompt = ev.system_prompt || "";
  } else if (ev.kind === "observation") {
    stepImages[ev.step] = { front: ev.front_png, top: ev.top_png };
    showImage("top", `data:image/png;base64,${ev.top_png}`);
    setStatus(`step ${ev.step}: thinking`);
  } else if (ev.kind === "reply") {
    setStatus(`step ${ev.step}: moving`);
  } else if (ev.kind === "step") {
    addTotals(ev.usage);
    const item = clickable(li(`Step ${ev.step}`, ev.success ? "success" : ""), ev);
    const detail = document.createElement("ul");
    detail.append(li(`Saw: ${ev.observation}`), li(`Plan: ${ev.plan}`), ...ev.results.map((r) => li(r, "result")));
    if (ev.done) detail.append(li("Claude says done", "done"));
    if (ev.truncated) detail.append(li("Plan cut to 8 actions", "bad"));
    item.append(detail);
    $("steps").prepend(item);
    $("todo").replaceChildren(...ev.todo.map((t) => li(t)));
    $("notes").textContent = ev.notes;
  } else if (ev.kind === "bad_reply") {
    addTotals(ev.usage);
    $("steps").prepend(clickable(li(`Step ${ev.step}: bad reply (${ev.error})`, "bad"), ev));
  } else if (ev.kind === "llm_error") {
    setStatus(`model error: ${ev.message}`, "bad");
    addTotals(ev.usage);
    $("steps").prepend(clickable(li(`Step ${ev.step}: model error (${ev.error_kind}): ${ev.message}`, "bad"), ev));
  } else if (ev.kind === "run_end") {
    running = false;
    if (ev.status === "success") setStatus("SUCCESS", "success");
    else if (ev.status === "done") setStatus("DONE (Claude says finished)", "success");
    else setStatus(`ended: ${ev.status}${ev.message ? ` (${ev.message})` : ""}`, ev.status === "error" ? "bad" : "");
  }
}

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.type === "frame") showImage("live", `data:image/jpeg;base64,${msg.jpg}`);
    else if (msg.type === "event") onEvent(msg);
  };
  ws.onclose = () => setTimeout(connect, 1000);
}

loadTasks();
loadPrompt();
loadPreview();
connect();
