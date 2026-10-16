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
  previewSeq++;  // drop a preview that is still on its way
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
  $("steps").replaceChildren();
  $("todo").replaceChildren();
  $("notes").textContent = "";
  setStatus("starting");
});

$("stop").addEventListener("click", () => fetch("/api/stop", { method: "POST" }));

function onEvent(ev) {
  if (ev.kind === "observation") {
    showImage("top", `data:image/png;base64,${ev.top_png}`);
    setStatus(`step ${ev.step}: thinking`);
  } else if (ev.kind === "reply") {
    setStatus(`step ${ev.step}: moving`);
  } else if (ev.kind === "step") {
    const item = li(`Step ${ev.step}`, ev.success ? "success" : "");
    const detail = document.createElement("ul");
    detail.append(li(`Saw: ${ev.observation}`), li(`Plan: ${ev.plan}`), ...ev.results.map((r) => li(r, "result")));
    if (ev.done) detail.append(li("Claude says done", "done"));
    if (ev.truncated) detail.append(li("Plan cut to 8 actions", "bad"));
    item.append(detail);
    $("steps").prepend(item);
    $("todo").replaceChildren(...ev.todo.map((t) => li(t)));
    $("notes").textContent = ev.notes;
  } else if (ev.kind === "bad_reply") {
    $("steps").prepend(li(`Step ${ev.step}: bad reply (${ev.error})`, "bad"));
  } else if (ev.kind === "llm_error") {
    setStatus(`model error: ${ev.message}`, "bad");
    $("steps").prepend(li(`Step ${ev.step}: model error (${ev.error_kind}): ${ev.message}`, "bad"));
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
loadPreview();
connect();
