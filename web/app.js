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

function showInstruction() {
  const t = tasks.find((x) => x.name === $("task").value);
  $("instruction").value = t ? t.instruction : "";
}

async function loadTasks() {
  tasks = await (await fetch("/api/tasks")).json();
  $("task").replaceChildren(...tasks.map((t) => new Option(t.name, t.name)));
  showInstruction();
}

$("task").addEventListener("change", showInstruction);

$("run").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = {
    task: $("task").value,
    seed: Number($("seed").value),
    max_steps: Number($("max_steps").value),
    show_positions: $("show_positions").checked,
    instruction: $("instruction").value.trim() || null,
  };
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
  $("steps").replaceChildren();
  $("todo").replaceChildren();
  $("notes").textContent = "";
  setStatus("starting");
});

$("stop").addEventListener("click", () => fetch("/api/stop", { method: "POST" }));

function onEvent(ev) {
  if (ev.kind === "observation") {
    $("top").src = `data:image/png;base64,${ev.top_png}`;
    setStatus(`step ${ev.step}: thinking`);
  } else if (ev.kind === "reply") {
    setStatus(`step ${ev.step}: moving`);
  } else if (ev.kind === "step") {
    const item = li(`Step ${ev.step}`, ev.success ? "success" : "");
    const detail = document.createElement("ul");
    detail.append(li(`Saw: ${ev.observation}`), li(`Plan: ${ev.plan}`), ...ev.results.map((r) => li(r, "result")));
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
    if (ev.status === "success") setStatus("SUCCESS", "success");
    else setStatus(`ended: ${ev.status}${ev.message ? ` (${ev.message})` : ""}`, ev.status === "error" ? "bad" : "");
  }
}

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.type === "frame") $("live").src = `data:image/jpeg;base64,${msg.jpg}`;
    else if (msg.type === "event") onEvent(msg);
  };
  ws.onclose = () => setTimeout(connect, 1000);
}

loadTasks();
connect();
