"""A self-contained HTML viewer for runs: open it in a browser, no server needed.

    from simsim.viewer import write_viewer
    write_viewer("runs/negotiation", "runs/negotiation/view.html")

or from the shell: `simsim view runs/negotiation`.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .record import read_jsonl

KEEP = {"start", "turn", "tool_call", "probe", "survey", "intervention", "evaluation", "repair", "end",
        "error", "resume", "component_error"}


def _trim(e: dict[str, Any], max_chars: int) -> dict[str, Any]:
    e = {k: v for k, v in e.items() if k not in ("messages", "run", "ts")}
    if e["event"] == "end":
        e["metrics"] = {k: v for k, v in (e.get("metrics") or {}).items() if k not in ("transcript", "posts")}
    s = json.dumps(e, default=str)
    if len(s) > max_chars:
        for k in ("observation", "reasoning"):
            if isinstance(e.get(k), str) and len(e[k]) > 2000:
                e[k] = e[k][:2000] + " ..."
    return e


def collect_runs(source: str | Path, include_prompts: bool = False, max_chars: int = 20000) -> list[dict[str, Any]]:
    source = Path(source)
    paths = [source] if source.is_file() else sorted(source.rglob("seed*.jsonl")) or sorted(source.rglob("*.jsonl"))
    runs = []
    for p in paths:
        events = read_jsonl(p)
        if not events or "event" not in events[0]:
            continue
        keep = KEEP | ({"llm_call"} if include_prompts else set())
        evs = [_trim(e, max_chars) if e["event"] != "llm_call" else e for e in events if e["event"] in keep]
        end = next((e for e in reversed(evs) if e["event"] == "end"), {})
        runs.append({"id": events[0].get("run", p.stem), "path": str(p.relative_to(source.parent if source.is_file()
                     else source)), "events": evs, "termination": end.get("termination", "unfinished"),
                     "steps": end.get("steps")})
    return runs


def write_viewer(source: str | Path, out: str | Path | None = None, include_prompts: bool = False,
                 title: str | None = None) -> Path:
    """Write an HTML viewer for an experiment directory (or one run log) and return its path."""
    source = Path(source)
    out = Path(out) if out else (source if source.is_dir() else source.parent) / "view.html"
    runs = collect_runs(source, include_prompts)
    data = json.dumps(runs, default=str).replace("</", "<\\/")
    page = PAGE.replace("__TITLE__", html.escape(title or f"SimSim: {source.name}")).replace("__DATA__", data)
    out.write_text(page)
    return out


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg:#fbfbf9; --panel:#fff; --ink:#1d1d1b; --muted:#6b6b66; --line:#e4e3dd; --accent:#2f5d8a;
  --priv:#f6f1e3; --tool:#eef3f7; --probe:#eef6ee; --warn:#a4452c; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#161615; --panel:#1f1f1d; --ink:#e9e8e3;
  --muted:#9a9890; --line:#33322e; --accent:#8fb3d9; --priv:#2b281f; --tool:#1f2a33; --probe:#1f2b20; --warn:#e08b73; } }
:root[data-theme="dark"] { --bg:#161615; --panel:#1f1f1d; --ink:#e9e8e3; --muted:#9a9890; --line:#33322e;
  --accent:#8fb3d9; --priv:#2b281f; --tool:#1f2a33; --probe:#1f2b20; --warn:#e08b73; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
header { padding:12px 16px; border-bottom:1px solid var(--line); display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
header h1 { font-size:16px; margin:0; font-weight:600; }
header input { flex:1; min-width:160px; padding:6px 8px; border:1px solid var(--line); border-radius:6px;
  background:var(--panel); color:var(--ink); }
.wrap { display:grid; grid-template-columns:300px 1fr; min-height:calc(100vh - 50px); }
nav { border-right:1px solid var(--line); overflow:auto; max-height:calc(100vh - 50px); }
nav button { display:block; width:100%; text-align:left; padding:8px 16px; border:0; border-bottom:1px solid var(--line);
  background:none; color:var(--ink); cursor:pointer; font:inherit; }
nav button:hover, nav button.on { background:var(--panel); }
nav small { color:var(--muted); display:block; }
main { padding:16px; overflow:auto; max-height:calc(100vh - 50px); }
.ev { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px 12px; margin:0 0 10px; }
.who { font-weight:600; } .meta { color:var(--muted); font-size:12px; margin-left:6px; }
.text { white-space:pre-wrap; margin-top:4px; }
details { margin-top:6px; } summary { cursor:pointer; color:var(--muted); font-size:12px; }
.box { white-space:pre-wrap; font:12px/1.45 ui-monospace,Menlo,monospace; padding:8px; border-radius:6px; margin-top:4px; overflow-x:auto; }
.priv { background:var(--priv); } .tool { background:var(--tool); } .probe { background:var(--probe); }
.note { color:var(--muted); font-size:13px; margin:6px 0 10px; } .err { color:var(--warn); }
.bar { display:flex; align-items:center; gap:6px; font-size:12px; } .bar i { display:inline-block; height:8px;
  background:var(--accent); border-radius:2px; }
table { border-collapse:collapse; font-size:13px; } td { padding:2px 10px 2px 0; vertical-align:top; }
@media (max-width: 760px) { .wrap { grid-template-columns:1fr; } nav { max-height:220px; border-right:0;
  border-bottom:1px solid var(--line); } main { max-height:none; } }
</style></head><body>
<header><h1>__TITLE__</h1><input id="q" placeholder="Filter runs or search text..."></header>
<div class="wrap"><nav id="nav"></nav><main id="main"></main></div>
<script>
const RUNS = __DATA__;
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const js = o => esc(JSON.stringify(o, null, 2));
let current = 0;
function matches(run, q) { return !q || JSON.stringify(run).toLowerCase().includes(q); }
function renderNav() {
  const q = $("#q").value.toLowerCase();
  $("#nav").innerHTML = RUNS.map((r, i) => matches(r, q) ? `<button data-i="${i}" class="${i===current?'on':''}">
    ${esc(r.id)}<small>${esc(r.termination)}${r.steps!=null ? " · "+r.steps+" steps" : ""}</small></button>` : "").join("")
    || '<p class="note" style="padding:0 16px">No runs match.</p>';
  document.querySelectorAll("nav button").forEach(b => b.onclick = () => { current = +b.dataset.i; renderNav(); renderRun(); });
}
function bars(probs) {
  const max = Math.max(...Object.values(probs), 1e-9);
  return Object.entries(probs).map(([k, v]) => `<div class="bar"><span style="width:2.5em">${esc(k)}</span>
    <i style="width:${Math.round(160*v/max)}px"></i>${v.toFixed(3)}</div>`).join("");
}
function renderEvent(e) {
  const t = e.time ? esc(e.time) : (e.t != null ? "t=" + e.t : "");
  switch (e.event) {
    case "start": return `<div class="ev"><div class="who">Agents</div><table>${(e.agents||[]).map(a =>
      `<tr><td><b>${esc(a.name)}</b></td><td>${esc(a.llm?.model || "")}${a.memory?.type ? " · " + esc(a.memory.type) : ""}
      ${a.components?.length ? " · " + esc(a.components.join(", ")) : ""}<br><span class="meta" style="margin:0">
      ${esc(a.persona?.description || "")}</span></td></tr>`).join("")}</table><div class="note">seed ${esc(e.seed)}</div></div>`;
    case "turn": return `<div class="ev"><span class="who">${esc(e.agent)}</span><span class="meta">${t}</span>
      ${e.error ? `<span class="meta err">${esc(e.error)}</span>` : ""}<div class="text">${esc(e.text)}</div>
      ${(e.tool_calls||[]).map(c => `<div class="box tool">${esc(c.name)}(${esc(JSON.stringify(c.arguments))})\n→ ${esc(String(c.result).slice(0, 1500))}</div>`).join("")}
      ${e.private && Object.keys(e.private).length ? `<details open><summary>private</summary><div class="box priv">${js(e.private)}</div></details>` : ""}
      ${e.components && Object.keys(e.components).length ? `<details><summary>components</summary><div class="box">${js(e.components)}</div></details>` : ""}
      ${e.reasoning ? `<details><summary>reasoning</summary><div class="box priv">${esc(e.reasoning)}</div></details>` : ""}
      <details><summary>what ${esc(e.agent)} saw</summary><div class="box">${esc(e.observation)}</div></details></div>`;
    case "probe": return `<div class="ev"><span class="who">Probe → ${esc(e.agent)}</span><span class="meta">${t}</span>
      <div class="box probe">Q: ${esc(e.question)}\nA: ${typeof e.answer === "string" ? esc(e.answer) : js(e.answer)}</div></div>`;
    case "survey": return `<div class="ev"><span class="who">Survey → ${esc(e.agent)}</span><span class="meta">${t}</span>
      <div class="note" style="margin:4px 0">${esc(e.question)}</div>${bars(e.probs || {})}</div>`;
    case "intervention": return `<div class="note">◆ intervention at t=${esc(e.t)}: ${esc(e.fn)}</div>`;
    case "evaluation": return `<div class="ev"><span class="who">Evaluation: ${esc(e.name)}</span><div class="box probe">${js(e.result)}</div></div>`;
    case "repair": return `<div class="note">↻ ${esc(e.agent)} repaired output (${e.ok ? "ok" : "failed"}): ${esc((e.errors||[]).join("; "))}</div>`;
    case "end": return `<div class="ev"><span class="who">End</span><span class="meta">${esc(e.termination)} after ${esc(e.steps)} steps</span>
      <details open><summary>metrics</summary><div class="box">${js(e.metrics)}</div></details>
      <details><summary>usage</summary><div class="box">${js(e.usage)}</div></details></div>`;
    case "llm_call": return `<details class="ev"><summary>LLM call: ${esc(e.agent)} ${t} step ${esc(e.step)}</summary>
      <div class="box">${(e.messages||[]).map(m => `[${m.role}]\n${m.content ?? JSON.stringify(m.tool_calls)}`).map(esc).join("\n\n")}</div></details>`;
    case "error": return `<div class="ev err"><b>Error</b><div class="box">${esc(e.traceback)}</div></div>`;
    default: return "";
  }
}
function renderRun() {
  const run = RUNS[current];
  if (!run) { $("#main").innerHTML = '<p class="note">No runs found.</p>'; return; }
  const q = $("#q").value.toLowerCase();
  const evs = run.events.filter(e => !q || e.event !== "turn" || JSON.stringify(e).toLowerCase().includes(q) || matches(run, q));
  $("#main").innerHTML = `<h2 style="font-size:15px;margin:0 0 4px">${esc(run.id)}</h2><div class="note">${esc(run.path)}</div>`
    + evs.map(renderEvent).join("");
}
$("#q").oninput = () => { renderNav(); renderRun(); };
renderNav(); renderRun();
</script></body></html>
"""
