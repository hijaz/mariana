"""
Mariana UI server — FastAPI + SSE.

Serves:
  GET  /                       → index.html
  GET  /api/docs               → list of historical reports
  GET  /api/docs/{doc_id}      → full doc metadata + TOC
  GET  /api/docs/{doc_id}/report → full report text (markdown)
  GET  /api/system             → Ollama / SearXNG / model info
  GET  /events?doc_id=...      → SSE stream (or all docs if omitted)
  POST /api/research           → start a research run (non-blocking)
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from mariana.utils.config import load_config
from mariana.utils.events import emit, subscribe
from mariana.utils.store import get_db, get_document, get_toc, init_db, load_section_content

_HERE = Path(__file__).parent
_UI_DIR = _HERE / "ui"

app = FastAPI(title="Mariana Research UI", docs_url=None, redoc_url=None)


# ── Static UI ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html = (_UI_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


# ── SSE stream ────────────────────────────────────────────────────────────────

@app.get("/events")
async def events(doc_id: str | None = None):
    async def generator():
        async for data in subscribe(doc_id):
            yield {"data": data}
    return EventSourceResponse(generator())


# ── API: system info ──────────────────────────────────────────────────────────

@app.get("/api/system")
async def system_info():
    from mariana.utils.llm import check_ollama
    cfg = load_config()
    ollama_ok, models = check_ollama()
    mem: dict[str, Any] = {}
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
        for line in lines:
            if line.startswith(("MemTotal:", "MemAvailable:", "SwapFree:", "SwapTotal:")):
                key, val = line.split(":", 1)
                mem[key.strip()] = int(val.split()[0]) // 1024  # MB
    except Exception:
        pass
    return {
        "ollama_ok": ollama_ok,
        "ollama_url": cfg.ollama_base_url,
        "models": models if isinstance(models, list) else [],
        "planner_model": cfg.planner_model,
        "summarizer_model": cfg.summarizer_model,
        "searxng_port": cfg.searxng_port,
        "output_dir": str(cfg.output_dir),
        "mem_mb": mem,
        "pid": os.getpid(),
    }


# ── API: list documents ───────────────────────────────────────────────────────

@app.get("/api/docs")
async def list_docs():
    init_db()
    with get_db() as db:
        rows = db.execute(
            "SELECT id, query, title, created_at, updated_at, total_words FROM documents ORDER BY created_at DESC"
        ).fetchall()
    docs = [dict(r) for r in rows]
    # Attach report path if it exists
    cfg = load_config()
    output_dir = Path(cfg.output_dir)
    for doc in docs:
        slug = doc["title"].lower().replace(" ", "_")[:30]
        # Look for a matching output file
        candidates = sorted(output_dir.glob(f"*_{slug[:20]}*.md"), reverse=True)
        doc["report_path"] = str(candidates[0]) if candidates else None
        doc["has_report"] = doc["report_path"] is not None
    return docs


# ── API: single doc detail ────────────────────────────────────────────────────

@app.get("/api/docs/{doc_id}")
async def doc_detail(doc_id: str):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    toc = get_toc(doc_id)
    # Enrich toc nodes with their content snippet
    enriched = []
    for node in toc:
        content = load_section_content(node["id"])
        enriched.append({**node, "content_snippet": (content[:300] + "…") if len(content) > 300 else content})
    # Get search queue state
    with get_db() as db:
        queue_rows = db.execute(
            "SELECT query, status, node_id FROM search_queue WHERE document_id=? ORDER BY created_at DESC LIMIT 50",
            (doc_id,),
        ).fetchall()
    return {
        "doc": doc,
        "toc": enriched,
        "search_queue": [dict(r) for r in queue_rows],
    }


# ── API: full report text ─────────────────────────────────────────────────────

@app.get("/api/docs/{doc_id}/report")
async def doc_report(doc_id: str):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    # Reassemble from sections
    toc = get_toc(doc_id)
    sections = [n for n in toc if n.get("status") == "complete"]
    if not sections:
        # Try partial file
        cfg = load_config()
        partial = Path(cfg.output_dir) / f"{doc_id[:8]}_partial.md"
        if partial.exists():
            return {"markdown": partial.read_text(encoding="utf-8"), "complete": False}
        return {"markdown": "", "complete": False}

    lines = [f"# {doc['title']}\n\n*{doc['query']}*\n"]
    for node in sorted(sections, key=lambda n: n["order_index"]):
        content = load_section_content(node["id"])
        if content:
            lines.append(f"\n## {node['title']}\n\n{content}\n")
    return {"markdown": "\n".join(lines), "complete": True}


# ── API: list output markdown files ──────────────────────────────────────────

@app.get("/api/reports")
async def list_reports():
    cfg = load_config()
    output_dir = Path(cfg.output_dir)
    reports = []
    for f in sorted(output_dir.glob("*.md"), reverse=True):
        if "_partial" in f.name or "_trace" in f.name:
            continue
        stat = f.stat()
        reports.append({
            "filename": f.name,
            "path": str(f),
            "size": stat.st_size,
            "modified": stat.st_mtime,
        })
    return reports


@app.get("/api/reports/{filename:path}")
async def get_report(filename: str):
    cfg = load_config()
    path = Path(cfg.output_dir) / filename
    # Prevent path traversal
    try:
        path.resolve().relative_to(Path(cfg.output_dir).resolve())
    except ValueError:
        raise HTTPException(403, "Forbidden")
    if not path.exists():
        raise HTTPException(404, "Report not found")
    return {"markdown": path.read_text(encoding="utf-8")}


# ── API: start research ────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str
    max_sources: int | None = None
    max_iterations: int | None = None
    target_words: int | None = None
    duration: str | None = None
    require: list[str] = []
    min_sections: int | None = None
    max_sections: int | None = None


_active_run: threading.Thread | None = None
_stop_event = threading.Event()
_active_goal: dict | None = None


@app.post("/api/research")
async def start_research(req: ResearchRequest):
    global _active_run, _active_goal
    if _active_run and _active_run.is_alive():
        return {"error": "A research run is already active"}

    _stop_event.clear()

    def _run():
        from mariana.state import ResearchGoal, initial_state
        from mariana.graph import get_graph
        import mariana.utils.cancel as _cancel
        _cancel.stop_event = _stop_event
        g = ResearchGoal.from_cli(
            duration=req.duration,
            max_sources=req.max_sources,
            target_words=req.target_words,
            max_iterations=req.max_iterations,
            required_topics=tuple(req.require),
        )
        if req.min_sections is not None:
            g.min_sections = req.min_sections
        if req.max_sections is not None:
            g.max_sections = req.max_sections
        state = initial_state(req.query, goal=g)
        try:
            get_graph().invoke(state)
        except Exception as exc:
            emit("error", {"message": str(exc)})
        finally:
            _cancel.stop_event = threading.Event()  # reset to a no-op cleared event

    _active_goal = {
        "query": req.query,
        "max_sources": req.max_sources,
        "max_iterations": req.max_iterations,
        "target_words": req.target_words,
        "duration": req.duration,
        "require": req.require,
        "min_sections": req.min_sections,
        "max_sections": req.max_sections,
    }
    _active_run = threading.Thread(target=_run, daemon=True)
    _active_run.start()
    return {"started": True, "query": req.query}


@app.post("/api/research/stop")
async def stop_research():
    if _active_run and _active_run.is_alive():
        _stop_event.set()
        emit("cancelled", {"message": "Research cancelled by user"})
        return {"stopped": True}
    return {"stopped": False, "reason": "No active run"}


@app.get("/api/research/status")
async def research_status():
    running = _active_run is not None and _active_run.is_alive()
    return {
        "running": running,
        "goal": _active_goal if running else None,
    }
