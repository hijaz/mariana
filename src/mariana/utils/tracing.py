"""Write a JSON trace file to disk after every run."""
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from mariana.state import SubQuestion
from mariana.utils.config import ensure_output_dir, load_config


def _sanitize(query: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", query.lower())[:40].strip("_") or "report"


def write_trace(state: dict, elapsed: float, report_path: Path | None = None) -> Path:
    """Write a JSON trace file capturing key metrics for this research session."""
    cfg = load_config()
    output_dir = ensure_output_dir(cfg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _sanitize(state.get("query", ""))
    trace_path = output_dir / f"{timestamp}_{slug}_trace.json"

    sub_questions = state.get("sub_questions", [])
    sections = []
    all_urls: list[str] = []
    for sq in sub_questions:
        if not isinstance(sq, SubQuestion):
            continue
        sections.append(sq.question)
        for r in sq.results:
            if r.url:
                all_urls.append(r.url)

    domains = list({urlparse(u).netloc.replace("www.", "") for u in all_urls})

    trace = {
        "timestamp": datetime.now().isoformat(),
        "query": state.get("query", ""),
        "document_title": state.get("document_title", ""),
        "planner_model": cfg.planner_model,
        "summarizer_model": cfg.summarizer_model,
        "iterations": state.get("iteration", 0),
        "sections": sections,
        "source_urls": all_urls,
        "unique_domains": domains,
        "llm_call_count": state.get("llm_call_count", 0),
        "elapsed_seconds": round(elapsed, 1),
        "report_path": str(report_path) if report_path else None,
    }

    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return trace_path
