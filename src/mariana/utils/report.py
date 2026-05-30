"""Write partial (in-progress) research reports to disk."""
import re
from pathlib import Path

from mariana.state import SubQuestion
from mariana.utils.config import ensure_output_dir, load_config


def _sanitize(query: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", query.lower())[:40].strip("_") or "report"


def write_partial_report(state: dict) -> Path:
    """Write an in-progress Markdown report after a summarize pass.

    The file is always named ``{query_slug}_partial.md`` so it is overwritten
    on every pass — only the final report gets a timestamp.
    """
    cfg = load_config()
    output_dir = ensure_output_dir(cfg)
    slug = _sanitize(state.get("query", ""))
    filepath = output_dir / f"{slug}_partial.md"

    lines = [f"# Research in Progress: {state.get('query', '')}\n"]
    lines.append(f"_Iteration {state.get('iteration', 0)} — partial results_\n")

    sub_questions = state.get("sub_questions", [])
    answered = [sq for sq in sub_questions if isinstance(sq, SubQuestion) and sq.answered]
    unanswered = [sq for sq in sub_questions if isinstance(sq, SubQuestion) and not sq.answered]

    for sq in answered:
        heading = re.sub(r"\*{1,3}", "", sq.question).strip()
        lines.append(f"\n## {heading}\n")
        lines.append(sq.summary or "_No summary yet._")
        lines.append("")

    if unanswered:
        lines.append("\n---\n_Pending sub-questions:_\n")
        for sq in unanswered:
            lines.append(f"- {sq.question}")

    filepath.write_text("\n".join(lines), encoding="utf-8")
    return filepath
