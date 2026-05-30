import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rich.console import Console

from mariana.prompts import (
    CONCLUSION_PROMPT,
    DISTILL_PROMPT,
    EXEC_SUMMARY_PROMPT,
    EXTRACT_PROMPT,
    ORCHESTRATOR_PROMPT,
    PLANNER_PROMPT,
    REPORT_PROMPT,
    SECTION_PROMPT,
    SYNTHESIZE_PROMPT,
)
from mariana.state import SearchResult, SubQuestion
from mariana.tools.search import filter_for_diversity, raw_scrape, raw_search
from mariana.utils.config import ensure_output_dir, load_config
from mariana.utils.llm import get_planner_llm, get_summarizer_llm, truncate_for_llm
from mariana.utils.report import write_partial_report
from mariana.utils.retry import with_llm_retry

console = Console()

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

# Phrases that indicate a vague, unhelpful summary — used in reflect heuristic
_VAGUE_PHRASES = [
    "in general",
    "it depends",
    "various factors",
    "many aspects",
    "further research",
    "not enough information",
    "unclear",
    "it is difficult",
]

_STOP_WORDS = {
    "what", "how", "why", "when", "where", "which", "who",
    "are", "is", "the", "a", "an", "and", "or", "of", "in",
    "to", "for", "with", "on", "at", "by", "from", "as",
    "do", "does", "did", "have", "has", "had", "be", "been",
    "describe", "explain", "compare", "contrast", "detail",
    "specifically", "including", "particularly", "currently",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _render_llm_output(result: Any) -> str:
    try:
        generation = result.generations[0][0]
    except Exception:
        return ""
    if hasattr(generation, "message"):
        content = getattr(generation.message, "content", "") or ""
        if content:
            return content
    return getattr(generation, "text", "") or ""


def _parse_numbered_list(text: str) -> list[str]:
    """Extract only lines that begin with a digit prefix like '1.' or '2)'."""
    lines = text.strip().splitlines()
    items = []
    for line in lines:
        match = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
        if match:
            item = match.group(1).strip()
            if item:
                items.append(item)
    return items


def _extract_keywords(question: str) -> str:
    """Deterministic fallback keyword extractor — no LLM call."""
    clean = re.sub(r"\*\*.*?\*\*:?\s*", "", question)
    words = re.findall(r"\b[a-zA-Z]{3,}\b", clean)
    keywords = [w.lower() for w in words if w.lower() not in _STOP_WORDS]
    return " ".join(keywords[:6])


def validate_and_clean_query(raw: str, fallback_question: str) -> str:
    """Strip markdown artifacts from a distilled query; fall back to keyword extraction."""
    cleaned = raw.strip()
    cleaned = re.sub(r"`+", "", cleaned)
    cleaned = re.sub(r"\*+", "", cleaned)
    cleaned = re.sub(r"^#+\s*", "", cleaned)
    cleaned = re.sub(r"[\[\](){}'\"']", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    words = [w for w in cleaned.split() if len(w) > 1]

    if len(words) == 0 or len(words) > 10:
        fallback = _extract_keywords(fallback_question)
        console.print(f"  [dim][distill] fallback triggered (raw: {repr(raw[:40])}) → {fallback!r}[/]")
        return fallback

    return " ".join(words[:7])


def _clean_heading(question: str) -> str:
    """Convert a verbose sub-question into a concise section heading."""
    # Remove bold label prefixes: **Background & Principles:** or **Label:**
    text = re.sub(r"\*\*[^*]+\*\*:?\s*", "", question)
    # Remove parenthetical examples: (e.g., tokamaks), (including ...)
    text = re.sub(r"\([^)]{0,80}\)", "", text)
    # Take only the first clause — stop at first ? , ; – or double space
    for delimiter in ["?", ",", ";", "–", "  "]:
        if delimiter in text:
            text = text.split(delimiter)[0]
            break
    text = text.strip().rstrip(":").strip()
    # Hard cap at 55 chars
    return text[:55] if len(text) > 55 else (text or question[:40])


def _unique_query(query: str, used: set) -> str:
    """Return query deduplicated against already-used queries."""
    if query not in used:
        used.add(query)
        return query
    for suffix in ["overview", "explained", "research 2024", "latest"]:
        candidate = f"{query} {suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
    return query


def _chunk_content(text: str) -> list[str]:
    """Split long text into overlapping chunks of CHUNK_SIZE chars."""
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        # Try to break at sentence boundary
        if end < len(text):
            last_period = text.rfind(".", start + int(CHUNK_SIZE * 0.6), end)
            if last_period > start + int(CHUNK_SIZE * 0.6):
                end = last_period + 1
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return chunks


# ── LLM call helpers (wrapped with retry) ────────────────────────────────────

def _inc_llm(state: dict) -> None:
    """Increment llm_call_count in state (in-place, best-effort)."""
    try:
        state["llm_call_count"] = state.get("llm_call_count", 0) + 1
    except Exception:
        pass


@with_llm_retry
def _llm_plan(query: str, gaps: str, num_questions: int) -> str:
    prompt = PLANNER_PROMPT.format_prompt(
        query=truncate_for_llm(query, "query"),
        gaps=truncate_for_llm(gaps, "gaps"),
        num_questions=num_questions,
    )
    return _render_llm_output(get_planner_llm().generate_prompt([prompt])).strip()


@with_llm_retry
def _llm_distill_raw(question: str) -> str:
    """Call LLM to distill a search query. Returns raw output before validation."""
    prompt = DISTILL_PROMPT.format_prompt(question=truncate_for_llm(question, "question"))
    return _render_llm_output(get_planner_llm().generate_prompt([prompt])).strip()


def _llm_distill(question: str, used_queries: set) -> str:
    """Distill a search query, validate, deduplicate against used set."""
    raw = _llm_distill_raw(question)
    # Take first non-empty line only
    raw = next((line.strip() for line in raw.splitlines() if line.strip()), raw)
    cleaned = validate_and_clean_query(raw, question)
    return _unique_query(cleaned, used_queries)


@with_llm_retry
def _llm_extract_chunk(question: str, domain: str, content: str) -> str:
    prompt = EXTRACT_PROMPT.format_prompt(
        question=truncate_for_llm(question, "question"),
        source_title=domain,
        source_url=domain,
        content=content,  # already chunked to CHUNK_SIZE
    )
    return _render_llm_output(get_summarizer_llm().generate_prompt([prompt])).strip()


def _extract_from_source(source: SearchResult, question: str) -> str:
    """Extract key points from one source, chunked if long."""
    chunks = _chunk_content(source.content or "")
    domain = urlparse(source.url).netloc.replace("www.", "")
    points = []
    for chunk in chunks:
        point = _llm_extract_chunk(question, domain, chunk)
        if point and "NOT RELEVANT" not in point.upper() and len(point) > 20:
            points.append(point)
    if not points:
        return ""
    return " ".join(points)


@with_llm_retry
def _llm_synthesize(question: str, findings: str) -> str:
    prompt = SYNTHESIZE_PROMPT.format_prompt(
        question=truncate_for_llm(question, "question"),
        findings=truncate_for_llm(findings, "findings"),
    )
    return _render_llm_output(get_summarizer_llm().generate_prompt([prompt])).strip()


@with_llm_retry
def _llm_section(question: str, summary: str) -> str:
    prompt = SECTION_PROMPT.format_prompt(
        question=truncate_for_llm(question, "question"),
        summary=truncate_for_llm(summary, "summary"),
    )
    return _render_llm_output(get_summarizer_llm().generate_prompt([prompt])).strip()


@with_llm_retry
def _llm_conclusion(query: str, findings: str) -> str:
    prompt = CONCLUSION_PROMPT.format_prompt(
        query=truncate_for_llm(query, "query"),
        findings=truncate_for_llm(findings, "findings"),
    )
    return _render_llm_output(get_summarizer_llm().generate_prompt([prompt])).strip()


@with_llm_retry
def _llm_exec_summary(query: str, summary_inputs: str) -> str:
    prompt = EXEC_SUMMARY_PROMPT.format_prompt(
        query=truncate_for_llm(query, "query"),
        section_texts=truncate_for_llm(summary_inputs, "exec_summary_inputs"),
    )
    return _render_llm_output(get_summarizer_llm().generate_prompt([prompt])).strip()


@with_llm_retry
def _llm_orchestrate(query: str, gaps: str) -> str:
    prompt = ORCHESTRATOR_PROMPT.format_prompt(
        query=truncate_for_llm(query, "query"),
        gaps=truncate_for_llm(gaps, "gaps"),
    )
    return _render_llm_output(get_planner_llm().generate_prompt([prompt])).strip()


# ── Nodes ─────────────────────────────────────────────────────────────────────

def orchestrator_node(state: dict) -> dict:
    """Generate a structured Table of Contents with pre-validated search queries."""
    query = state.get("query", "")
    gaps = state.get("gaps", []) or []
    gaps_text = "\n".join(f"- {gap}" for gap in gaps) if gaps else "None yet"
    used_queries: set = set(state.get("used_queries") or set())
    console.print(f"[yellow]Orchestrating[/] outline for: [bold]{query}[/]")
    if gaps:
        console.print(f"[yellow]Addressing gaps[/]: {', '.join(g[:50] for g in gaps)}")

    raw = _llm_orchestrate(query, gaps_text)
    _inc_llm(state)

    # Strip code fences if the LLM wrapped the JSON anyway
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    toc: dict = {}
    try:
        toc = json.loads(raw)
    except json.JSONDecodeError:
        console.print("[yellow]Orchestrator JSON parse failed — falling back to plan_node output[/]")

    sub_questions = []
    existing = {
        sq.question: sq
        for sq in state.get("sub_questions", [])
        if isinstance(sq, SubQuestion) and sq.answered
    }

    for section in toc.get("sections", []):
        heading = section.get("heading", "").strip()
        if not heading:
            continue
        raw_queries = section.get("search_queries", [])
        validated_queries = [
            _unique_query(validate_and_clean_query(q, heading), used_queries)
            for q in raw_queries
            if q.strip()
        ]
        if heading in existing:
            existing[heading].search_queries = validated_queries
            sub_questions.append(existing[heading])
        else:
            sub_questions.append(SubQuestion(question=heading, search_queries=validated_queries))

    if not sub_questions:
        # JSON was empty or unparseable — fall back to gap-based sub-questions
        console.print("[yellow]Orchestrator returned no sections — using gaps as sub-questions[/]")
        for gap in gaps or [query]:
            sub_questions.append(SubQuestion(question=gap))

    document_title = toc.get("title", "").strip() or query

    return {
        "sub_questions": sub_questions,
        "document_title": document_title,
        "used_queries": used_queries,
        "llm_call_count": state.get("llm_call_count", 0) + 1,
        "status": f"Outline: {len(sub_questions)} sections planned",
    }


def plan_node(state: dict) -> dict:
    """Legacy planner (used if orchestrator is bypassed). Outputs numbered sub-questions."""
    query = state.get("query", "")
    gaps = state.get("gaps", []) or []
    gaps_text = "\n".join(f"- {gap}" for gap in gaps) if gaps else "None yet"
    cfg = load_config()
    console.print(f"[yellow]Planning[/] query: [bold]{query}[/]")
    if gaps:
        console.print(f"[yellow]Existing gaps[/]: {', '.join(gaps)}")

    answer = _llm_plan(query, gaps_text, cfg.num_subquestions)
    _inc_llm(state)
    questions = _parse_numbered_list(answer)

    existing = {
        sq.question: sq
        for sq in state.get("sub_questions", [])
        if isinstance(sq, SubQuestion) and sq.answered
    }
    new_sub_questions = []
    for question in questions:
        if question in existing:
            new_sub_questions.append(existing[question])
        else:
            new_sub_questions.append(SubQuestion(question=question))

    return {
        "sub_questions": new_sub_questions,
        "llm_call_count": state.get("llm_call_count", 0) + 1,
        "status": f"Planned {len(new_sub_questions)} sub-questions",
    }


def search_node(state: dict) -> dict:
    cfg = load_config()
    sub_questions = state.get("sub_questions", [])
    scraped_urls: set = set(state.get("scraped_urls") or set())
    used_queries: set = set(state.get("used_queries") or set())
    session_domain_counts: dict = dict(state.get("session_domain_counts") or {})
    console.print(f"[cyan]Searching[/] {len(sub_questions)} sub-question(s), max {cfg.max_results} results each")
    updated_questions = []

    for sq in sub_questions:
        if not isinstance(sq, SubQuestion):
            updated_questions.append(sq)
            continue
        if sq.answered:
            updated_questions.append(sq)
            continue

        # Use pre-validated search_queries from orchestrator if available; else distill on the fly
        if sq.search_queries:
            search_queries = [
                _unique_query(q, used_queries) for q in sq.search_queries
            ]
        else:
            search_queries = [_llm_distill(sq.question, used_queries)]
            _inc_llm(state)

        all_results: list[SearchResult] = []
        section_domain_counts: dict = {}

        for search_query in search_queries:
            console.print(f"[cyan]Query:[/] [italic]{search_query}[/] (for: [bold]{sq.question[:60]}[/])")
            raw_results = raw_search(search_query, cfg.searxng_port, cfg.max_results)
            if not raw_results:
                console.print(f"[red]No results for:[/] {search_query}")
                continue

            # Domain diversity filter
            diverse = filter_for_diversity(raw_results, section_domain_counts, session_domain_counts)

            for idx, item in enumerate(diverse, start=1):
                url = item.get("url", "")
                title = item.get("title", "[no title]")
                console.print(f"  • {idx}/{len(diverse)}: [bold]{title}[/] ({url})")
                if url in scraped_urls:
                    console.print(f"    Already scraped — skipping")
                    content = ""
                else:
                    content = raw_scrape(url, cfg.max_page_chars)
                    if content:
                        scraped_urls.add(url)
                console.print(f"    {len(content):,} chars scraped")
                all_results.append(SearchResult(title=title, url=url, snippet=item.get("snippet", ""), content=content))
                if idx < len(diverse):
                    time.sleep(cfg.search_delay_seconds)

            if len(raw_results) > 1:
                time.sleep(cfg.search_delay_seconds)

        sq.results = all_results
        updated_questions.append(sq)

    return {
        "sub_questions": updated_questions,
        "scraped_urls": scraped_urls,
        "used_queries": used_queries,
        "session_domain_counts": session_domain_counts,
        "status": "Search complete",
    }


def summarize_node(state: dict) -> dict:
    """Map-reduce summarization: chunk-extract per source, then synthesize."""
    sub_questions = state.get("sub_questions", [])
    llm_calls = state.get("llm_call_count", 0)

    for sq in sub_questions:
        if not isinstance(sq, SubQuestion):
            continue
        if sq.answered or not sq.results:
            continue

        usable = [r for r in sq.results if len(r.content or "") > 200]
        sq.total_scraped_chars = sum(len(r.content or "") for r in usable)

        console.print(
            f"[magenta]Summarizing[/] [bold]{sq.question[:70]}[/] "
            f"— {len(usable)}/{len(sq.results)} usable ({sq.total_scraped_chars:,} chars)"
        )

        if not usable:
            console.print(f"  [red]No usable sources — marking unanswered[/]")
            continue

        findings: list[str] = []
        for idx, result in enumerate(usable, start=1):
            console.print(f"  • Extracting [{idx}/{len(usable)}]: [bold]{result.title or 'Untitled'}[/]")
            extracted = _extract_from_source(result, sq.question)
            llm_calls += len(_chunk_content(result.content or ""))
            if extracted:
                domain = urlparse(result.url).netloc.replace("www.", "")
                findings.append(f"[{domain}]: {extracted}")

        if not findings:
            console.print(f"  [red]All sources not relevant — marking unanswered[/]")
            continue

        findings_text = "\n\n".join(findings)
        sq.summary = _llm_synthesize(sq.question, findings_text)
        llm_calls += 1
        sq.answered = True
        console.print(f"  ✓ {len(sq.summary)} chars")

    partial_path = write_partial_report(state)
    console.print(f"[dim]Partial report: {partial_path}[/]")

    return {
        "sub_questions": sub_questions,
        "llm_call_count": llm_calls,
        "status": "Summarization complete",
    }


def reflect_node(state: dict) -> dict:
    """Deterministic quality check — no LLM call."""
    cfg = load_config()
    iteration = state.get("iteration", 0) + 1
    console.print(f"[blue]Reflecting[/] iteration {iteration}/{cfg.max_iterations}")

    if iteration >= cfg.max_iterations:
        console.print("[blue]Max iterations reached.[/] Moving to report.")
        return {"iteration": iteration, "gaps": [], "status": f"Iteration {iteration} complete"}

    sub_questions = state.get("sub_questions", [])
    gaps: list[str] = []

    unanswered = [sq.question for sq in sub_questions if isinstance(sq, SubQuestion) and not sq.answered]
    gaps.extend(unanswered)
    if unanswered:
        console.print(f"[blue]Unanswered sub-questions re-queued:[/] {len(unanswered)}")

    for sq in sub_questions:
        if not isinstance(sq, SubQuestion) or not sq.answered:
            continue
        if sq.total_scraped_chars < 500:
            console.print(f"  [yellow]Low content ({sq.total_scraped_chars} chars):[/] {sq.question[:55]}")
            if sq.question not in gaps:
                gaps.append(sq.question)
            continue
        if len(sq.summary or "") < 50:
            console.print(f"  [yellow]Summary too short ({len(sq.summary)} chars):[/] {sq.question[:55]}")
            if sq.question not in gaps:
                gaps.append(sq.question)
            continue
        summary_lower = (sq.summary or "").lower()
        vague_count = sum(1 for p in _VAGUE_PHRASES if p in summary_lower)
        if vague_count > 4:
            console.print(f"  [yellow]Vague summary ({vague_count} phrases):[/] {sq.question[:55]}")
            if sq.question not in gaps:
                gaps.append(sq.question)

    if iteration < 2 and not gaps:
        console.print("[blue]Minimum 2 iterations not yet reached — continuing.[/]")
        unanswered_now = [sq.question for sq in sub_questions if isinstance(sq, SubQuestion) and not sq.answered]
        if unanswered_now:
            gaps = unanswered_now

    if gaps:
        console.print(f"[blue]Gaps:[/] {', '.join(g[:40] for g in gaps)}")
    else:
        console.print("[blue]No gaps.[/] Ready to generate report.")

    return {"iteration": iteration, "gaps": gaps, "status": f"Iteration {iteration} complete"}


def report_node(state: dict) -> dict:
    """Generate a per-section report with proper exec summary and LLM conclusion."""
    cfg = load_config()
    query = state.get("query", "")
    document_title = state.get("document_title", "") or f"Research Report: {query}"
    answered = [sq for sq in state.get("sub_questions", []) if isinstance(sq, SubQuestion) and sq.answered]
    llm_calls = state.get("llm_call_count", 0)
    console.print(f"[green]Generating report[/] — {len(answered)} section(s)")

    # Generate each section body
    sections: list[tuple[str, str]] = []
    for sq in answered:
        heading = _clean_heading(sq.question)
        console.print(f"  • Section: [bold]{heading}[/]")
        body = _llm_section(sq.question, sq.summary)
        llm_calls += 1
        sections.append((heading, body))

    # Executive summary: built from sq.summary fields (short, ≤300 chars each)
    summary_inputs = "\n\n".join(
        f"{_clean_heading(sq.question)}:\n{sq.summary}"
        for sq in answered
        if sq.summary
    )
    exec_summary = _llm_exec_summary(query, summary_inputs)
    llm_calls += 1

    # Conclusion: also from sq.summary fields
    conclusion = _llm_conclusion(query, summary_inputs)
    llm_calls += 1

    # Assemble report
    lines = [f"# {document_title}\n"]
    lines.append("## Summary\n")
    lines.append(exec_summary)
    lines.append("")
    for heading, body in sections:
        lines.append(f"\n## {heading}\n")
        lines.append(body)
        lines.append("")
    lines.append("\n## Conclusion\n")
    lines.append(conclusion)
    report = "\n".join(lines)

    output_dir = ensure_output_dir(cfg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized = re.sub(r"[^a-z0-9]+", "_", query.lower())[:40].strip("_")
    filename = f"{timestamp}_{sanitized or 'report'}.md"
    filepath = output_dir / filename
    filepath.write_text(report, encoding="utf-8")

    console.print(f"[green]Report →[/] {filepath}")
    return {
        "final_report": report,
        "llm_call_count": llm_calls,
        "status": f"Report saved to {filepath}",
    }


console = Console()

# Phrases that indicate a vague, unhelpful summary — used in reflect heuristic
_VAGUE_PHRASES = [
    "in general",
    "it depends",
    "various factors",
    "many aspects",
    "further research",
    "not enough information",
    "unclear",
    "it is difficult",
]


def _render_llm_output(result: Any) -> str:
    try:
        generation = result.generations[0][0]
    except Exception:
        return ""
    if hasattr(generation, "message"):
        content = getattr(generation.message, "content", "") or ""
        if content:
            return content
    return getattr(generation, "text", "") or ""


def _parse_numbered_list(text: str) -> list[str]:
    """Extract only lines that begin with a digit prefix like '1.' or '2)'."""
    lines = text.strip().splitlines()
    items = []
    for line in lines:
        match = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
        if match:
            item = match.group(1).strip()
            if item:
                items.append(item)
    return items


def _clean_heading(question: str) -> str:
    """Strip markdown bold/italic markers and trim for use as a section heading."""
    return re.sub(r"\*{1,3}([^*]*)\*{1,3}", r"\1", question).strip()


# ── LLM call helpers (wrapped with retry) ────────────────────────────────────

@with_llm_retry
def _llm_plan(query: str, gaps: str, num_questions: int) -> str:
    prompt = PLANNER_PROMPT.format_prompt(
        query=truncate_for_llm(query, "query"),
        gaps=truncate_for_llm(gaps, "gaps"),
        num_questions=num_questions,
    )
    return _render_llm_output(get_planner_llm().generate_prompt([prompt])).strip()


@with_llm_retry
def _llm_distill(question: str) -> str:
    prompt = DISTILL_PROMPT.format_prompt(question=truncate_for_llm(question, "question"))
    raw = _render_llm_output(get_planner_llm().generate_prompt([prompt])).strip()
    # Trim to ≤100 chars and remove surrounding quotes the LLM might add
    raw = raw.strip('"\'').split("\n")[0].strip()
    return raw[:100] if raw else question[:80]


@with_llm_retry
def _llm_extract(question: str, title: str, url: str, content: str) -> str:
    prompt = EXTRACT_PROMPT.format_prompt(
        question=truncate_for_llm(question, "question"),
        source_title=title,
        source_url=url,
        content=truncate_for_llm(content, f"source:{url}"),
    )
    return _render_llm_output(get_summarizer_llm().generate_prompt([prompt])).strip()


@with_llm_retry
def _llm_synthesize(question: str, findings: str) -> str:
    prompt = SYNTHESIZE_PROMPT.format_prompt(
        question=truncate_for_llm(question, "question"),
        findings=truncate_for_llm(findings, "findings"),
    )
    return _render_llm_output(get_summarizer_llm().generate_prompt([prompt])).strip()


@with_llm_retry
def _llm_section(question: str, summary: str) -> str:
    prompt = SECTION_PROMPT.format_prompt(
        question=truncate_for_llm(question, "question"),
        summary=truncate_for_llm(summary, "summary"),
    )
    return _render_llm_output(get_summarizer_llm().generate_prompt([prompt])).strip()


@with_llm_retry
def _llm_exec_summary(query: str, section_texts: str) -> str:
    prompt = EXEC_SUMMARY_PROMPT.format_prompt(
        query=truncate_for_llm(query, "query"),
        section_texts=truncate_for_llm(section_texts, "sections"),
    )
    return _render_llm_output(get_summarizer_llm().generate_prompt([prompt])).strip()


# ── Nodes ─────────────────────────────────────────────────────────────────────

def plan_node(state: dict) -> dict:
    query = state.get("query", "")
    gaps = state.get("gaps", []) or []
    gaps_text = "\n".join(f"- {gap}" for gap in gaps) if gaps else "None yet"
    cfg = load_config()
    console.print(f"[yellow]Planning[/] query: [bold]{query}[/]")
    if gaps:
        console.print(f"[yellow]Existing gaps[/]: {', '.join(gaps)}")

    answer = _llm_plan(query, gaps_text, cfg.num_subquestions)
    questions = _parse_numbered_list(answer)

    existing = {
        sq.question: sq
        for sq in state.get("sub_questions", [])
        if isinstance(sq, SubQuestion) and sq.answered
    }
    new_sub_questions = []
    for question in questions:
        if question in existing:
            new_sub_questions.append(existing[question])
        else:
            new_sub_questions.append(SubQuestion(question=question))

    return {
        "sub_questions": new_sub_questions,
        "status": f"Planned {len(new_sub_questions)} sub-questions",
    }


def search_node(state: dict) -> dict:
    cfg = load_config()
    sub_questions = state.get("sub_questions", [])
    scraped_urls: set = set(state.get("scraped_urls") or set())
    console.print(f"[cyan]Searching[/] {len(sub_questions)} sub-question(s), max {cfg.max_results} results each")
    updated_questions = []
    for sq in sub_questions:
        if not isinstance(sq, SubQuestion):
            updated_questions.append(sq)
            continue

        if sq.answered:
            updated_questions.append(sq)
            continue

        # LLM-based query distillation
        search_query = _llm_distill(sq.question)
        console.print(f"[cyan]Searching question:[/] [bold]{sq.question[:80]}[/]")
        console.print(f"  Query: [italic]{search_query}[/]")
        raw_results = raw_search(search_query, cfg.searxng_port, cfg.max_results)
        if not raw_results:
            console.print(f"[red]No search results found for:[/] {search_query}")

        results = []
        for idx, item in enumerate(raw_results, start=1):
            title = item.get("title", "[no title]")
            url = item.get("url", "")
            console.print(f"  • Result {idx}/{len(raw_results)}: [bold]{title}[/] ({url})")
            if url in scraped_urls:
                console.print(f"    Already scraped — skipping duplicate URL")
                content = ""
            else:
                content = raw_scrape(url, cfg.max_page_chars)
                if content:
                    scraped_urls.add(url)
            console.print(f"    Scraped {len(content)} chars from result {idx}")
            results.append(SearchResult(title=title, url=url, snippet=item.get("snippet", ""), content=content))
            if idx < len(raw_results):
                console.print(f"    Waiting {cfg.search_delay_seconds:.1f}s before next result to reduce blocking risk")
                time.sleep(cfg.search_delay_seconds)

        if len(raw_results) > 1:
            console.print(f"  Waiting {cfg.search_delay_seconds:.1f}s before next search question")
            time.sleep(cfg.search_delay_seconds)

        sq.results = results
        updated_questions.append(sq)

    return {"sub_questions": updated_questions, "scraped_urls": scraped_urls, "status": "Search complete"}


def summarize_node(state: dict) -> dict:
    """Map-reduce summarization: extract per source, then synthesize."""
    sub_questions = state.get("sub_questions", [])
    for sq in sub_questions:
        if not isinstance(sq, SubQuestion):
            continue
        if sq.answered or not sq.results:
            continue

        # Filter to sources with enough content
        usable = [r for r in sq.results if len(r.content or "") > 200]
        sq.total_scraped_chars = sum(len(r.content or "") for r in usable)

        total_chars = sum(len(r.content or "") for r in sq.results)
        console.print(
            f"[magenta]Summarizing[/] question: [bold]{sq.question[:80]}[/] "
            f"— {len(usable)}/{len(sq.results)} usable sources ({sq.total_scraped_chars} chars)"
        )

        if len(usable) == 0:
            console.print(f"  [red]No usable sources (all < 200 chars) — marking unanswered for re-search[/]")
            continue

        # Map: extract relevant facts from each source
        findings: list[str] = []
        for idx, result in enumerate(usable, start=1):
            console.print(f"  • Extracting from source {idx}/{len(usable)}: [bold]{result.title or 'Untitled'}[/]")
            extracted = _llm_extract(sq.question, result.title or "Untitled", result.url, result.content or "")
            if extracted and extracted.lower().strip() != "not relevant.":
                findings.append(f"[{result.title or result.url}]: {extracted}")

        if not findings:
            console.print(f"  [red]All sources marked not relevant — marking unanswered[/]")
            continue

        # Reduce: synthesize findings into a single answer
        findings_text = "\n\n".join(findings)
        sq.summary = _llm_synthesize(sq.question, findings_text)
        sq.answered = True
        console.print(f"  ✓ Summarized: {len(sq.summary)} chars")

    # Write in-progress partial report after each summarize pass
    partial_path = write_partial_report(state)
    console.print(f"[dim]Partial report updated: {partial_path}[/]")

    return {"sub_questions": sub_questions, "status": "Summarization complete"}


def reflect_node(state: dict) -> dict:
    """Deterministic quality check — no LLM call."""
    cfg = load_config()
    iteration = state.get("iteration", 0) + 1
    console.print(f"[blue]Reflecting[/] iteration {iteration}/{cfg.max_iterations}")

    if iteration >= cfg.max_iterations:
        console.print("[blue]Max iterations reached.[/] Moving to report.")
        return {"iteration": iteration, "gaps": [], "status": f"Iteration {iteration} complete"}

    sub_questions = state.get("sub_questions", [])
    gaps: list[str] = []

    # Always re-queue unanswered sub-questions
    unanswered = [sq.question for sq in sub_questions if isinstance(sq, SubQuestion) and not sq.answered]
    gaps.extend(unanswered)
    if unanswered:
        console.print(f"[blue]Unanswered sub-questions re-queued:[/] {len(unanswered)}")

    # Check answered questions for quality issues
    for sq in sub_questions:
        if not isinstance(sq, SubQuestion) or not sq.answered:
            continue

        # 1. Insufficient content
        if sq.total_scraped_chars < 500:
            console.print(f"  [yellow]Low content ({sq.total_scraped_chars} chars) for:[/] {sq.question[:60]}")
            if sq.question not in gaps:
                gaps.append(sq.question)
            continue

        # 2. Summary too short
        if len(sq.summary or "") < 50:
            console.print(f"  [yellow]Summary too short ({len(sq.summary)} chars) for:[/] {sq.question[:60]}")
            if sq.question not in gaps:
                gaps.append(sq.question)
            continue

        # 3. Too many vague phrases
        summary_lower = (sq.summary or "").lower()
        vague_count = sum(1 for phrase in _VAGUE_PHRASES if phrase in summary_lower)
        if vague_count > 4:
            console.print(f"  [yellow]Vague summary ({vague_count} vague phrases) for:[/] {sq.question[:60]}")
            if sq.question not in gaps:
                gaps.append(sq.question)

    # Enforce minimum 2 iterations
    if iteration < 2 and not gaps:
        console.print("[blue]Minimum 2 iterations not yet reached — continuing.[/]")
        # Re-queue all unanswered questions; if all answered, add a refinement pass marker
        unanswered_now = [sq.question for sq in sub_questions if isinstance(sq, SubQuestion) and not sq.answered]
        if unanswered_now:
            gaps = unanswered_now

    if gaps:
        console.print(f"[blue]Gaps identified:[/] {', '.join(g[:40] for g in gaps)}")
    else:
        console.print("[blue]No gaps identified.[/] Ready to generate report.")

    return {"iteration": iteration, "gaps": gaps, "status": f"Iteration {iteration} complete"}


def report_node(state: dict) -> dict:
    """Generate a per-section report using SECTION_PROMPT + EXEC_SUMMARY_PROMPT."""
    cfg = load_config()
    query = state.get("query", "")
    answered = [sq for sq in state.get("sub_questions", []) if isinstance(sq, SubQuestion) and sq.answered]
    console.print(f"[green]Generating report[/] from {len(answered)} answered question(s)")

    # Generate each section independently
    sections: list[tuple[str, str]] = []  # (heading, body)
    for sq in answered:
        heading = _clean_heading(sq.question)
        console.print(f"  • Writing section: [bold]{heading[:60]}[/]")
        body = _llm_section(sq.question, sq.summary)
        sections.append((heading, body))

    # Build section texts for executive summary
    section_texts = "\n\n".join(f"## {h}\n{b}" for h, b in sections)

    # Generate executive summary from section text
    exec_summary = _llm_exec_summary(query, section_texts)

    # Assemble report
    lines = [f"# Research Report: {query}\n"]
    lines.append("## Summary\n")
    lines.append(exec_summary)
    lines.append("")
    for heading, body in sections:
        lines.append(f"\n## {heading}\n")
        lines.append(body)
        lines.append("")
    lines.append("\n## Conclusion\n")
    lines.append(f"This report covered {len(answered)} sub-questions about: {query}")
    report = "\n".join(lines)

    output_dir = ensure_output_dir(cfg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized = re.sub(r"[^a-z0-9]+", "_", query.lower())[:40].strip("_")
    filename = f"{timestamp}_{sanitized or 'report'}.md"
    filepath = output_dir / filename
    filepath.write_text(report, encoding="utf-8")

    console.print(f"[green]Report written to[/] {filepath}")
    return {"final_report": report, "status": f"Report saved to {filepath}"}

