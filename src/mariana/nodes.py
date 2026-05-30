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
    QUERY_GENERATOR_PROMPT,
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
    "work", "works", "tell", "give", "list", "about",
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


_LEADING_STRIP = {'how', 'what', 'why', 'when', 'where', 'which', 'does', 'do', 'is', 'are', 'can', 'will'}
_TRAILING_STRIP = {'how', 'work', 'works', 'do', 'does', 'like', 'mean', 'about'}


def _strip_function_words(words: list[str]) -> list[str]:
    """Remove leading/trailing function words that add no search value."""
    while words and words[0].lower() in _LEADING_STRIP:
        words = words[1:]
    while words and words[-1].lower() in _TRAILING_STRIP:
        words = words[:-1]
    return words


def validate_and_clean_query(raw: str, fallback_question: str) -> str:
    """Strip markdown artifacts from a distilled query; fall back to keyword extraction."""
    cleaned = raw.strip()
    cleaned = re.sub(r"`+", "", cleaned)
    cleaned = re.sub(r"\*+", "", cleaned)
    cleaned = re.sub(r"^#+\s*", "", cleaned)
    cleaned = re.sub(r"[\[\](){}'\"']", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    words = [w for w in cleaned.split() if len(w) > 1]
    words = _strip_function_words(words)

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


def _extract_topic_keywords(query: str) -> list[str]:
    """Return the most content-bearing words from the research query."""
    QUERY_STOP = {
        'how', 'does', 'do', 'what', 'why', 'when', 'where', 'is', 'are',
        'the', 'a', 'an', 'and', 'or', 'of', 'in', 'to', 'for', 'work',
        'works', 'explain', 'describe', 'tell', 'me', 'about', 'can', 'will',
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
    return [w for w in words if w not in QUERY_STOP][:3]


def _inject_topic(search_query: str, topic_keywords: list[str]) -> str:
    """Prepend the primary topic keyword if the query contains none."""
    query_lower = search_query.lower()
    if any(kw in query_lower for kw in topic_keywords):
        return search_query
    prefix = topic_keywords[0] if topic_keywords else ""
    return f"{prefix} {search_query}".strip() if prefix else search_query


_GENERIC_HEADINGS = {
    'introduction', 'overview', 'background', 'conclusion',
    'challenges', 'applications', 'future directions', 'current research',
    'fundamental physics', 'summary', 'history', 'related work',
    'potential applications', 'future directions research',
    'challenges and obstacles', 'current research developments',
}


def _parse_toc_json(text: str, query: str) -> dict:
    """Parse orchestrator JSON output, hard-capping sections and normalising field names."""
    text = re.sub(r'```(?:json)?', '', text).strip().rstrip('`').strip()
    try:
        toc = json.loads(text)
    except json.JSONDecodeError:
        return {"title": query.title(), "sections": []}
    # Normalise 'queries' → 'search_queries' (new schema uses 'queries')
    for section in toc.get("sections", []):
        if "queries" in section and "search_queries" not in section:
            section["search_queries"] = section.pop("queries")
    # Hard cap — never more than max_sections
    cfg = load_config()
    toc["sections"] = toc.get("sections", [])[:cfg.max_sections]
    return toc


def _sources_are_on_topic(sources: list, topic_keywords: list[str]) -> bool:
    """Return True if any source contains at least one topic keyword in the first 500 chars."""
    combined = " ".join(r.content[:500] for r in sources if r.content).lower()
    return any(kw in combined for kw in topic_keywords)


FALLBACK_ANGLES = [
    "{kw0} {kw1} basic principles explained",
    "{kw0} current technology research 2024",
    "{kw0} {kw1} challenges limitations",
    "{kw0} future applications progress",
]


def _deterministic_queries(query: str, n: int) -> list[str]:
    """Generate n topic-specific search queries without any LLM call."""
    keywords = _extract_topic_keywords(query)
    kw0 = keywords[0] if keywords else query.split()[0].lower()
    kw1 = keywords[1] if len(keywords) > 1 else ""
    queries = []
    for template in FALLBACK_ANGLES[:n]:
        q = template.format(kw0=kw0, kw1=kw1).strip()
        q = re.sub(r'\s+', ' ', q).strip()
        queries.append(q)
    return queries




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


def _extract_from_source(source: SearchResult, question: str) -> tuple[str, int]:
    """Extract key points from one source, chunked if long. Returns (text, llm_calls)."""
    chunks = _chunk_content(source.content or "")
    domain = urlparse(source.url).netloc.replace("www.", "")
    points = []
    calls = 0
    for chunk in chunks:
        point = _llm_extract_chunk(question, domain, chunk)
        calls += 1
        if point and "NOT RELEVANT" not in point.upper() and len(point) > 20:
            points.append(point)
    if not points:
        return "", calls
    return " ".join(points), calls


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
    chain = CONCLUSION_PROMPT | get_summarizer_llm()
    result = chain.invoke({
        "query": truncate_for_llm(query, "query"),
        "findings": truncate_for_llm(findings, "findings"),
    })
    return (getattr(result, "content", None) or "").strip()


@with_llm_retry
def _llm_exec_summary(query: str, summary_inputs: str) -> str:
    prompt = EXEC_SUMMARY_PROMPT.format_prompt(
        query=truncate_for_llm(query, "query"),
        section_texts=truncate_for_llm(summary_inputs, "exec_summary_inputs"),
    )
    return _render_llm_output(get_summarizer_llm().generate_prompt([prompt])).strip()


@with_llm_retry
def _llm_generate_queries_raw(query: str, n: int) -> str:
    """Ask the LLM for n search queries as a numbered list."""
    chain = QUERY_GENERATOR_PROMPT | get_planner_llm()
    result = chain.invoke({
        "query": truncate_for_llm(query, "query"),
        "n": n,
    })
    return (getattr(result, "content", None) or "").strip()


def _llm_generate_queries(query: str, n: int) -> list[str]:
    """Call LLM to generate n search queries; parse numbered list output."""
    raw = _llm_generate_queries_raw(query, n)
    return _parse_numbered_list(raw)


# ── Nodes ─────────────────────────────────────────────────────────────────────

def orchestrator_node(state: dict) -> dict:
    """Plan research by generating N targeted search queries as a numbered list."""
    query = state.get("query", "")
    gaps = state.get("gaps", []) or []
    cfg = load_config()
    used_queries: set = set(state.get("used_queries") or set())
    topic_keywords = _extract_topic_keywords(query)

    console.print(f"[yellow]Planning[/] research for: [bold]{query}[/]")
    if gaps:
        console.print(f"[yellow]Addressing gaps[/]: {', '.join(g[:50] for g in gaps)}")

    # Focus on gaps when present, otherwise research fresh
    planner_query = (
        f"{query} — specifically: {', '.join(gaps[:3])}" if gaps else query
    )

    raw_queries = _llm_generate_queries(planner_query, cfg.max_sections)

    # Validate every query and inject topic keywords as safety net
    validated: list[str] = []
    for q in raw_queries:
        if not q.strip():
            continue
        clean = validate_and_clean_query(q, query)
        validated.append(_inject_topic(clean, topic_keywords))
    validated = [q for q in validated if q][:cfg.max_sections]

    # Deterministic fallback — always produce min_sections good queries
    if len(validated) < cfg.min_sections:
        console.print(
            f"[yellow]Planner returned {len(validated)} queries "
            f"(min {cfg.min_sections}) — using deterministic fallback[/]"
        )
        validated = _deterministic_queries(query, cfg.max_sections)

    # Build SubQuestion objects — query text doubles as heading and search query
    existing = {
        sq.question: sq
        for sq in state.get("sub_questions", [])
        if isinstance(sq, SubQuestion) and sq.answered
    }
    sub_questions: list[SubQuestion] = []
    for q in validated:
        heading = q.title()
        deduped = _unique_query(q, used_queries)
        if heading in existing:
            existing[heading].search_queries = [deduped]
            sub_questions.append(existing[heading])
        else:
            sub_questions.append(SubQuestion(question=heading, search_queries=[deduped]))

    document_title = query.title()
    console.print(f"[yellow]Sections:[/] {', '.join(sq.question[:40] for sq in sub_questions)}")
    return {
        "sub_questions": sub_questions,
        "document_title": document_title,
        "used_queries": used_queries,
        "llm_call_count": state.get("llm_call_count", 0) + 1,
        "status": f"Planned {len(sub_questions)} sections",
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
                # Adaptive delay: shorter when we got good content
                if idx < len(diverse):
                    delay = cfg.search_delay_seconds * (0.5 if len(content) > 500 else 1.0)
                    time.sleep(delay)

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
    topic_keywords = _extract_topic_keywords(state.get("query", ""))

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

        # Skip sections where all scraped content is off-topic
        if topic_keywords and not _sources_are_on_topic(usable, topic_keywords):
            console.print(
                f"  [red]Sources off-topic (no '{', '.join(topic_keywords)}' found) — skipping[/]"
            )
            sq.answered = False
            sq.search_queries = [
                _inject_topic(q, topic_keywords) for q in (sq.search_queries or [])
            ]
            continue

        findings: list[str] = []
        for idx, result in enumerate(usable, start=1):
            console.print(f"  • Extracting [{idx}/{len(usable)}]: [bold]{result.title or 'Untitled'}[/]")
            extracted, calls = _extract_from_source(result, sq.question)
            llm_calls += calls
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

