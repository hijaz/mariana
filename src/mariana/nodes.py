"""Research nodes — URL-by-URL incremental generation pipeline."""
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from rich.console import Console

from mariana.prompts import (
    CONCLUSION_PROMPT,
    EXEC_SUMMARY_PROMPT,
    EXTRACT_PROMPT,
    FOLLOW_UP_PROMPT,
    QUERY_GENERATOR_PROMPT,
    SUMMARIZE_NODE_PROMPT,
    SYNTHESIZE_PROMPT,
    TOC_PLANNER_PROMPT,
)
from mariana.state import ResearchGoal
from mariana.tools.search import filter_for_diversity, raw_scrape, raw_search
from mariana.utils.config import ensure_output_dir, load_config
from mariana.utils.events import emit
from mariana.utils.llm import get_planner_llm, get_summarizer_llm, truncate_for_llm
from mariana.utils.retry import with_llm_retry
from mariana.utils.store import (
    create_toc_node,
    get_pending_sections,
    get_toc,
    load_section_content,
    save_section_content,
    update_node_status,
    update_node_summary,
)

console = Console()

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

# ── Pure helpers ──────────────────────────────────────────────────────────────

def _parse_numbered_list(text: str) -> list[str]:
    """Extract lines beginning with a digit prefix like '1.' or '2)'."""
    items = []
    for line in text.strip().splitlines():
        m = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
        if m:
            item = m.group(1).strip()
            if item:
                items.append(item)
    return items


_QUERY_STOP = {
    'how', 'does', 'do', 'what', 'why', 'when', 'where', 'is', 'are',
    'the', 'a', 'an', 'and', 'or', 'of', 'in', 'to', 'for', 'work',
    'works', 'explain', 'describe', 'tell', 'me', 'about', 'can', 'will',
}


def _extract_topic_keywords(query: str) -> list[str]:
    """Return the most content-bearing words from the research query."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
    return [w for w in words if w not in _QUERY_STOP][:3]


FALLBACK_ANGLES = [
    "{kw0} {kw1} basic principles explained",
    "{kw0} current technology research 2024",
    "{kw0} {kw1} challenges limitations",
    "{kw0} future applications progress",
]


def _deterministic_queries(topic: str, n: int) -> list[str]:
    """Generate n topic-specific search queries without any LLM call."""
    keywords = _extract_topic_keywords(topic)
    kw0 = keywords[0] if keywords else topic.split()[0].lower()
    kw1 = keywords[1] if len(keywords) > 1 else ""
    queries = []
    for template in FALLBACK_ANGLES[:n]:
        q = re.sub(r'\s+', ' ', template.format(kw0=kw0, kw1=kw1)).strip()
        queries.append(q)
    return queries


def _chunk_content(text: str) -> list[str]:
    if not text or len(text) <= CHUNK_SIZE:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            last_period = text.rfind('.', start + int(CHUNK_SIZE * 0.6), end)
            if last_period > start + int(CHUNK_SIZE * 0.6):
                end = last_period + 1
        chunks.append(text[start:end])
        if end >= len(text):
            break  # reached the end — do not subtract overlap or we loop forever
        start = end - CHUNK_OVERLAP
    return chunks


def _should_skip(url: str) -> bool:
    """True for URLs that never contain research content."""
    skip = [
        'google.com/search', 'bing.com/search', 'yahoo.com/search',
        'amazon.com/', 'twitter.com/', 'x.com/status', 'facebook.com/',
        'instagram.com/', 'youtube.com/results', 'linkedin.com/in/',
        '/login', '/signup', '/register', '/cart', '/checkout',
    ]
    return any(p in url for p in skip)


# ── LLM helpers ───────────────────────────────────────────────────────────────

@with_llm_retry
def _extract_best_point(chunks: list[str], section_title: str, domain: str, llm) -> str:
    """
    Run EXTRACT_PROMPT over each chunk.
    Join valid results; return empty string if nothing relevant found.
    """
    points = []
    for chunk in chunks:
        result = (EXTRACT_PROMPT | llm).invoke({
            "section_title": section_title,
            "domain": domain,
            "content": chunk.replace("{", "{{").replace("}", "}}"),
        })
        point = (getattr(result, "content", None) or "").strip()
        if point and "NOT RELEVANT" not in point.upper() and len(point) > 15:
            points.append(point)
    return " ".join(points)


@with_llm_retry
def _synthesize_section(section_title: str, points: list[str], llm) -> str:
    """Synthesize a list of one-sentence findings into flowing prose."""
    result = (SYNTHESIZE_PROMPT | llm).invoke({
        "section_title": section_title,
        "points": "\n".join(f"- {p}" for p in points),
    })
    return (getattr(result, "content", None) or "").strip()


@with_llm_retry
def _llm_generate_queries_raw(topic: str, n: int) -> str:
    chain = QUERY_GENERATOR_PROMPT | get_planner_llm()
    result = chain.invoke({"query": truncate_for_llm(topic, "query"), "n": n})
    return (getattr(result, "content", None) or "").strip()


@with_llm_retry
def _llm_section_titles_raw(query: str, n: int) -> str:
    chain = TOC_PLANNER_PROMPT | get_planner_llm()
    result = chain.invoke({"query": truncate_for_llm(query, "query"), "n": n})
    return (getattr(result, "content", None) or "").strip()


@with_llm_retry
def _llm_conclusion(query: str, findings: str) -> str:
    chain = CONCLUSION_PROMPT | get_summarizer_llm()
    result = chain.invoke({
        "query": truncate_for_llm(query, "query"),
        "findings": truncate_for_llm(findings, "findings"),
    })
    return (getattr(result, "content", None) or "").strip()


@with_llm_retry
def _llm_exec_summary(query: str, section_texts: str) -> str:
    chain = EXEC_SUMMARY_PROMPT | get_summarizer_llm()
    result = chain.invoke({
        "query": truncate_for_llm(query, "query"),
        "section_texts": truncate_for_llm(section_texts, "section_texts"),
    })
    return (getattr(result, "content", None) or "").strip()


def _generate_node_summary(title: str, text: str) -> str:
    """Generate a one-sentence summary for a completed section."""
    try:
        chain = SUMMARIZE_NODE_PROMPT | get_summarizer_llm()
        result = chain.invoke({"title": title, "text": text[:1200]})
        return (getattr(result, "content", None) or "").strip()
    except Exception:
        return text[:80].replace("\n", " ")


def _seed_queries_from_content(
    doc_id: str,
    node_id: str,
    section_title: str,
    content: str,
    current_generation: int,
    llm,
) -> None:
    """Generate follow-up topics from completed section for forever-mode expansion."""
    if current_generation >= 2:
        return
    existing_toc = get_toc(doc_id)
    if len(existing_toc) >= 8:
        return
    try:
        result = (FOLLOW_UP_PROMPT | llm).invoke({
            "section_title": section_title,
            "content": content[:600],
        })
        topics = _parse_numbered_list(getattr(result, "content", "") or "")
    except Exception:
        return
    existing_titles = {n["title"].lower() for n in existing_toc}
    added = 0
    for topic in topics[:2]:
        clean = topic.strip().title()
        if clean.lower() not in existing_titles:
            create_toc_node(doc_id, clean, depth=1, order_index=100 + len(existing_toc) + added)
            added += 1
    if added:
        console.print(f"  [dim]Seeded {added} follow-up topic(s)[/]")


# ── Report helpers ────────────────────────────────────────────────────────────

def _write_incremental_report(
    doc_id: str,
    query: str,
    document_title: str,
    cfg,
) -> Path | None:
    """Write the current state of the report from all completed sections in the DB."""
    sections = [n for n in get_toc(doc_id) if n["status"] == "complete" and n["depth"] == 0]
    if not sections:
        return None
    output_dir = ensure_output_dir(cfg)
    filepath = output_dir / f"{doc_id[:8]}_partial.md"
    title = document_title or query.title()
    lines = [f"# {title}\n", "*Research in progress…*\n"]
    for node in sorted(sections, key=lambda n: n["order_index"]):
        content = load_section_content(node["id"])
        if content:
            lines.append(f"\n## {node['title']}\n")
            lines.append(content)
            lines.append("")
    filepath.write_text("\n".join(lines), encoding="utf-8")
    return filepath


# ── Nodes ─────────────────────────────────────────────────────────────────────

def init_document_node(state: dict) -> dict:
    """Create the DB document record and derive document_title."""
    from mariana.utils.store import create_document, init_db
    cfg = load_config()
    init_db()
    query = state["query"]
    doc_id = create_document(query)
    # Simple title: capitalise query, cap at 60 chars
    title = query.strip().rstrip("?").title()[:60]
    emit("doc_created", {"doc_id": doc_id, "query": query, "title": title}, doc_id=doc_id)
    return {
        "doc_id": doc_id,
        "document_title": title,
        "status": "Document initialised",
    }


def plan_toc_node(state: dict) -> dict:
    """Generate section titles and insert them as toc_nodes."""
    cfg = load_config()
    query = state["query"]
    doc_id = state["doc_id"]
    goal_cfg = state.get("goal_config", {})
    g = ResearchGoal.from_dict(goal_cfg)

    # How many sections?
    n = min(g.max_sources // 3, 6) if g.max_sources else 5
    n = max(n, 3)

    try:
        raw = _llm_section_titles_raw(query, n)
        titles = _parse_numbered_list(raw)
        if len(titles) < 2:
            raise ValueError("Not enough titles")
    except Exception:
        titles = _deterministic_queries(query, n)

    for idx, title in enumerate(titles[:n]):
        create_toc_node(doc_id, title, depth=0, order_index=idx)

    console.print(f"[bold]TOC:[/] {len(titles[:n])} sections planned")
    emit("toc_planned", {"sections": titles[:n]}, doc_id=doc_id)
    return {
        "llm_call_count": state.get("llm_call_count", 0) + 1,
        "status": f"TOC planned: {len(titles[:n])} sections",
    }


def select_section_node(state: dict) -> dict:
    """Pick the next pending section and load it into state."""
    from mariana.utils.cancel import is_cancelled
    if is_cancelled():
        return {
            "should_stop": True,
            "stop_reason": "Cancelled by user",
            "status": "Cancelled",
        }
    doc_id = state["doc_id"]
    pending = get_pending_sections(doc_id)
    if not pending:
        # Safety valve — shouldn't happen normally
        return {
            "should_stop": True,
            "stop_reason": "No pending sections remaining",
            "status": "No sections left",
        }
    section = pending[0]
    update_node_status(section["id"], "active")
    return {
        "current_node_id": section["id"],
        "current_section_title": section["title"],
        "current_search_queries": [],
        "query_generation": state.get("query_generation", 0),
        "status": f"Working on: {section['title']}",
    }


def generate_queries_node(state: dict) -> dict:
    """Generate search queries for the current section."""
    cfg = load_config()
    section_title = state["current_section_title"]
    query = state["query"]
    gen = state.get("query_generation", 0)

    combined_topic = f"{query} — {section_title}"
    try:
        raw = _llm_generate_queries_raw(combined_topic, 4)
        queries = _parse_numbered_list(raw)
        if not queries:
            raise ValueError("No queries generated")
    except Exception:
        queries = _deterministic_queries(combined_topic, 4)

    doc_id = state.get("doc_id", "")
    emit("queries_generated", {"section": section_title, "queries": queries}, doc_id=doc_id)
    return {
        "current_search_queries": queries,
        "query_generation": gen + 1,
        "llm_call_count": state.get("llm_call_count", 0) + 1,
        "status": f"Generated {len(queries)} queries for: {section_title}",
    }


def process_section_node(state: dict) -> dict:
    """
    For each query → search → for each URL: scrape → extract one point →
    resynthesize section → save to DB → write partial report.

    This is the core URL-by-URL incremental loop.
    """
    cfg = load_config()
    llm = get_summarizer_llm()

    doc_id = state["doc_id"]
    section_title = state["current_section_title"]
    node_id = state["current_node_id"]
    queries = state.get("current_search_queries", [])
    query = state["query"]
    goal_cfg = state.get("goal_config", {})
    g = ResearchGoal.from_dict(goal_cfg)

    scraped_urls: set = state.get("scraped_urls", set())
    session_domain_counts: dict = state.get("session_domain_counts", {})
    total_sources = state.get("total_sources_scraped", 0)
    total_words = state.get("total_words_written", 0)
    llm_calls = state.get("llm_call_count", 0)
    consec_empty = state.get("consecutive_empty_searches", 0)

    # Load any previously accumulated points for this section
    existing_content = load_section_content(node_id) or ""
    points: list[str] = [
        line.lstrip("- ").strip()
        for line in existing_content.splitlines()
        if line.startswith("- ")
    ] if existing_content.startswith("- ") else []
    # If content is already synthesised prose, keep it and treat as 1 point
    if existing_content and not points:
        points = [existing_content[:600]]

    goal_met = g.max_sources and total_sources >= g.max_sources
    stop_processing = False

    for qi, search_query in enumerate(queries, 1):
        if stop_processing:
            break
        from mariana.utils.cancel import is_cancelled
        if is_cancelled():
            stop_processing = True
            break
        if goal_met:
            stop_processing = True
            break

        console.print(f"  [dim]  query {qi}/{len(queries)}:[/] {search_query}")
        try:
            raw_results = raw_search(search_query, cfg.searxng_port, cfg.max_results)
        except Exception as exc:
            console.print(f"  [yellow]  search error:[/] {exc}")
            raw_results = []

        section_domain_counts: dict = {}
        diverse_results = filter_for_diversity(raw_results, section_domain_counts, session_domain_counts)
        console.print(f"  [dim]  {len(raw_results)} results → {len(diverse_results)} after diversity filter[/]")

        if not diverse_results:
            consec_empty += 1
        else:
            consec_empty = 0

        for result in diverse_results:
            if stop_processing:
                break

            url = result.get("url") if isinstance(result, dict) else (getattr(result, "url", None) or "")
            if not url or url in scraped_urls or _should_skip(url):
                continue

            domain = urlparse(url).netloc.replace("www.", "")
            console.print(f"  [dim]  scraping {domain} …[/]")
            emit("url_visiting", {"url": url, "domain": domain, "section": section_title}, doc_id=doc_id)
            try:
                content = raw_scrape(url, max_chars=cfg.max_page_chars)
            except Exception as exc:
                console.print(f"  [dim]  scrape failed {domain}: {exc}[/]")
                emit("url_failed", {"url": url, "domain": domain, "error": str(exc)}, doc_id=doc_id)
                continue

            if not content or len(content) < 100:
                console.print(f"  [dim]  {domain}: empty/too short, skipping[/]")
                emit("url_failed", {"url": url, "domain": domain, "error": "empty"}, doc_id=doc_id)
                continue

            scraped_urls.add(url)
            total_sources += 1
            session_domain_counts[domain] = session_domain_counts.get(domain, 0) + 1
            console.print(f"  [dim]  {domain}: {len(content)} chars → extracting point (LLM #{llm_calls+1}) …[/]")

            chunks = _chunk_content(content)
            point = _extract_best_point(chunks, section_title, domain, llm)
            llm_calls += 1

            if point:
                points.append(f"[{domain}] {point}")
                console.print(f"  [green]+[/] {domain}: {point[:80]}…" if len(point) > 80 else f"  [green]+[/] {domain}: {point}")

                # Resynthesize every time we have a new point
                console.print(f"  [dim]  synthesising section (LLM #{llm_calls+1}) …[/]")
                section_text = _synthesize_section(section_title, points, llm)
                llm_calls += 1

                if section_text:
                    save_section_content(node_id, section_text)
                    total_words = sum(
                        len((load_section_content(n["id"]) or "").split())
                        for n in get_toc(doc_id)
                        if n.get("status") in ("active", "complete")
                    )
                    _write_incremental_report(doc_id, query, state.get("document_title", ""), cfg)
                    emit("section_updated", {"section": section_title, "node_id": node_id, "text": section_text, "total_words": total_words, "total_sources": total_sources}, doc_id=doc_id)
                emit("url_done", {"url": url, "domain": domain, "point": point, "section": section_title, "relevant": True}, doc_id=doc_id)
            else:
                console.print(f"  [dim]  {domain}: NOT RELEVANT[/]")
                emit("url_done", {"url": url, "domain": domain, "point": None, "section": section_title, "relevant": False}, doc_id=doc_id)

            # Check goal after each URL
            if g.max_sources and total_sources >= g.max_sources:
                stop_processing = True
                break
            if g.target_words and total_words >= g.target_words:
                stop_processing = True
                break

        console.print(f"  [dim]  sleeping {cfg.search_delay_seconds}s …[/]")
        time.sleep(cfg.search_delay_seconds)

    # Finalise section
    final_content = load_section_content(node_id) or ""
    if final_content:
        summary = _generate_node_summary(section_title, final_content)
        update_node_summary(node_id, summary)
        update_node_status(node_id, "complete")
        # Try to seed follow-up topics if we're in forever-mode
        current_gen = state.get("query_generation", 0)
        _seed_queries_from_content(doc_id, node_id, section_title, final_content, current_gen, llm)
    else:
        update_node_status(node_id, "complete")  # nothing found, move on

    return {
        "scraped_urls": scraped_urls,
        "session_domain_counts": session_domain_counts,
        "total_sources_scraped": total_sources,
        "total_words_written": total_words,
        "consecutive_empty_searches": consec_empty,
        "llm_call_count": llm_calls,
        "iteration": state.get("iteration", 0) + 1,
        "status": f"Completed section: {section_title}",
    }


def check_completion_node(state: dict) -> dict:
    """Decide whether to stop or continue to the next section."""
    from mariana.utils.retrieval import check_required_topics

    doc_id = state.get("doc_id", "")
    goal_cfg = state.get("goal_config", {})
    g = ResearchGoal.from_dict(goal_cfg)

    reason = ""

    # 1. Goal: all sections processed
    pending = get_pending_sections(doc_id)
    if not pending:
        reason = "All sections completed"

    # 2. Goal: max sources
    if not reason and g.max_sources:
        if state.get("total_sources_scraped", 0) >= g.max_sources:
            reason = f"Reached max sources ({g.max_sources})"

    # 3. Goal: max words
    if not reason and g.target_words:
        if state.get("total_words_written", 0) >= g.target_words:
            reason = f"Reached max words ({g.target_words})"

    # 4. Goal: max duration
    if not reason and g.max_duration_seconds:
        started = state.get("started_at", "")
        if started:
            from datetime import datetime
            elapsed = (datetime.now() - datetime.fromisoformat(started)).total_seconds()
            if elapsed >= g.max_duration_seconds:
                reason = f"Reached time limit ({g.max_duration_seconds}s)"

    # 5. Goal: required topics covered
    if not reason and g.require_topics:
        toc = get_toc(doc_id) if doc_id else []
        completed_titles = [n["title"] for n in toc if n.get("status") == "complete"]
        if completed_titles:
            missing = check_required_topics(g.require_topics, completed_titles)
            if missing:
                console.print(f"[dim]Still need: {missing}[/]")
            # Don't stop just for this — let section loop handle it

    # 6. Consecutive empty searches safety valve
    if not reason and state.get("consecutive_empty_searches", 0) >= 5:
        reason = "5 consecutive searches returned no usable content"

    if reason:
        console.print(f"[blue]Stopping:[/] {reason}")
    else:
        console.print("[blue]Completion check:[/] continuing research")

    return {
        "should_stop": bool(reason),
        "stop_reason": reason,
        "status": f"Completion check: {'stop' if reason else 'continue'}",
    }


def finalize_node(state: dict) -> dict:
    """Write the final Markdown report from all completed sections."""
    cfg = load_config()
    doc_id = state.get("doc_id", "")
    query = state.get("query", "")
    document_title = state.get("document_title", query.title())

    toc = get_toc(doc_id) if doc_id else []
    completed = [n for n in toc if n["status"] == "complete" and n["depth"] == 0]

    if not completed:
        console.print("[yellow]No completed sections — nothing to write.[/]")
        return {"status": "No content to write"}

    # Build sections text for exec summary
    sections_for_summary: list[str] = []
    for node in sorted(completed, key=lambda n: n["order_index"]):
        content = load_section_content(node["id"]) or ""
        if content:
            sections_for_summary.append(f"## {node['title']}\n{content}")

    section_texts = "\n\n".join(sections_for_summary)

    exec_summary = _llm_exec_summary(query, section_texts)
    conclusion = _llm_conclusion(query, section_texts)

    # Assemble full report
    lines = [f"# {document_title}\n"]
    if exec_summary:
        lines.append("## Summary\n")
        lines.append(exec_summary)
        lines.append("")
    for text in sections_for_summary:
        lines.append(text)
        lines.append("")
    if conclusion:
        lines.append("## Conclusion\n")
        lines.append(conclusion)
        lines.append("")

    report = "\n".join(lines)
    output_dir = ensure_output_dir(cfg)
    safe_title = re.sub(r"[^\w\s-]", "", document_title.lower())
    safe_title = re.sub(r"[\s]+", "_", safe_title)[:60]
    report_path = output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_title}.md"
    report_path.write_text(report, encoding="utf-8")

    # Remove partial file
    partial = output_dir / f"{doc_id[:8]}_partial.md"
    if partial.exists():
        partial.unlink()

    word_count = len(report.split())
    console.print(f"[bold green]Report written:[/] {report_path} ({word_count} words)")
    emit("report_ready", {"path": str(report_path), "word_count": word_count, "report": report}, doc_id=doc_id)

    return {
        "final_report": str(report_path),
        "total_words_written": word_count,
        "llm_call_count": state.get("llm_call_count", 0) + 2,
        "status": "Report complete",
    }
