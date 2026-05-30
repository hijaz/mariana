import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from mariana.prompts import PLANNER_PROMPT, REPORT_PROMPT, REFLECT_PROMPT, SUMMARIZER_PROMPT
from mariana.state import SearchResult, SubQuestion
from mariana.tools.search import raw_scrape, raw_search
from mariana.utils.config import ensure_output_dir, load_config
from mariana.utils.llm import get_planner_llm, get_summarizer_llm

console = Console()


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


def _distill_search_query(question: str) -> str:
    """Compress a verbose sub-question to a concise search query string."""
    # strip markdown bold/italic markers
    q = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", question)
    # drop a leading "Label:" prefix (e.g. "Background & Fundamental Principles:")
    q = re.sub(r"^[A-Za-z0-9 &,/()'\"\-]+:\s*", "", q)
    # take only the first sentence / question clause
    first = re.split(r"[?.!]", q)[0].strip()
    return first[:100] if first else q[:100]


def plan_node(state: dict) -> dict:
    query = state.get("query", "")
    gaps = state.get("gaps", []) or []
    gaps_text = "\n".join(f"- {gap}" for gap in gaps) if gaps else "None yet"
    cfg = load_config()
    console.print(f"[yellow]Planning[/] query: [bold]{query}[/]")
    if gaps:
        console.print(f"[yellow]Existing gaps[/]: {', '.join(gaps)}")
    prompt = PLANNER_PROMPT.format_prompt(query=query, gaps=gaps_text, num_questions=cfg.num_subquestions)
    result = get_planner_llm().generate_prompt([prompt])
    answer = _render_llm_output(result)
    questions = _parse_numbered_list(answer)

    existing = {sq.question: sq for sq in state.get("sub_questions", []) if isinstance(sq, SubQuestion) and sq.answered}
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

        search_query = _distill_search_query(sq.question)
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
    sub_questions = state.get("sub_questions", [])
    for sq in sub_questions:
        if not isinstance(sq, SubQuestion):
            continue

        if sq.answered or not sq.results:
            continue

        total_chars = sum(len(r.content or "") for r in sq.results)
        console.print(f"[magenta]Summarizing[/] question: [bold]{sq.question[:80]}[/] using {len(sq.results)} source(s) ({total_chars} chars)")
        if total_chars < 300:
            console.print(f"  [red]Insufficient content ({total_chars} chars) — marking unanswered for re-search[/]")
            continue

        sources = []
        for idx, result in enumerate(sq.results, start=1):
            content = result.content or ""
            console.print(f"  • Using source {idx}: [bold]{result.title or 'Untitled'}[/] ({result.url})")
            sources.append(f"### {result.title}\nURL: {result.url}\n\n{content[:2000]}")

        prompt_text = "\n\n---\n\n".join(sources)
        prompt = SUMMARIZER_PROMPT.format_prompt(question=sq.question, sources=prompt_text)
        result = get_summarizer_llm().generate_prompt([prompt])
        sq.summary = _render_llm_output(result).strip()
        sq.answered = True

    updated_questions = [sq for sq in sub_questions]
    return {"sub_questions": updated_questions, "status": "Summarization complete"}


def reflect_node(state: dict) -> dict:
    cfg = load_config()
    iteration = state.get("iteration", 0) + 1
    console.print(f"[blue]Reflecting[/] iteration {iteration}/{cfg.max_iterations}")

    if iteration >= cfg.max_iterations:
        console.print("[blue]Max iterations reached.[/] Moving to report.")
        return {"iteration": iteration, "gaps": [], "status": f"Iteration {iteration} complete"}

    sub_questions = state.get("sub_questions", [])
    summaries = []
    for sq in sub_questions:
        if isinstance(sq, SubQuestion) and sq.answered:
            summaries.append(f"Question: {sq.question}\nSummary: {sq.summary}")
    summaries_text = "\n\n".join(summaries)
    prompt = REFLECT_PROMPT.format_prompt(query=state.get("query", ""), summaries=summaries_text)
    result = get_planner_llm().generate_prompt([prompt])
    raw_text = _render_llm_output(result).strip()

    if raw_text.startswith("```") and raw_text.endswith("```"):
        raw_text = raw_text.strip("`\n")

    gaps = []
    try:
        payload = json.loads(raw_text)
        if not payload.get("sufficient", True):
            gaps = payload.get("gaps", []) or []
    except Exception:
        gaps = []

    unanswered = [
        sq.question for sq in sub_questions
        if isinstance(sq, SubQuestion) and not sq.answered
    ]
    for q in unanswered:
        if q not in gaps:
            gaps.append(q)
    if unanswered:
        console.print(f"[blue]Unanswered sub-questions re-queued as gaps:[/] {len(unanswered)}")

    if gaps:
        console.print(f"[blue]Gaps identified:[/] {', '.join(gaps)}")
    else:
        console.print("[blue]No gaps identified.[/] Ready to generate report.")

    return {"iteration": iteration, "gaps": gaps, "status": f"Iteration {iteration} complete"}


def report_node(state: dict) -> dict:
    cfg = load_config()
    summaries = []
    for sq in state.get("sub_questions", []):
        if isinstance(sq, SubQuestion) and sq.answered:
            summaries.append(f"## {sq.question}\n{sq.summary}")
    summaries_text = "\n\n".join(summaries)

    console.print(f"[green]Generating report[/] from {len(summaries)} answered question(s)")
    prompt = REPORT_PROMPT.format_prompt(query=state.get("query", ""), summaries=summaries_text)
    result = get_summarizer_llm().generate_prompt([prompt])
    report = _render_llm_output(result).strip()

    output_dir = ensure_output_dir(cfg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized = re.sub(r"[^a-z0-9]+", "_", state.get("query", "").lower())[:40].strip("_")
    filename = f"{timestamp}_{sanitized or 'report'}.md"
    filepath = output_dir / filename
    filepath.write_text(report, encoding="utf-8")

    console.print(f"[green]Report written to[/] {filepath}")
    return {"final_report": report, "status": f"Report saved to {filepath}"}
