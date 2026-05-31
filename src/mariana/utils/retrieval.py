"""
PageIndex-style tree search retrieval.
Each call is cheap: reads 3-4 section summaries and lets the LLM navigate by reasoning.
"""
import re

from mariana.utils.llm import SimplePrompt
from rich.console import Console

from mariana.utils.store import get_db, load_section_content

console = Console()

TREE_SEARCH_PROMPT = SimplePrompt([
    ("system",
     "You are navigating a research document to find sections relevant "
     "to a query. Read the section summaries and select which ones "
     "are relevant.\n"
     "Output ONLY the numbers of relevant sections, comma-separated. "
     "If none are relevant, output: none\n"
     "Example: 1, 3"),
    ("human",
     "Query: {query}\n\n"
     "Sections:\n{summaries}"),
])

TOPIC_CHECK_PROMPT = SimplePrompt([
    ("system",
     "Does the provided text answer the question? "
     "Reply with exactly one word: YES or NO."),
    ("human", "Question: {question}\n\nText:\n{text}"),
])


def _parse_selection(text: str, max_idx: int) -> list[int]:
    """Parse comma-separated numbers from LLM output. Returns 0-indexed list."""
    if "none" in text.lower():
        return []
    nums = re.findall(r"\d+", text)
    return [int(n) - 1 for n in nums if 0 < int(n) <= max_idx]


def _search_level(
    doc_id: str,
    query: str,
    parent_id: str | None,
    depth: int,
    relevant: list,
    llm,
    max_results: int,
) -> None:
    if depth > 3 or len(relevant) >= max_results:
        return

    with get_db() as db:
        rows = db.execute(
            """SELECT id, title, summary, status, depth
               FROM toc_nodes
               WHERE document_id=?
                 AND parent_id IS ?
                 AND status IN ('complete', 'needs_revision')
               ORDER BY order_index""",
            (doc_id, parent_id),
        ).fetchall()

    nodes = [dict(r) for r in rows]
    if not nodes:
        return

    summaries = "\n".join(
        f"{i + 1}. {n['title']}: {n['summary'] or '(no summary yet)'}"
        for i, n in enumerate(nodes)
    )

    try:
        chain = TREE_SEARCH_PROMPT | llm
        result = chain.invoke({"query": query, "summaries": summaries})
        selected = _parse_selection(getattr(result, "content", ""), len(nodes))
    except Exception:
        selected = []

    for idx in selected:
        node = nodes[idx]
        relevant.append(node["id"])
        _search_level(
            doc_id, query, parent_id=node["id"],
            depth=depth + 1, relevant=relevant,
            llm=llm, max_results=max_results,
        )


def tree_search(
    doc_id: str,
    query: str,
    llm,
    max_results: int = 3,
) -> list[str]:
    """
    PageIndex-style reasoning-based retrieval.
    Returns list of node_ids most relevant to query.
    """
    relevant: list[str] = []
    _search_level(
        doc_id, query,
        parent_id=None, depth=0,
        relevant=relevant, llm=llm,
        max_results=max_results,
    )
    return relevant[:max_results]


def check_required_topics(
    doc_id: str,
    required_topics: list[str],
    llm,
) -> list[str]:
    """
    Returns the subset of required_topics not yet answered by any section.
    Uses tree search to find the most relevant section per topic.
    """
    unanswered: list[str] = []

    for topic in required_topics:
        relevant_ids = tree_search(doc_id, topic, llm, max_results=1)

        if not relevant_ids:
            unanswered.append(topic)
            continue

        content = load_section_content(relevant_ids[0])
        if not content:
            unanswered.append(topic)
            continue

        try:
            chain = TOPIC_CHECK_PROMPT | llm
            result = chain.invoke({
                "question": topic,
                "text": content[:1200],
            })
            if "YES" not in getattr(result, "content", "").upper():
                unanswered.append(topic)
        except Exception:
            unanswered.append(topic)

    return unanswered


def get_related_context(
    doc_id: str,
    current_node_id: str,
    source_summaries: str,
    llm,
) -> str:
    """
    Find completed sections relevant to source_summaries, for cross-referencing.
    Returns a formatted string to inject as context, or empty string.
    """
    relevant_ids = tree_search(doc_id, source_summaries[:400], llm, max_results=2)
    relevant_ids = [nid for nid in relevant_ids if nid != current_node_id]

    if not relevant_ids:
        return ""

    parts = []
    for node_id in relevant_ids:
        content = load_section_content(node_id)
        with get_db() as db:
            row = db.execute(
                "SELECT title FROM toc_nodes WHERE id=?", (node_id,)
            ).fetchone()
        if content and row:
            parts.append(f"[Related — {row['title']}]\n{content[:400]}…")

    return ("\nRELATED SECTIONS:\n" + "\n\n".join(parts)) if parts else ""
