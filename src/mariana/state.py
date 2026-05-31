from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Annotated, Any

from langgraph.graph.message import add_messages


@dataclass
class ResearchGoal:
    """Stopping conditions for a research run. Omit all fields for forever mode."""
    max_runtime:      timedelta | None = None
    max_sources:      int | None       = None
    target_words:     int | None       = None
    required_topics:  list[str]        = field(default_factory=list)
    max_iterations:   int | None       = None
    min_sections:     int              = 3
    max_sections:     int              = 4
    max_depth:        int              = 2

    @classmethod
    def from_cli(
        cls,
        duration: str | None = None,
        max_sources: int | None = None,
        target_words: int | None = None,
        max_iterations: int | None = None,
        required_topics: tuple[str, ...] = (),
    ) -> "ResearchGoal":
        from mariana.utils.config import load_config
        cfg = load_config()
        return cls(
            max_runtime=_parse_duration(duration),
            max_sources=max_sources,
            target_words=target_words,
            required_topics=list(required_topics),
            max_iterations=max_iterations if max_iterations is not None else cfg.max_iterations,
            min_sections=cfg.min_sections,
            max_sections=cfg.max_sections,
        )

    def to_dict(self) -> dict:
        return {
            "max_runtime_seconds": self.max_runtime.total_seconds() if self.max_runtime else None,
            "max_sources": self.max_sources,
            "target_words": self.target_words,
            "required_topics": self.required_topics,
            "max_iterations": self.max_iterations,
            "min_sections": self.min_sections,
            "max_sections": self.max_sections,
            "max_depth": self.max_depth,
        }


def _parse_duration(s: str | None) -> timedelta | None:
    """Parse a duration string like '2h', '30m', '1h30m' into a timedelta."""
    if not s:
        return None
    import re
    total = 0
    for value, unit in re.findall(r'(\d+)([hms])', s.lower()):
        v = int(value)
        if unit == 'h':
            total += v * 3600
        elif unit == 'm':
            total += v * 60
        elif unit == 's':
            total += v
    return timedelta(seconds=total) if total > 0 else None


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    content: str = ""


@dataclass
class SubQuestion:
    question: str
    results: list[SearchResult] = field(default_factory=list)
    summary: str = ""
    answered: bool = False
    total_scraped_chars: int = 0
    search_queries: list[str] = field(default_factory=list)


class ResearchState(dict):
    query: str
    sub_questions: list[SubQuestion]
    iteration: int
    gaps: list[str]
    final_report: str
    status: str
    scraped_urls: set
    messages: Annotated[list[Any], add_messages]
    document_title: str
    used_queries: set
    session_domain_counts: dict
    llm_call_count: int
    # v0.6 additions
    doc_id: str
    goal_config: dict
    started_at: str
    should_stop: bool
    stop_reason: str
    total_sources_scraped: int
    total_words_written: int
    consecutive_empty_searches: int


def initial_state(query: str, goal: ResearchGoal | None = None) -> dict:
    from mariana.utils.store import create_document, init_db
    init_db()
    doc_id = create_document(query)
    g = goal or ResearchGoal()
    return {
        "query": query,
        "sub_questions": [],
        "iteration": 0,
        "gaps": [],
        "final_report": "",
        "status": "Starting...",
        "scraped_urls": set(),
        "messages": [],
        "document_title": "",
        "used_queries": set(),
        "session_domain_counts": {},
        "llm_call_count": 0,
        # v0.6
        "doc_id": doc_id,
        "goal_config": g.to_dict(),
        "started_at": datetime.now().isoformat(),
        "should_stop": False,
        "stop_reason": "",
        "total_sources_scraped": 0,
        "total_words_written": 0,
        "consecutive_empty_searches": 0,
    }
