from dataclasses import dataclass, field
from typing import Annotated, Any

from langgraph.graph.message import add_messages


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


def initial_state(query: str) -> dict:
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
    }
