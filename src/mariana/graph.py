from pathlib import Path

from langgraph.graph import END, START, StateGraph

from mariana.nodes import (
    check_completion_node,
    finalize_node,
    generate_queries_node,
    init_document_node,
    plan_toc_node,
    process_section_node,
    select_section_node,
)
from mariana.state import ResearchState

_CHECKPOINT_DIR = Path.home() / ".mariana" / "checkpoints"


def route_after_check(state: dict) -> str:
    if state.get("should_stop"):
        return "finalize"
    return "select_section"


def build_graph(checkpointer=None):
    builder = StateGraph(ResearchState)
    builder.add_node("init_document",    init_document_node)
    builder.add_node("plan_toc",         plan_toc_node)
    builder.add_node("select_section",   select_section_node)
    builder.add_node("generate_queries", generate_queries_node)
    builder.add_node("process_section",  process_section_node)
    builder.add_node("check_completion", check_completion_node)
    builder.add_node("finalize",         finalize_node)

    builder.add_edge(START,               "init_document")
    builder.add_edge("init_document",     "plan_toc")
    builder.add_edge("plan_toc",          "select_section")
    builder.add_edge("select_section",    "generate_queries")
    builder.add_edge("generate_queries",  "process_section")
    builder.add_edge("process_section",   "check_completion")
    builder.add_edge("finalize",          END)

    builder.add_conditional_edges(
        "check_completion",
        route_after_check,
        {"select_section": "select_section", "finalize": "finalize"},
    )
    return builder.compile(checkpointer=checkpointer)


def get_graph(use_checkpointing: bool = True):
    """Return a compiled graph, optionally with SQLite checkpointing."""
    if not use_checkpointing:
        return build_graph()

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        db_path = str(_CHECKPOINT_DIR / "mariana.db")
        checkpointer = SqliteSaver.from_conn_string(db_path)
        return build_graph(checkpointer=checkpointer)
    except Exception:
        return build_graph()
