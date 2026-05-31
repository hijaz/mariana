from pathlib import Path

from langgraph.graph import END, START, StateGraph

from mariana.nodes import (
    check_completion_node,
    orchestrator_node,
    reflect_node,
    report_node,
    save_to_store_node,
    search_node,
    summarize_node,
)
from mariana.state import ResearchState
from mariana.utils.config import load_config

_CHECKPOINT_DIR = Path.home() / ".mariana" / "checkpoints"


def route_after_reflect(state: dict) -> str:
    gaps = state.get("gaps", [])
    iteration = state.get("iteration", 0)
    max_iter = load_config().max_iterations
    if gaps and iteration < max_iter:
        return "check"
    return "check"   # always pass through check before reporting


def route_after_check(state: dict) -> str:
    if state.get("should_stop"):
        return "report"
    gaps = state.get("gaps", [])
    iteration = state.get("iteration", 0)
    max_iter = load_config().max_iterations
    if gaps and iteration < max_iter:
        return "plan"
    return "report"


def build_graph(checkpointer=None):
    builder = StateGraph(ResearchState)
    builder.add_node("plan", orchestrator_node)
    builder.add_node("search", search_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("save", save_to_store_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("check", check_completion_node)
    builder.add_node("report", report_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "search")
    builder.add_edge("search", "summarize")
    builder.add_edge("summarize", "save")
    builder.add_edge("save", "reflect")
    builder.add_edge("report", END)

    builder.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"check": "check"},
    )
    builder.add_conditional_edges(
        "check",
        route_after_check,
        {"plan": "plan", "report": "report"},
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
        # Fall back to no checkpointing if sqlite saver unavailable
        return build_graph()

