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

_graph = None


def route_after_check(state: dict) -> str:
    if state.get("should_stop"):
        return "finalize"
    return "select_section"


def build_graph():
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
    return builder.compile()


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
