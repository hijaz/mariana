from mariana.state import ResearchState
from mariana.nodes import plan_node, search_node, summarize_node, reflect_node, report_node
from mariana.utils.config import load_config
from langgraph.graph import StateGraph, START, END


def route_after_reflect(state: dict) -> str:
    gaps = state.get("gaps", [])
    iteration = state.get("iteration", 0)
    max_iter = load_config().max_iterations
    if gaps and iteration < max_iter:
        return "plan"
    return "report"


def build_graph():
    builder = StateGraph(ResearchState)
    builder.add_node("plan", plan_node)
    builder.add_node("search", search_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("report", report_node)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "search")
    builder.add_edge("search", "summarize")
    builder.add_edge("summarize", "reflect")
    builder.add_edge("report", END)
    builder.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"plan": "plan", "report": "report"},
    )
    return builder.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
