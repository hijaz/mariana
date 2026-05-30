from types import SimpleNamespace

from mariana.graph import build_graph, route_after_reflect


def test_build_graph_returns_compiled_graph():
    graph = build_graph()
    assert hasattr(graph, "invoke")


def test_route_after_reflect_returns_plan_when_gaps_exist(monkeypatch):
    monkeypatch.setattr("mariana.graph.load_config", lambda: SimpleNamespace(max_iterations=3))
    assert route_after_reflect({"gaps": ["missing detail"], "iteration": 0}) == "plan"


def test_route_after_reflect_returns_report_when_no_gaps(monkeypatch):
    monkeypatch.setattr("mariana.graph.load_config", lambda: SimpleNamespace(max_iterations=3))
    assert route_after_reflect({"gaps": [], "iteration": 0}) == "report"


def test_route_after_reflect_returns_report_when_max_iterations_reached(monkeypatch):
    monkeypatch.setattr("mariana.graph.load_config", lambda: SimpleNamespace(max_iterations=3))
    assert route_after_reflect({"gaps": ["more"], "iteration": 3}) == "report"
