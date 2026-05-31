from types import SimpleNamespace

from mariana.graph import build_graph, route_after_check, route_after_reflect


def test_build_graph_returns_compiled_graph():
    graph = build_graph()
    assert hasattr(graph, "invoke")


# route_after_reflect now always routes to "check"
def test_route_after_reflect_always_goes_to_check(monkeypatch):
    monkeypatch.setattr("mariana.graph.load_config", lambda: SimpleNamespace(max_iterations=3))
    assert route_after_reflect({"gaps": ["missing detail"], "iteration": 0}) == "check"


def test_route_after_reflect_no_gaps_still_goes_to_check(monkeypatch):
    monkeypatch.setattr("mariana.graph.load_config", lambda: SimpleNamespace(max_iterations=3))
    assert route_after_reflect({"gaps": [], "iteration": 0}) == "check"


def test_route_after_reflect_max_iterations_still_goes_to_check(monkeypatch):
    monkeypatch.setattr("mariana.graph.load_config", lambda: SimpleNamespace(max_iterations=3))
    assert route_after_reflect({"gaps": ["more"], "iteration": 3}) == "check"


# route_after_check decides plan vs report
def test_route_after_check_stops_when_should_stop(monkeypatch):
    monkeypatch.setattr("mariana.graph.load_config", lambda: SimpleNamespace(max_iterations=3))
    assert route_after_check({"should_stop": True, "gaps": ["x"], "iteration": 0}) == "report"


def test_route_after_check_continues_with_gaps(monkeypatch):
    monkeypatch.setattr("mariana.graph.load_config", lambda: SimpleNamespace(max_iterations=3))
    assert route_after_check({"should_stop": False, "gaps": ["x"], "iteration": 0}) == "plan"


def test_route_after_check_reports_with_no_gaps(monkeypatch):
    monkeypatch.setattr("mariana.graph.load_config", lambda: SimpleNamespace(max_iterations=3))
    assert route_after_check({"should_stop": False, "gaps": [], "iteration": 0}) == "report"


def test_route_after_check_reports_at_max_iterations(monkeypatch):
    monkeypatch.setattr("mariana.graph.load_config", lambda: SimpleNamespace(max_iterations=3))
    assert route_after_check({"should_stop": False, "gaps": ["x"], "iteration": 3}) == "report"
