from mariana.graph import build_graph, route_after_check


def test_build_graph_returns_compiled_graph():
    graph = build_graph()
    assert hasattr(graph, "invoke")


# route_after_check: stops → finalize, otherwise → select_section
def test_route_after_check_stops_when_should_stop():
    assert route_after_check({"should_stop": True}) == "finalize"


def test_route_after_check_continues_when_not_stopped():
    assert route_after_check({"should_stop": False}) == "select_section"


def test_route_after_check_defaults_to_continue():
    assert route_after_check({}) == "select_section"
