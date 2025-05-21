from app.llm_interface import get_response, cost_tracker

def test_response_cost():
    response = get_response("Explique como funciona a fotossíntese com detalhes")
    trace_data = list(cost_tracker.values())[-1]
    assert trace_data["cost"] < 0.05