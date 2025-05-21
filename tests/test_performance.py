from app.llm_interface import get_response, cost_tracker

def test_response_time():
    response = get_response("Resuma o livro Dom Casmurro")
    trace_data = list(cost_tracker.values())[-1]
    assert trace_data["duration"] < 5