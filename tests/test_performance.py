from app.llm_interface import cost_tracker


def test_response_time() -> None:
    trace_data = list(cost_tracker.values())[-1]
    assert trace_data["duration"] < 5
