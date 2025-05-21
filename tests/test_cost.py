from app.llm_interface import cost_tracker


def test_response_cost() -> None:
    trace_data = list(cost_tracker.values())[-1]
    assert trace_data["cost"] < 0.05
