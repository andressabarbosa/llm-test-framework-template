from app.llm_interface import get_response

def test_completude_fotossintese():
    answer = get_response("O que é fotossíntese?")
    required_points = ["dióxido de carbono", "luz solar", "oxigênio", "glicose"]
    for point in required_points:
        assert point in answer.lower()