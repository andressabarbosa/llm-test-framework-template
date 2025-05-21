import pytest

from app.llm_interface import get_response


@pytest.mark.parametrize(
    "prompt,expected_keywords",
    [
        ("Explique a fotossíntese", ["luz", "clorofila", "plantas"]),
        ("Como funciona a fotossíntese?", ["energia", "CO2", "oxigênio"]),
    ],
)
def test_prompt_keywords(prompt: str, expected_keywords: list[str]) -> None:
    response = get_response(prompt)
    for keyword in expected_keywords:
        assert keyword.lower() in response.lower()
