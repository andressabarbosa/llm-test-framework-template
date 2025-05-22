import re

import pytest

from app.llm_interface import get_response


@pytest.mark.parametrize(
    "prompt,expected_keywords",
    [
        ("Explique a fotossíntese", ["luz", "clorofila", "plantas"]),
        ("Como funciona a fotossíntese?", ["energia", "dióxido de carbono", "oxigênio", "co2"]),
    ],
)
def test_prompt_keywords(prompt: str, expected_keywords: list[str]) -> None:
    response = get_response(prompt)
    for keyword in expected_keywords:
        assert re.search(rf"\b({keyword}|co2|dióxido de carbono)\b", response, re.IGNORECASE)
