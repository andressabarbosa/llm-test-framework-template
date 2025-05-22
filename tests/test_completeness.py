from app.llm_interface import get_response
from app.similarity import semantic_similarity


def test_photosynthesis_semantic_completeness() -> None:
    prompt = (
        "Descreva detalhadamente a fotossíntese, incluindo de forma explícita o uso do dióxido de carbono, "
        "a luz solar como fonte de energia, a produção de oxigênio e a formação de glicose."
    )
    answer = get_response(prompt)

    expected_concepts = [
        "o dióxido de carbono é utilizado",
        "a luz solar é utilizada como fonte de energia",
        "oxigênio é produzido",
        "glicose é produzida",
    ]
    minimum_score = 0.3
    weak_concepts = []

    for concept in expected_concepts:
        score = semantic_similarity(concept, answer)
        print(f"Conceito: '{concept}' - Similaridade: {score:.2f}")
        if score < minimum_score:
            weak_concepts.append((concept, score))

    assert (
        not weak_concepts
    ), "Alguns conceitos esperados estão ausentes ou pouco explícitos:\n" + "\n".join(
        [f"- '{c}' (similarity={s:.2f})" for c, s in weak_concepts]
    )
