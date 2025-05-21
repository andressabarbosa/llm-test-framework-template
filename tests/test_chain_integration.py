import os

from app.llm_chain import run_chain, run_graph_chain, run_rag_chain


def test_chain_output_schema() -> None:
    result = run_chain("Explique o que é fotossíntese e aponte conceitos importantes envolvidos.")
    assert "resumo" in result
    assert "conceitos_chave" in result
    assert isinstance(result["conceitos_chave"], list)
    assert len(result["conceitos_chave"]) > 0


def test_rag_response() -> None:
    docs_path = os.path.join("data", "golden_answers.json")
    response = run_rag_chain("O que é fotossíntese?", docs_path)
    assert isinstance(response, str)
    assert len(response) > 20


def test_graph_chain() -> None:
    question = "Qual a relação entre fotossíntese e dióxido de carbono?"
    result = run_graph_chain(question)
    assert isinstance(result, str)
    assert "fotossíntese" in result.lower() or "co2" in result.lower()
