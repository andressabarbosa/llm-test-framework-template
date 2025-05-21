from app.llm_interface import get_response
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_similarity(ref: str, candidate: str) -> float:
    ref_emb = model.encode(ref, convert_to_tensor=True)
    cand_emb = model.encode(candidate, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(ref_emb, cand_emb)[0][0])

def test_regression_known_prompt():
    prompt = "O que é uma célula-tronco?"
    old_response = "Células-tronco são células indiferenciadas com capacidade de se transformar em diferentes tipos celulares."
    new_response = get_response(prompt)
    assert semantic_similarity(old_response, new_response) > 0.85