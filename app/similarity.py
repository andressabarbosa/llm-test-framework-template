import os

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small",
    )
    return response.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot_product / (norm_a * norm_b)


def semantic_similarity(text1: str, text2: str) -> float:
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    return cosine_similarity(emb1, emb2)
