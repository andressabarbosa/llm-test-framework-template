import os
import time
from collections.abc import Callable
from uuid import uuid4

import openai
import promptlayer as pl
from dotenv import load_dotenv

import wandb as wb

try:
    from langsmith import Client, traceable
except ImportError:

    def traceable(*args: object, **kwargs: object) -> Callable:
        return lambda f: f

    Client = None

load_dotenv()

cost_tracker = {}

MODEL_PRICING = {
    "gpt-4": {"input": 0.03, "output": 0.06},
}

# --- Inicialização condicional de ferramentas externas ---


def identity(f: Callable) -> Callable:
    return f


PL_TRACK_DECORATOR: Callable = identity
if os.getenv("PROMPTLAYER_API_KEY"):
    pl.api_key = os.getenv("PROMPTLAYER_API_KEY")
    PL_TRACK_DECORATOR = pl.track

# Weights & Biases
wb.init(  # type: ignore[attr-defined]
    project="llm-test-framework",
    name="test_run",
    mode=os.getenv("WANDB_MODE", "offline"),
)

# LangSmith client (opcional)
if os.getenv("LANGSMITH_API_KEY") and Client:
    client = Client(
        api_url="https://api.smith.langchain.com", api_key=os.getenv("LANGSMITH_API_KEY")
    )

# --- Função principal com rastreamento seguro ---


@traceable(name="LLM Test Call")
@PL_TRACK_DECORATOR
def get_response(
    prompt: str, metadata: dict | None = None, return_trace: bool = False
) -> str | tuple[str, str]:
    """
    Envia um prompt ao modelo GPT-4 com rastreamento de custo e tempo.

    Args:
        prompt (str): prompt enviado ao modelo
        metadata (dict, opcional): dados adicionais com 'trace_id'
        return_trace (bool): se True, retorna também o trace_id

    Returns:
        resposta (str) ou (resposta, trace_id) se return_trace=True
    """
    trace_id = metadata.get("trace_id") if metadata else str(uuid4())
    start = time.time()

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        user=trace_id,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    duration = time.time() - start
    usage = response.usage

    cost = (usage.prompt_tokens / 1000) * MODEL_PRICING["gpt-4"]["input"] + (
        usage.completion_tokens / 1000
    ) * MODEL_PRICING["gpt-4"]["output"]

    cost_tracker[trace_id] = {
        "duration": round(duration, 4),
        "cost": round(cost, 4),
    }

    wb.log(  # type: ignore[attr-defined]
        {
            "prompt": prompt,
            "trace_id": trace_id,
            "duration": duration,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "cost": cost,
        }
    )

    if return_trace:
        return str(response.choices[0].message["content"]), str(trace_id)
    return response.choices[0].message["content"]
