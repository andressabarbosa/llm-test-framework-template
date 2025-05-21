import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, Union
from uuid import uuid4

import promptlayer as pl
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

import wandb as wb

if TYPE_CHECKING:
    from langsmith import Client

try:
    from langsmith import Client as LangsmithClient
    from langsmith import traceable as langsmith_traceable
except ImportError:

    def dummy_traceable(*args: object, **kwargs: object) -> Callable:
        return lambda f: f

    langsmith_traceable = dummy_traceable
    LangsmithClient = None  # type: ignore

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

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
wandb_mode: Literal["online", "offline", "disabled"]
_raw_mode = os.getenv("WANDB_MODE")
if _raw_mode in {"online", "offline", "disabled"}:
    wandb_mode = _raw_mode  # type: ignore[assignment]
else:
    wandb_mode = "offline"

wb.init(  # type: ignore[attr-defined]
    project="llm-test-framework",
    name="test_run",
    mode=wandb_mode,
)

# LangSmith client (opcional)
langsmith_client: Union["Client", None] = None
if os.getenv("LANGSMITH_API_KEY") and LangsmithClient is not None:
    langsmith_client = LangsmithClient(
        api_url="https://api.smith.langchain.com",
        api_key=os.getenv("LANGSMITH_API_KEY"),
    )

# OpenAI client
openai_client = OpenAI(api_key=OPENAI_KEY)

# --- Função principal com rastreamento seguro ---


@langsmith_traceable(name="LLM Test Call")
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

    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        user=str(trace_id),
    )

    duration = time.time() - start
    usage = response.usage

    if usage is None:
        raise ValueError("OpenAI response missing usage information")

    cost = (usage.prompt_tokens / 1000) * MODEL_PRICING["gpt-4"]["input"] + (
        usage.completion_tokens / 1000
    ) * MODEL_PRICING["gpt-4"]["output"]

    cost_tracker[str(trace_id)] = {
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

    content = response.choices[0].message.content or ""
    if return_trace:
        return content, str(trace_id)
    return content
