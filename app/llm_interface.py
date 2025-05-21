from uuid import uuid4
import openai
import time
import promptlayer
import wandb
from langsmith import traceable, Client

cost_tracker = {}

MODEL_PRICING = {
    "gpt-4": {"input": 0.03, "output": 0.06},
}

promptlayer.api_key = "YOUR_PROMPTLAYER_API_KEY"
wandb.init(project="llm-test-framework", name="test_run", mode="offline")
client = Client(api_url="https://api.smith.langchain.com", api_key="YOUR_LANGSMITH_API_KEY")

@traceable(name="LLM Test Call")
@promptlayer.track
def get_response(prompt, metadata=None):
    trace_id = metadata.get("trace_id") if metadata else str(uuid4())
    start = time.time()
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        user=trace_id
    )
    duration = time.time() - start

    usage = response.usage
    cost = (usage.prompt_tokens / 1000) * MODEL_PRICING["gpt-4"]["input"] +            (usage.completion_tokens / 1000) * MODEL_PRICING["gpt-4"]["output"]
    cost_tracker[trace_id] = {"duration": duration, "cost": round(cost, 4)}

    wandb.log({
        "prompt": prompt,
        "trace_id": trace_id,
        "duration": duration,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "cost": cost
    })

    return response.choices[0].message["content"]