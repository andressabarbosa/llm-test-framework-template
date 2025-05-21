install:
	python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

test:
	pytest tests/

dashboard:
	python dashboard/metrics_summary.py

run-agent:
	python -c 'from app.llm_chain import run_agent_with_tools; print(run_agent_with_tools("Qual a capital da França?"))'