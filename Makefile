install:
	python3 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

test:
	PYTHONPATH=. .venv/bin/python -m pytest tests/

dashboard:
	. .venv/bin/activate && python dashboard/metrics_summary.py

run-agent:
	. .venv/bin/activate && python -c 'from app.llm_chain import run_agent_with_tools; print(run_agent_with_tools("Qual a capital da França?"))'
