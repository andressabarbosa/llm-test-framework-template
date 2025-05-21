# Caminho do ambiente virtual
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# --- Instalação ---

setup: $(VENV)/bin/activate

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install-deps:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements-dev.txt

# --- Testes e Cobertura ---

test:
	PYTHONPATH=. $(PYTHON) -m pytest tests/

coverage:
	PYTHONPATH=. $(PYTHON) -m pytest --cov=app --cov-report=term-missing tests/

coverage-html:
	PYTHONPATH=. $(PYTHON) -m pytest --cov=app --cov-report=html tests/

# --- Qualidade de código ---

lint:
	ruff check . && black --check .

format:
	black .

type-check:
	mypy app tests

check-all: format lint type-check test

# --- Conveniência ---

run-agent:
	PYTHONPATH=. $(PYTHON) -c "from app.llm_chain import run_agent_with_tools; print(run_agent_with_tools('Qual a capital da França?'))"

# --- Limpeza ---

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .mypy_cache .ruff_cache

# --- Marcação de alvos auxiliares ---

.PHONY: setup install-deps install-dev test coverage coverage-html lint format type-check run-agent clean check-all
