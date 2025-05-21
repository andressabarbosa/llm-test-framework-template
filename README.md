# LLM Test Framework Template

Este projeto é um template para criação de frameworks de testes para aplicações com LLMs (Large Language Models), com suporte a:

- Testes de prompts, semântica, completude, custo e performance
- Chains estruturados com LangChain
- Agentes com ferramentas externas (DuckDuckGo)
- Chains com RAG, memória e grafos
- Observabilidade com Weights & Biases, PromptLayer e LangSmith
- Dashboard simples para análise de métricas

## Como usar

```bash
make install
cp .env.example .env  # edite suas chaves
make test
make dashboard
```

## Requisitos
- Python 3.10+
- Acesso às APIs: OpenAI, LangSmith, PromptLayer