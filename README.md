# pedido-extractor-ptbr

[![CI](https://github.com/pedrohbenetton/pedido-extractor-ptbr/actions/workflows/ci.yml/badge.svg)](https://github.com/pedrohbenetton/pedido-extractor-ptbr/actions/workflows/ci.yml)

Fine-tuning **QLoRA** de um LLM pequeno (Qwen2.5-1.5B-Instruct) para transformar mensagens livres de
pedido de delivery em PT-BR em **JSON validado por schema**, com avaliação separada em dados
sintéticos e num conjunto difícil escrito à mão.

> "2 pizzas de calabresa sem cebola, ah não, 3. pago no pix" → `{"itens":[{"nome":"pizza de calabresa","quantidade":3,"observacoes":["sem cebola"]}],"restricoes":[],"pagamento":"pix"}`

## Status

| Etapa | Estado |
|---|---|
| Spec, schema, métricas, gerador sintético | pronto (testes passando) |
| Conjunto difícil (60 casos, 8 categorias) | pronto, revisão humana pendente |
| Treino QLoRA e avaliação no Colab/Kaggle | script e notebook prontos, **ainda não executados** |
| Baseline com LLM de API (`eval_api`, com latência, tokens e custo) | pronto, **ainda não executado** |
| Resultados reais, modelo no Hugging Face | pendentes |

Nenhum número de desempenho é divulgado aqui antes de ser medido.

## Como funciona
- `SPEC.md`: requisitos e métricas (escritos antes do código).
- `schema.py`: `Pedido`/`Item` (Pydantic, `extra="forbid"`) e `parse_pedido` tolerante a texto em volta do JSON.
- `data.py`: gerador sintético determinístico; observações coerentes com cada item, plurais no texto e singular no rótulo; splits sem vazamento.
- `data/hard_test.jsonl` (fonte: `scripts/build_hard_test.py`): correções, gíria, erros de digitação, ordem confusa, negação, quantidade por extenso, itens fora do cardápio, restrição + pagamento. **Nunca entra no treino.**
- `metrics.py` / `evaluation.py`: JSON válido, exact match, F1 por item, acerto de pagamento, quebra por categoria.

## Rodar
```bash
uv sync
uv run pytest                        # testes (sem GPU)
uv run python scripts/build_hard_test.py
```
Treino e avaliação: `notebooks/train_colab.ipynb` (GPU T4).

Baseline com LLM de API (gasta dinheiro; use `--limit` para um teste barato primeiro):
```bash
uv sync --extra api
uv run python -m pedido_extractor.eval_api --model claude-haiku-5-5 --limit 100 --price-in <US$/1M> --price-out <US$/1M>
uv run python -m pedido_extractor.eval_api --model claude-opus-5-5 --price-in <US$/1M> --price-out <US$/1M>
```
Os preços são informados por quem roda (consulte a tabela oficial); o relatório inclui latência p95, tokens e custo.

## Limites conhecidos
- O treino usa dados de template: espera-se nota alta no teste sintético e queda no difícil, principalmente em correções e quantidade por extenso. O relatório mostra isso por categoria.
- Os gabaritos do conjunto difícil são decisões de convenção (ex.: dúzia = 12, "vr" = `vale_refeicao`).
