# pedido-extractor-ptbr

[![CI](https://github.com/pedrohbenetton/pedido-extractor-ptbr/actions/workflows/ci.yml/badge.svg)](https://github.com/pedrohbenetton/pedido-extractor-ptbr/actions/workflows/ci.yml)

Fine-tuning **QLoRA** de um LLM pequeno (Qwen2.5-1.5B-Instruct) para transformar mensagens livres de
pedido de delivery em PT-BR em **JSON validado por schema**, com avaliação separada em dados
sintéticos e num conjunto difícil escrito à mão.

> "2 pizzas de calabresa sem cebola, ah não, 3. pago no pix" → `{"itens":[{"nome":"pizza de calabresa","quantidade":3,"observacoes":["sem cebola"]}],"restricoes":[],"pagamento":"pix"}`

## Resultados

Qwen2.5-1.5B-Instruct em 4 bits, QLoRA (r=16, 2 épocas, 6.000 exemplos sintéticos), uma execução, T4. Teste sintético com 600 exemplos; teste difícil com 60 casos escritos à mão, nunca usados no treino.

| Modelo | Sintético: JSON válido | Sintético: exact match | Difícil: JSON válido | Difícil: exact match | Difícil: F1 de itens |
|---|---|---|---|---|---|
| base (sem fine-tuning) | 48,8% | 1,8% | 68,3% | 3,3% | 12,8% |
| + LoRA | 99,7% | 99,7% | 100% | **71,7%** | 80,5% |

Exact match no conjunto difícil, por categoria:

| Categoria | base | LoRA |
|---|---|---|
| fora_do_cardapio | 0% | 100% |
| giria | 12,5% | 87,5% |
| negacao | 0% | 87,5% |
| restricao_pagamento | 0% | 83,3% |
| ordem_confusa | 12,5% | 75% |
| typo | 0% | 62,5% |
| quantidade_extenso | 0% | 62,5% |
| correcao | 0% | **25%** |

Leitura honesta: o teste sintético vem do mesmo gerador do treino, então 99,7% mostra que o formato foi aprendido, não que o problema está resolvido. O número que importa é o do conjunto difícil. A fraqueza principal é a **correção no meio da frase** ("2 coxinhas, ah não, 3"), que o gerador sintético não cobre. Com 60 casos (6 a 8 por categoria), cada caso vale cerca de 1,7 ponto no total e 12 a 17 pontos numa categoria, e é uma única execução com uma única seed: trate as diferenças pequenas como ruído.

## Status

| Etapa | Estado |
|---|---|
| Spec, schema, métricas, gerador sintético | pronto (testes passando) |
| Conjunto difícil (60 casos, 8 categorias) | pronto, revisão humana pendente |
| Treino QLoRA e avaliação (base vs LoRA) | **executado** no Colab (T4) |
| Baseline com LLM de API (`eval_api`, com latência, tokens e custo) | pronto, **ainda não executado** |
| Segunda versão com correções/typos/quantidades no gerador | planejada |
| Adapter no Hugging Face | pendente |

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
- O treino usa dados de template: a queda do sintético (99,7%) para o difícil (71,7%) mostra a distância entre dado gerado e linguagem real, e se concentra em correções, typos e quantidade por extenso.
- Os gabaritos do conjunto difícil foram escritos por mim com ajuda de IA e ainda sem revisão externa. Alguns casos são ambíguos (observações compostas, "bem gelado"). São decisões de convenção (ex.: dúzia = 12, "vr" = `vale_refeicao`).
