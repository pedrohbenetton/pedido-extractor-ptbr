# pedido-extractor-ptbr

[![CI](https://github.com/pedrohbenetton/pedido-extractor-ptbr/actions/workflows/ci.yml/badge.svg)](https://github.com/pedrohbenetton/pedido-extractor-ptbr/actions/workflows/ci.yml) [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-adapter%20v2-yellow)](https://huggingface.co/pedrohbenetton/pedido-extractor-ptbr-lora)

Fine-tuning **QLoRA** de um LLM pequeno (Qwen2.5-1.5B-Instruct) para transformar mensagens livres de
pedido de delivery em PT-BR em **JSON validado por schema**, com avaliação separada em dados
sintéticos e num conjunto difícil escrito à mão.

> "2 pizzas de calabresa sem cebola, ah não, 3. pago no pix" → `{"itens":[{"nome":"pizza de calabresa","quantidade":3,"observacoes":["sem cebola"]}],"restricoes":[],"pagamento":"pix"}`


> **Avaliação complementar:** os adapters v1 e v2 foram avaliados também no [ptbr-llm-bench](https://github.com/pedrohbenetton/ptbr-llm-bench) (intenção, ambiguidade e extração). O v2 confirma 90,0% de exact match no conjunto difícil, contra 75,0% do Haiku 5.5 zero-shot nessa rodada, e preserva mais as outras tarefas que o v1 (uma execução, amostras pequenas).

## Resultados

Qwen2.5-1.5B-Instruct em 4 bits, QLoRA (r=16, 2 épocas, 6.000 exemplos sintéticos), uma execução, T4. Teste sintético com 600 exemplos; teste difícil com 60 casos escritos à mão, nunca usados no treino.

| Modelo | Sintético: JSON válido | Sintético: exact match | Difícil: JSON válido | Difícil: exact match | Difícil: F1 de itens |
|---|---|---|---|---|---|
| base (sem fine-tuning) | 48,8% | 1,8% | 68,3% | 3,3% | 12,8% |
| + LoRA v1 | 99,7% | 99,7% | 100% | 71,7% | 80,5% |
| + LoRA v2 (dados com correções, typos, extenso) | 100% | 99,8% | 100% | **90,0%** | 93,4% |
| Claude Haiku 5.5 via API (zero-shot, esforço baixo; sintético só n=20) | 100% | 40% | 100% | 68,3% | 84,2% |

Exact match no conjunto difícil, por categoria:

| Categoria | base | LoRA v1 | LoRA v2 | Haiku 5.5 (API) |
|---|---|---|---|---|
| correcao | 0% | 25% | 75% | 100% |
| typo | 0% | 62,5% | 100% | 37,5% |
| quantidade_extenso | 0% | 62,5% | 100% | 100% |
| negacao | 0% | 87,5% | 100% | 50% |
| ordem_confusa | 12,5% | 75% | 87,5% | 50% |
| giria | 12,5% | 87,5% | 87,5% | 62,5% |
| restricao_pagamento | 0% | 83,3% | 83,3% | 66,7% |
| fora_do_cardapio | 0% | 100% | 83,3% | 83,3% |

Leitura honesta:
- **Teste sintético:** vem do mesmo gerador do treino, então perto de 100% mostra que o formato foi aprendido, não que o problema está resolvido. O número que importa é o do conjunto difícil.
- **v1:** a fraqueza principal foi a **correção no meio da frase** ("2 coxinhas, ah não, 3"), que o gerador não cobria; typo e quantidade por extenso (dúzia, par) também ficaram abaixo.
- **v2:** só os dados mudaram (mesmo modelo, hiperparâmetros e teste). O gerador passou a cobrir correções, itens cancelados ou trocados, typos e quantidade por extenso, com marcadores e frases diferentes dos do teste difícil. O exact match no difícil foi de 71,7% para 90,0%.
- **Uma categoria piorou:** `fora_do_cardapio` foi de 100% para 83,3%, ou seja, um caso a menos entre seis.
- **Ressalva:** o v2 ensina de propósito conceitos que o teste cobra, então parte do ganho em correção, typo e extenso é o que foi ensinado, não generalização livre.
- **Haiku 5.5 (API):** rodado zero-shot (só o prompt de sistema, sem exemplos), esforço baixo, nos 60 casos difíceis e em 20 exemplos sintéticos. Em 80 chamadas, 0 erros, latência média de cerca de 1,0 s (p95 1,8 s). Ele erra sobretudo em convenções do rótulo (nome no singular, observação canônica "sem X"), que o fine-tuning ensina e o prompt sozinho não, então a comparação mede em parte aderência à convenção. Uma versão com exemplos no prompt seria mais justa e ainda não foi feita. O custo em dólares não foi calculado porque os preços não foram informados.
- **Amostra pequena:** são 6 a 8 casos por categoria, então cada caso vale cerca de 1,7 ponto no total e 12 a 17 pontos numa categoria. É uma única execução com uma única seed: trate as diferenças pequenas como ruído.

## Status

| Etapa | Estado |
|---|---|
| Spec, schema, métricas, gerador sintético | pronto (testes passando) |
| Conjunto difícil (60 casos, 8 categorias) | pronto, revisão humana pendente |
| Treino QLoRA e avaliação (base vs LoRA) | **executado** no Colab (T4) |
| Baseline com LLM de API (`eval_api`, com latência, tokens e custo) | pronto, **ainda não executado** |
| Segunda versão com correções/typos/quantidades no gerador | **executado** (exact match difícil 71,7% → 90,0%) |
| Adapter no Hugging Face | publicado (v2): https://huggingface.co/pedrohbenetton/pedido-extractor-ptbr-lora |

## Como funciona
- `SPEC.md`: requisitos e métricas (escritos antes do código).
- `schema.py`: `Pedido`/`Item` (Pydantic, `extra="forbid"`) e `parse_pedido` tolerante a texto em volta do JSON.
- `data.py`: gerador sintético determinístico; observações coerentes com cada item, plurais no texto e singular no rótulo; splits sem vazamento.
- `data.py --augment` (perfil v2): correção de quantidade no meio da frase, item pedido e depois cancelado ou trocado, typos (letra trocada, omitida, dobrada, erro fonético) e quantidade por extenso (dúzia, meia dúzia, par, cinco a dez). Os marcadores e frases são diferentes dos do conjunto difícil, e um teste garante que nenhum texto do conjunto difícil é gerado.
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
- Na v2 as categorias `correcao`, `typo` e `quantidade_extenso` deixam de ser totalmente "fora da distribuição": o gerador ensina esses conceitos. Uma melhora ali mede em parte o que foi ensinado de propósito, não generalização livre.
- O treino usa dados de template: a queda do sintético (99,7%) para o difícil (71,7%) mostra a distância entre dado gerado e linguagem real, e se concentra em correções, typos e quantidade por extenso.
- Os gabaritos do conjunto difícil foram escritos por mim com ajuda de IA e ainda sem revisão externa. Alguns casos são ambíguos (observações compostas, "bem gelado"). São decisões de convenção (ex.: dúzia = 12, "vr" = `vale_refeicao`).
