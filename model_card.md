---
language: pt
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
  - qlora
  - peft
  - structured-output
  - json
  - portuguese
---

# pedido-extractor-ptbr-lora

Adapter LoRA (QLoRA) para o [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) que transforma mensagens livres de pedido de delivery em **PT-BR** em um **JSON** com itens, quantidades, observações, restrições alimentares e forma de pagamento.

> "2 pizzas de calabresa sem cebola, ah não, 3. pago no pix" → `{"itens":[{"nome":"pizza de calabresa","quantidade":3,"observacoes":["sem cebola"]}],"restricoes":[],"pagamento":"pix"}`

Código, dados, testes e avaliação: https://github.com/pedrohbenetton/pedido-extractor-ptbr

## Como usar

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, "SEU_USUARIO/pedido-extractor-ptbr-lora")

SYSTEM = (
    "Você extrai pedidos de delivery em português do Brasil. Responda somente com um JSON "
    'no formato {"itens": [{"nome": str, "quantidade": int, "observacoes": [str]}], '
    '"restricoes": [str], "pagamento": "pix"|"dinheiro"|"credito"|"debito"|"vale_refeicao"|null}.'
)
msgs = [{"role": "system", "content": SYSTEM},
        {"role": "user", "content": "me vê 1 x-bacon sem cebola e 2 refrigerantes lata, pago no pix"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=256, do_sample=False)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

Os nomes dos itens seguem o cardápio fictício do projeto (20 pratos, nome no singular).

## Treino

- QLoRA 4 bits, r=16, alpha=32, 2 épocas, lr 2e-4, GPU T4.
- 6.000 exemplos **sintéticos** gerados por templates (perfil `--augment`: correções no meio da frase, itens cancelados ou trocados, erros de digitação e quantidade por extenso).
- Nenhum dado de pessoas reais.

## Resultados

Conjunto difícil: 60 casos escritos à mão, nunca usados no treino (8 categorias).

| Modelo | JSON válido | Exact match | F1 de itens |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct (sem fine-tuning) | 68,3% | 3,3% | 12,8% |
| + este adapter | 100% | **90,0%** | 93,4% |

No teste sintético (600 exemplos, mesmo gerador do treino): 100% de JSON válido e 99,8% de exact match, o que mostra que o formato foi aprendido, não que a tarefa está resolvida.

## Limites

- O adapter ensina conceitos (correção, typo, dúzia/par) que o conjunto difícil também cobre, então parte do ganho nessas categorias é o que foi ensinado de propósito.
- Só 60 casos de teste (6 a 8 por categoria), uma execução e uma seed: diferenças pequenas são ruído.
- Os gabaritos foram escritos com ajuda de IA e não têm revisão externa.
- Cardápio fechado e dados sintéticos: **não use em produção** sem avaliar em mensagens reais.
