# SPEC: extração de pedidos de delivery em PT-BR

**Entrada:** mensagem livre de cliente ("2 pizza de calabresa sem cebola e 1 coca, pago no pix").
**Saída:** JSON único que valida em `Pedido` (schema.py).

## Requisitos
- R1. `Pedido` rejeita campos extras, quantidade < 1, nome vazio e lista de itens vazia.
- R2. `parse_pedido` aceita JSON puro ou em bloco ```json, e devolve `None` (nunca levanta) para qualquer saída inválida.
- R3. Métricas ignoram caixa, acentos e ordem de itens/observações.
- R4. Saída inválida conta como pedido vazio no F1 (não é descartada) e reduz `valid_json_rate`.
- R5. Dataset sintético é determinístico por seed; rótulo e texto são gerados juntos (sem ruído de anotação).
- R6. Splits train/val/test não compartilham nenhum texto (sem vazamento).
- R7. O formato de chat de treino e de inferência é o mesmo; a resposta do assistente re-parseia para o gabarito.

## Métricas
`valid_json_rate`, `exact_match`, `item_precision/recall/f1` (multiconjunto de (nome, qtd, observações)), `pagamento_acc`.

## Comparação final (README)
Modelo base vs. modelo ajustado (LoRA) vs. LLM de API, mesmos prompts, conjunto de teste fixo, com custo e latência.
