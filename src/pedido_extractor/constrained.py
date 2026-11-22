"""Geração restrita ao schema do `Pedido` (xgrammar): o modelo só pode emitir JSON válido.

Serve para separar ganho de formato de ganho de extração no baseline. `xgrammar` é importado
pelo chamador e passado como argumento, para este módulo não depender dele.
"""

from __future__ import annotations

from typing import Any

from .schema import Pedido

# JSON compacto: sem isso o modelo pode gastar os tokens em espaços dentro da gramática
SEPARATORS = (",", ":")


def pedido_json_schema() -> dict[str, Any]:
    return Pedido.model_json_schema()


def pedido_grammar(xgr):
    return xgr.Grammar.from_json_schema(
        pedido_json_schema(), any_whitespace=False, separators=SEPARATORS
    )


def make_logits_processor_factory(xgr, tok, vocab_size: int):
    """Devolve uma função que cria um LogitsProcessor novo por chamada de `generate` (ele tem estado)."""
    info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab_size)
    compiled = xgr.GrammarCompiler(info).compile_grammar(pedido_grammar(xgr))
    from xgrammar.contrib.hf import LogitsProcessor

    return lambda: LogitsProcessor(compiled)
