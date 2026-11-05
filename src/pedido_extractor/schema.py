"""Schema do pedido e parsing tolerante da saída do modelo."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nome: str = Field(min_length=1)
    quantidade: int = Field(ge=1)
    observacoes: list[str] = Field(default_factory=list)


class Pedido(BaseModel):
    model_config = ConfigDict(extra="forbid")

    itens: list[Item] = Field(min_length=1)
    restricoes: list[str] = Field(default_factory=list)
    pagamento: Literal["pix", "dinheiro", "credito", "debito", "vale_refeicao"] | None = None


def _first_json_object(text: str) -> str | None:
    """Primeiro objeto `{...}` balanceado do texto, respeitando strings."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                esc = (c == "\\") and not esc
                if c == '"' and not esc:
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def parse_pedido(text: str | None) -> Pedido | None:
    """Extrai e valida um `Pedido`; devolve None para qualquer saída inválida."""
    if not isinstance(text, str):
        return None
    candidate = _first_json_object(text)
    if candidate is None:
        return None
    try:
        return Pedido.model_validate(json.loads(candidate))
    except (ValueError, ValidationError):
        return None
