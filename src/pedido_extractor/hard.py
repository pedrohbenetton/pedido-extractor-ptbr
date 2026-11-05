"""Conjunto de teste difícil, escrito à mão (data/hard_test.jsonl); nunca entra no treino."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .metrics import evaluate
from .schema import Pedido

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "hard_test.jsonl"


@dataclass(frozen=True)
class HardExample:
    id: str
    tag: str
    texto: str
    pedido: Pedido


def load_hard_test(path: Path | str = DEFAULT_PATH) -> list[HardExample]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out.append(HardExample(d["id"], d["tag"], d["texto"], Pedido.model_validate(d["pedido"])))
    return out


def evaluate_by_tag(
    predictions: list[str | None], examples: list[HardExample]
) -> dict[str, dict[str, float]]:
    if len(predictions) != len(examples):
        raise ValueError("predictions e examples precisam ter o mesmo tamanho")
    grupos: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        grupos[ex.tag].append(i)
    result = {
        tag: evaluate([predictions[i] for i in idx], [examples[i].pedido for i in idx])
        for tag, idx in grupos.items()
    }
    result["all"] = evaluate(predictions, [e.pedido for e in examples])
    return result
