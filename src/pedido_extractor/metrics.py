"""Métricas de extração: validade, exact match, F1 por item e acerto de pagamento."""

from __future__ import annotations

import unicodedata
from collections import Counter

from .schema import Pedido, parse_pedido


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().split())


def _item_keys(p: Pedido) -> Counter:
    return Counter(
        (normalize(i.nome), i.quantidade, tuple(sorted(normalize(o) for o in i.observacoes)))
        for i in p.itens
    )


def _canonical(p: Pedido) -> tuple:
    return (
        tuple(sorted(_item_keys(p).elements())),
        tuple(sorted(normalize(r) for r in p.restricoes)),
        p.pagamento,
    )


def evaluate(predictions: list[str | None], golds: list[Pedido]) -> dict[str, float]:
    if len(predictions) != len(golds):
        raise ValueError("predictions e golds precisam ter o mesmo tamanho")
    n = len(golds)
    if n == 0:
        raise ValueError("conjunto vazio")

    valid = exact = pag_ok = 0
    tp = fp = fn = 0
    for raw, gold in zip(predictions, golds, strict=True):
        pred = parse_pedido(raw)
        gk = _item_keys(gold)
        pk = _item_keys(pred) if pred else Counter()
        hit = sum((gk & pk).values())
        tp += hit
        fp += sum(pk.values()) - hit
        fn += sum(gk.values()) - hit
        if pred:
            valid += 1
            exact += _canonical(pred) == _canonical(gold)
            pag_ok += pred.pagamento == gold.pagamento

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "valid_json_rate": valid / n,
        "exact_match": exact / n,
        "item_precision": precision,
        "item_recall": recall,
        "item_f1": f1,
        "pagamento_acc": pag_ok / n,
    }
