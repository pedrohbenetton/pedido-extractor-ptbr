"""Avaliação independente de modelo: recebe qualquer `generate_fn(lotes_de_mensagens) -> textos`."""

from __future__ import annotations

from collections.abc import Callable

from .data import Example, to_chat
from .hard import HardExample, evaluate_by_tag
from .metrics import evaluate

Messages = list[dict[str, str]]
GenerateFn = Callable[[list[Messages]], list[str]]


def _generate(gen: GenerateFn, prompts: list[Messages], batch_size: int) -> list[str]:
    out: list[str] = []
    for i in range(0, len(prompts), batch_size):
        lote = prompts[i : i + batch_size]
        resp = gen(lote)
        if len(resp) != len(lote):
            raise ValueError("generate_fn devolveu quantidade diferente de respostas")
        out.extend(resp)
    return out


def run_eval(
    gen: GenerateFn, teste: list[Example], hard: list[HardExample], batch_size: int = 16
) -> dict:
    sint = _generate(gen, [to_chat(e, with_answer=False) for e in teste], batch_size)
    dificil = _generate(
        gen, [to_chat(Example(h.texto, h.pedido), with_answer=False) for h in hard], batch_size
    )
    return {
        "synthetic": evaluate(sint, [e.pedido for e in teste]),
        "hard": evaluate_by_tag(dificil, hard),
        "n": {"synthetic": len(teste), "hard": len(hard)},
    }


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_report(results: dict[str, dict]) -> str:
    linhas = [
        "| Modelo | Sintético: JSON válido | Sintético: exact match | Difícil: JSON válido | Difícil: exact match | Difícil: F1 itens |",
        "|---|---|---|---|---|---|",
    ]
    for nome, r in results.items():
        s, h = r["synthetic"], r["hard"]["all"]
        linhas.append(
            f"| {nome} | {_pct(s['valid_json_rate'])} | {_pct(s['exact_match'])} | "
            f"{_pct(h['valid_json_rate'])} | {_pct(h['exact_match'])} | {_pct(h['item_f1'])} |"
        )
    tags = sorted({t for r in results.values() for t in r["hard"] if t != "all"})
    linhas += ["", "Exact match no conjunto difícil, por categoria:", ""]
    linhas += ["| Categoria | " + " | ".join(results) + " |", "|---|" + "---|" * len(results)]
    for t in tags:
        linhas.append(
            f"| {t} | " + " | ".join(_pct(r["hard"][t]["exact_match"]) for r in results.values()) + " |"
        )
    return "\n".join(linhas)
