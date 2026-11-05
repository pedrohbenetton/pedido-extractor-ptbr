"""CLI: avalia um LLM de API no mesmo teste dos modelos locais. Requer ANTHROPIC_API_KEY
(ou `ant auth login`) e `pip install -e '.[api]'`. Cada execução gasta dinheiro real."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import build_splits
from .evaluation import render_report, run_eval
from .hard import load_hard_test


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--name", default=None, help="rótulo no relatório (padrão: o id do modelo)")
    ap.add_argument("--effort", default="low", help="low|medium|high|xhigh|max, ou 'none' para omitir")
    ap.add_argument("--limit", type=int, default=None, help="usa só os N primeiros exemplos sintéticos")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--price-in", type=float, default=None, help="US$ por 1M de tokens de entrada")
    ap.add_argument("--price-out", type=float, default=None, help="US$ por 1M de tokens de saída")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)

    import anthropic

    from .api_runtime import make_anthropic_generate_fn

    gen, stats = make_anthropic_generate_fn(
        anthropic.Anthropic(),
        a.model,
        effort=None if a.effort == "none" else a.effort,
        workers=a.workers,
    )
    teste = build_splits(a.n, a.seed)["test"]
    if a.limit:
        teste = teste[: a.limit]
    r = run_eval(gen, teste, load_hard_test(), batch_size=a.workers * 4)
    r["api"] = {**stats.summary(), "cost_usd": stats.cost(a.price_in, a.price_out), "model": a.model}

    name = a.name or a.model
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print(render_report({name: r}))
    print("\nAPI:", json.dumps(r["api"], ensure_ascii=False))


if __name__ == "__main__":
    main()
