"""CLI: avalia um modelo HF (base ou com adapter LoRA) e grava results/<nome>.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import build_splits
from .evaluation import render_report, run_eval
from .hard import load_hard_test


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="caminho ou repo HF do adapter LoRA")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--n", type=int, default=6000, help="mesmo n/seed do treino, para o teste ser o mesmo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-4bit", action="store_true")
    a = ap.parse_args(argv)

    from .hf_runtime import load, make_generate_fn

    m, tok = load(a.model, a.adapter, load_in_4bit=not a.no_4bit)
    teste = build_splits(a.n, a.seed)["test"]
    r = run_eval(make_generate_fn(m, tok), teste, load_hard_test(), batch_size=a.batch_size)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.name}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print(render_report({a.name: r}))


if __name__ == "__main__":
    main()
