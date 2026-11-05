"""Exporta os splits sintéticos em JSONL de chat (um objeto {"messages": [...]} por linha)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import build_splits, to_chat


def export_splits(
    out_dir: Path | str,
    n: int = 6000,
    seed: int = 0,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    augment: bool = False,
) -> dict[str, int]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = build_splits(n, seed, val_frac, test_frac, augment)
    for nome, exemplos in splits.items():
        with open(out / f"{nome}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps({"messages": to_chat(e)}, ensure_ascii=False) + "\n" for e in exemplos)
    return {nome: len(v) for nome, v in splits.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", action="store_true", help="perfil v2: correções, typos, extenso")
    a = ap.parse_args()
    print(export_splits(a.out, a.n, a.seed, augment=a.augment))


if __name__ == "__main__":
    main()
