import json

from pedido_extractor.export import export_splits
from pedido_extractor.schema import parse_pedido


def test_export_escreve_jsonl_em_formato_de_chat(tmp_path):
    counts = export_splits(tmp_path, n=200, seed=0, val_frac=0.1, test_frac=0.1)
    assert counts == {"train": 160, "val": 20, "test": 20}
    for nome, n in counts.items():
        linhas = (tmp_path / f"{nome}.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(linhas) == n
    row = json.loads((tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
    assert parse_pedido(row["messages"][2]["content"]) is not None


def test_export_e_deterministico(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    export_splits(a, n=100, seed=7)
    export_splits(b, n=100, seed=7)
    assert (a / "train.jsonl").read_bytes() == (b / "train.jsonl").read_bytes()
