import json

import pytest

from pedido_extractor.constrained import pedido_json_schema
from pedido_extractor.eval_hf import build_parser

VALIDO = {
    "itens": [{"nome": "pizza de calabresa", "quantidade": 2, "observacoes": ["sem cebola"]}],
    "restricoes": [],
    "pagamento": "pix",
}


def _compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def test_schema_fechado_e_com_enum_de_pagamento():
    s = pedido_json_schema()
    assert s["additionalProperties"] is False
    assert "itens" in s["required"]
    assert "pix" in json.dumps(s["properties"]["pagamento"])


def test_cli_constrained_flag():
    assert build_parser().parse_args(["--name", "x"]).constrained is False
    assert build_parser().parse_args(["--name", "x", "--constrained"]).constrained is True


@pytest.fixture(scope="module")
def accepts():
    xgr = pytest.importorskip("xgrammar")
    from xgrammar.testing import _is_grammar_accept_string

    from pedido_extractor.constrained import pedido_grammar

    g = pedido_grammar(xgr)
    return lambda s: _is_grammar_accept_string(g, s)


def test_gramatica_aceita_pedido_valido(accepts):
    assert accepts(_compact(VALIDO))
    assert accepts(_compact({**VALIDO, "pagamento": None}))


@pytest.mark.parametrize(
    "ruim",
    [
        {**VALIDO, "total": 10},
        {**VALIDO, "pagamento": "cartao"},
        {**VALIDO, "itens": []},
        {**VALIDO, "itens": [{"nome": "x", "quantidade": 0}]},
    ],
)
def test_gramatica_rejeita_fora_do_schema(accepts, ruim):
    assert not accepts(_compact(ruim))


def test_gramatica_rejeita_texto_fora_do_json(accepts):
    assert not accepts("Aqui está o pedido: " + _compact(VALIDO))
