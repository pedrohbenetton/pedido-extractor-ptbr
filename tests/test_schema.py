import pytest
from pydantic import ValidationError

from pedido_extractor.schema import Item, Pedido, parse_pedido

VALIDO = '{"itens": [{"nome": "pizza de calabresa", "quantidade": 2, "observacoes": ["sem cebola"]}], "restricoes": [], "pagamento": "pix"}'


def test_pedido_valido():
    p = Pedido.model_validate_json(VALIDO)
    assert p.itens[0].quantidade == 2 and p.pagamento == "pix"


@pytest.mark.parametrize("quantidade", [0, -1])
def test_quantidade_invalida(quantidade):
    with pytest.raises(ValidationError):
        Item(nome="coxinha", quantidade=quantidade)


def test_nome_vazio_e_itens_vazios():
    with pytest.raises(ValidationError):
        Item(nome="", quantidade=1)
    with pytest.raises(ValidationError):
        Pedido(itens=[])


def test_campo_extra_rejeitado():
    with pytest.raises(ValidationError):
        Pedido.model_validate({"itens": [{"nome": "x", "quantidade": 1}], "total": 10})


def test_pagamento_fora_da_lista():
    with pytest.raises(ValidationError):
        Pedido.model_validate({"itens": [{"nome": "x", "quantidade": 1}], "pagamento": "bitcoin"})


@pytest.mark.parametrize(
    "texto",
    [VALIDO, f"```json\n{VALIDO}\n```", f"Claro! Aqui está:\n{VALIDO}\nAlgo mais?"],
)
def test_parse_extrai_json(texto):
    assert parse_pedido(texto) == Pedido.model_validate_json(VALIDO)


@pytest.mark.parametrize("texto", ["", "oi", "{", "{}", '{"itens": []}', "[1,2]", None])
def test_parse_nunca_levanta(texto):
    assert parse_pedido(texto) is None
