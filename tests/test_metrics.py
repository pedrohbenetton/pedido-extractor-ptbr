import pytest

from pedido_extractor.metrics import evaluate, normalize
from pedido_extractor.schema import Item, Pedido


def pedido(*itens, pagamento=None, restricoes=()):
    return Pedido(itens=list(itens), pagamento=pagamento, restricoes=list(restricoes))


def js(p: Pedido) -> str:
    return p.model_dump_json()


A = pedido(Item(nome="Pizza de Calabresa", quantidade=2, observacoes=["sem cebola"]), pagamento="pix")
B = pedido(Item(nome="coxinha", quantidade=3))


def test_normalize_caixa_e_acento():
    assert normalize("  Açaí  500ML ") == "acai 500ml"


def test_perfeito():
    r = evaluate([js(A), js(B)], [A, B])
    assert r["valid_json_rate"] == r["exact_match"] == r["item_f1"] == 1.0
    assert r["pagamento_acc"] == 1.0


def test_ordem_e_acento_nao_importam():
    pred = pedido(
        Item(nome="coxinha", quantidade=1),
        Item(nome="pizza de calabresa", quantidade=2, observacoes=["SEM CEBOLA"]),
        pagamento="pix",
    )
    gold = pedido(
        Item(nome="Pizza de Calabresa", quantidade=2, observacoes=["sem cebola"]),
        Item(nome="Coxinha", quantidade=1),
        pagamento="pix",
    )
    assert evaluate([js(pred)], [gold])["exact_match"] == 1.0


def test_invalido_conta_como_vazio():
    r = evaluate(["lixo", None], [A, B])
    assert r["valid_json_rate"] == 0.0 and r["item_f1"] == 0.0 and r["exact_match"] == 0.0


def test_f1_parcial():
    gold = pedido(Item(nome="coxinha", quantidade=1), Item(nome="pastel", quantidade=1))
    pred = pedido(Item(nome="coxinha", quantidade=1), Item(nome="esfiha", quantidade=1))
    r = evaluate([js(pred)], [gold])
    assert r["item_precision"] == r["item_recall"] == pytest.approx(0.5)
    assert r["item_f1"] == pytest.approx(0.5) and r["exact_match"] == 0.0


def test_quantidade_errada_nao_casa():
    pred = pedido(Item(nome="coxinha", quantidade=2))
    gold = pedido(Item(nome="coxinha", quantidade=1))
    assert evaluate([js(pred)], [gold])["item_f1"] == 0.0


def test_tamanhos_diferentes():
    with pytest.raises(ValueError):
        evaluate([js(A)], [A, B])
