from collections import Counter

from pedido_extractor.data import build_splits
from pedido_extractor.hard import evaluate_by_tag, load_hard_test
from pedido_extractor.schema import Item, Pedido


def test_conjunto_dificil_carrega_e_e_valido():
    hard = load_hard_test()
    assert len(hard) >= 60
    assert len({h.id for h in hard}) == len(hard)
    assert len({h.texto for h in hard}) == len(hard)


def test_cobre_varias_categorias_de_dificuldade():
    tags = Counter(h.tag for h in load_hard_test())
    assert len(tags) >= 8
    assert all(n >= 5 for n in tags.values()), tags


def test_sem_vazamento_com_dados_de_treino():
    sintetico = {e.texto for ex in build_splits(6000, 0).values() for e in ex}
    assert not ({h.texto for h in load_hard_test()} & sintetico)


def test_avaliacao_por_tag_perfeita():
    hard = load_hard_test()
    r = evaluate_by_tag([h.pedido.model_dump_json() for h in hard], hard)
    assert set(r) == {h.tag for h in hard} | {"all"}
    assert all(v["exact_match"] == 1.0 for v in r.values())


def test_avaliacao_por_tag_detecta_erro():
    hard = load_hard_test()
    ruim = Pedido(itens=[Item(nome="nada", quantidade=1)]).model_dump_json()
    preds = [h.pedido.model_dump_json() for h in hard]
    preds[0] = ruim
    r = evaluate_by_tag(preds, hard)
    assert r[hard[0].tag]["exact_match"] < 1.0 and r["all"]["exact_match"] < 1.0
