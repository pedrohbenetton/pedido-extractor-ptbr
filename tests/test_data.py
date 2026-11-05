import json

from pedido_extractor.data import (
    CARDAPIO,
    SYSTEM_PROMPT,
    VARIANTES_OBS,
    build_splits,
    generate_dataset,
    to_chat,
)
from pedido_extractor.metrics import normalize
from pedido_extractor.schema import parse_pedido


def test_deterministico_por_seed():
    assert generate_dataset(50, seed=1) == generate_dataset(50, seed=1)
    assert generate_dataset(50, seed=1) != generate_dataset(50, seed=2)


def test_tamanho_e_textos_unicos():
    ds = generate_dataset(300, seed=0)
    assert len(ds) == 300 and len({e.texto for e in ds}) == 300


def test_rotulo_consistente_com_texto():
    for e in generate_dataset(300, seed=3):
        texto = normalize(e.texto)
        for item in e.pedido.itens:
            prato = CARDAPIO[item.nome]
            formas = {normalize(prato.nome), normalize(prato.plural)}
            assert any(f in texto for f in formas), (e.texto, item.nome)
            for obs in item.observacoes:
                superficies = [obs, *VARIANTES_OBS.get(obs, [])]
                assert any(normalize(s) in texto for s in superficies), (e.texto, obs)


def test_observacoes_fazem_sentido_para_o_item():
    for e in generate_dataset(500, seed=4):
        for item in e.pedido.itens:
            assert set(item.observacoes) <= set(CARDAPIO[item.nome].obs), item


def test_plural_no_texto_e_singular_no_rotulo():
    ds = generate_dataset(500, seed=6)
    com_plural = [
        e for e in ds
        if any(i.quantidade > 1 and normalize(CARDAPIO[i.nome].plural) in normalize(e.texto)
               and normalize(CARDAPIO[i.nome].plural) != normalize(i.nome) for i in e.pedido.itens)
    ]
    assert len(com_plural) > 50
    assert all(i.nome in CARDAPIO for e in ds for i in e.pedido.itens)


def test_splits_sem_vazamento():
    s = build_splits(n=400, seed=0, val_frac=0.1, test_frac=0.2)
    tr, va, te = ({e.texto for e in s[k]} for k in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(s["train"]) + len(s["val"]) + len(s["test"]) == 400
    assert len(s["test"]) == 80 and len(s["val"]) == 40


def test_chat_roundtrip():
    e = generate_dataset(1, seed=5)[0]
    msgs = to_chat(e)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == SYSTEM_PROMPT and msgs[1]["content"] == e.texto
    assert parse_pedido(msgs[2]["content"]) == e.pedido
    json.loads(msgs[2]["content"])


def test_chat_sem_resposta_para_inferencia():
    e = generate_dataset(1, seed=5)[0]
    assert [m["role"] for m in to_chat(e, with_answer=False)] == ["system", "user"]
