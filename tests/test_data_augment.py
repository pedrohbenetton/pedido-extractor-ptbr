import random

from pedido_extractor.data import (
    CARDAPIO,
    aplicar_typos,
    generate_dataset,
    qtd_extenso,
    texto_correcao,
)
from pedido_extractor.hard import load_hard_test
from pedido_extractor.metrics import normalize

N = 1500


def test_default_continua_identico_ao_v1():
    textos = [e.texto for e in generate_dataset(3, seed=1)]
    assert textos[0] == "ola, queria quatro aguas sem gas. dinheiro na entrega"
    assert generate_dataset(200, seed=1) == generate_dataset(200, seed=1, augment=False)


def test_augment_deterministico_e_diferente_do_v1():
    a = generate_dataset(100, seed=2, augment=True)
    assert a == generate_dataset(100, seed=2, augment=True)
    assert a != generate_dataset(100, seed=2)


def test_rotulos_validos_com_augment():
    for e in generate_dataset(600, seed=3, augment=True):
        for i in e.pedido.itens:
            assert i.nome in CARDAPIO and i.quantidade >= 1
            assert set(i.observacoes) <= set(CARDAPIO[i.nome].obs)


def test_qtd_extenso_conceitos():
    assert qtd_extenso(12) == "uma dúzia de"
    assert qtd_extenso(6) == "meia dúzia de"
    assert qtd_extenso(24) == "duas dúzias de"
    assert qtd_extenso(2) == "um par de"
    assert qtd_extenso(5) == "cinco"
    assert qtd_extenso(7) == "sete"


def test_texto_correcao_termina_na_quantidade_final():
    r = random.Random(1)
    prato = CARDAPIO["coxinha"]
    t = texto_correcao(r, prato, 2, 3, [])
    assert any(m in t for m in ("corrigindo", "quis dizer", "na verdade", "ops, troca"))
    depois = t.split("coxinha")[-2]  # trecho entre a última e a penúltima ocorrência
    assert ("3" in depois) or ("três" in depois)


def test_typos_deterministicos_preservam_numeros():
    t = "quero 3 pizzas de calabresa sem cebola, pago no pix"
    a = aplicar_typos(random.Random(0), t, taxa=0.5)
    assert a == aplicar_typos(random.Random(0), t, taxa=0.5)
    assert a != t
    assert "3" in a
    assert aplicar_typos(random.Random(0), t, taxa=0.0) == t


def test_augment_gera_correcoes_cancelamentos_typos_e_extenso():
    ds = generate_dataset(N, seed=4, augment=True)
    textos = [normalize(e.texto) for e in ds]
    assert sum(any(m in t for m in ("corrigindo", "quis dizer", "na verdade", "ops, troca")) for t in textos) > 60
    assert sum("duzia" in t or "um par de" in t for t in textos) > 30
    # prato citado no texto mas ausente do rótulo = item cancelado ou trocado
    fantasmas = 0
    for e, t in zip(ds, textos, strict=True):
        no_rotulo = {i.nome for i in e.pedido.itens}
        if any(p not in no_rotulo and normalize(p) in t for p in CARDAPIO):
            fantasmas += 1
    assert fantasmas > 60


def test_v1_nao_cita_prato_fora_do_rotulo():
    for e in generate_dataset(500, seed=5):
        t = normalize(e.texto)
        no_rotulo = {i.nome for i in e.pedido.itens}
        assert not any(p not in no_rotulo and normalize(p) in t for p in CARDAPIO), e.texto


def test_nenhum_texto_do_teste_dificil_vaza_para_o_treino():
    duros = {normalize(h.texto) for h in load_hard_test()}
    gerados = {normalize(e.texto) for e in generate_dataset(4000, seed=6, augment=True)}
    assert not (duros & gerados)
