"""Gerador sintético determinístico: texto e rótulo nascem juntos (sem ruído de anotação)."""

from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass

from .schema import Item, Pedido

SYSTEM_PROMPT = (
    "Você extrai pedidos de delivery em português do Brasil. Responda somente com um JSON "
    'no formato {"itens": [{"nome": str, "quantidade": int, "observacoes": [str]}], '
    '"restricoes": [str], "pagamento": "pix"|"dinheiro"|"credito"|"debito"|"vale_refeicao"|null}.'
)

@dataclass(frozen=True)
class Prato:
    nome: str  # forma singular: é o valor do rótulo
    plural: str
    genero: str
    obs: tuple[str, ...]


_PRATOS = [
    Prato("pizza de calabresa", "pizzas de calabresa", "f", ("sem cebola", "com borda recheada", "com queijo extra")),
    Prato("pizza margherita", "pizzas margherita", "f", ("com borda recheada", "com queijo extra")),
    Prato("pizza de frango com catupiry", "pizzas de frango com catupiry", "f", ("com borda recheada", "sem cebola")),
    Prato("hambúrguer artesanal", "hambúrgueres artesanais", "m", ("sem cebola", "sem tomate", "bem passado", "ao ponto", "com molho extra", "com queijo extra")),
    Prato("x-bacon", "x-bacons", "m", ("sem cebola", "sem tomate", "bem passado", "ao ponto", "com molho extra")),
    Prato("x-salada", "x-saladas", "m", ("sem cebola", "sem tomate", "ao ponto", "com molho extra")),
    Prato("açaí 500ml", "açaís 500ml", "m", ("sem granola", "com leite condensado")),
    Prato("coxinha", "coxinhas", "f", ("sem pimenta", "com molho extra")),
    Prato("esfiha de carne", "esfihas de carne", "f", ("sem cebola", "sem pimenta")),
    Prato("pastel de queijo", "pastéis de queijo", "m", ("sem pimenta", "com molho extra")),
    Prato("combo sushi 20 peças", "combos sushi 20 peças", "m", ("sem wasabi", "com molho extra")),
    Prato("marmita de frango", "marmitas de frango", "f", ("sem pimenta", "sem cebola")),
    Prato("marmita de carne", "marmitas de carne", "f", ("sem pimenta", "sem cebola")),
    Prato("refrigerante lata", "refrigerantes lata", "m", ("sem gelo", "bem gelado")),
    Prato("suco de laranja", "sucos de laranja", "m", ("sem gelo", "sem açúcar", "bem gelado")),
    Prato("batata frita", "batatas fritas", "f", ("sem sal", "com cheddar e bacon")),
    Prato("lasanha", "lasanhas", "f", ("com queijo extra", "sem pimenta")),
    Prato("sorvete de chocolate", "sorvetes de chocolate", "m", ("com calda extra", "sem cobertura")),
    Prato("água sem gás", "águas sem gás", "f", ("sem gelo", "com limão")),
    Prato("milkshake de morango", "milkshakes de morango", "m", ("sem chantilly", "com chantilly extra")),
]
CARDAPIO = {p.nome: p for p in _PRATOS}

# Formas de falar a mesma observação; o rótulo é sempre a forma canônica (chave).
VARIANTES_OBS = {
    "sem cebola": ["tira a cebola", "sem a cebola"],
    "sem gelo": ["tira o gelo", "sem o gelo"],
    "sem pimenta": ["tira a pimenta", "sem a pimenta"],
    "sem tomate": ["tira o tomate", "sem o tomate"],
    "bem passado": ["bem passadinho"],
    "ao ponto": ["no ponto"],
}
RESTRICOES = [
    ("alergia a amendoim", ["sou alérgico a amendoim", "tenho alergia a amendoim"]),
    ("sem glúten", ["não posso comer glúten", "preciso de opção sem glúten"]),
    ("sem lactose", ["sou intolerante a lactose", "não pode ter lactose"]),
    ("vegetariano", ["sou vegetariano", "não como carne"]),
]
PAGAMENTOS = [
    ("pix", ["pago no pix", "vou pagar via pix", "pix"]),
    ("dinheiro", ["pago em dinheiro", "dinheiro na entrega", "tenho troco pra 50"]),
    ("credito", ["cartão de crédito", "passa no crédito"]),
    ("debito", ["débito na maquininha", "pago no débito"]),
    ("vale_refeicao", ["vale refeição", "pago com vale-refeição"]),
]
ABERTURAS = ["", "oi, ", "boa noite! ", "olá, queria ", "quero ", "me vê ", "gostaria de pedir "]
LIGACOES = [", ", " e ", ", mais ", " + "]
NUM_M = {1: "um", 2: "dois", 3: "três", 4: "quatro"}
NUM_F = {1: "uma", 2: "duas", 3: "três", 4: "quatro"}


@dataclass(frozen=True)
class Example:
    texto: str
    pedido: Pedido


def _sem_acento(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _item_text(rng: random.Random, prato: Prato, qtd: int, obs: list[str]) -> str:
    num = NUM_F if prato.genero == "f" else NUM_M
    q = num[qtd] if qtd <= 4 and rng.random() < 0.4 else str(qtd)
    nome = prato.plural if qtd > 1 and rng.random() < 0.7 else prato.nome
    falas = [
        rng.choice(VARIANTES_OBS[o]) if o in VARIANTES_OBS and rng.random() < 0.35 else o
        for o in obs
    ]
    return " ".join([q, nome, *falas])


def _make_one(rng: random.Random) -> Example:
    escolhidos = rng.sample(_PRATOS, k=rng.choice([1, 1, 2, 2, 3, 4]))
    itens, partes = [], []
    for prato in escolhidos:
        qtd = rng.choice([1, 1, 1, 2, 2, 3, 4])
        obs = rng.sample(prato.obs, k=min(len(prato.obs), rng.choice([0, 0, 1, 2])))
        itens.append(Item(nome=prato.nome, quantidade=qtd, observacoes=obs))
        partes.append(_item_text(rng, prato, qtd, obs))

    texto = rng.choice(ABERTURAS) + partes[0]
    for p in partes[1:]:
        texto += rng.choice(LIGACOES) + p

    restricoes: list[str] = []
    if rng.random() < 0.2:
        canon, frases = rng.choice(RESTRICOES)
        restricoes.append(canon)
        texto += rng.choice([". ", ", "]) + rng.choice(frases)

    pagamento = None
    if rng.random() < 0.6:
        pagamento, frases = rng.choice(PAGAMENTOS)
        texto += rng.choice([". ", ", "]) + rng.choice(frases)

    # ruído de digitação: o rótulo mantém a forma canônica, o texto pode perder acento/caixa
    if rng.random() < 0.3:
        texto = _sem_acento(texto)
    if rng.random() < 0.4:
        texto = texto.lower()
    return Example(texto=texto, pedido=Pedido(itens=itens, restricoes=restricoes, pagamento=pagamento))


def generate_dataset(n: int, seed: int = 0) -> list[Example]:
    rng = random.Random(seed)
    vistos: set[str] = set()
    out: list[Example] = []
    while len(out) < n:
        ex = _make_one(rng)
        if ex.texto not in vistos:
            vistos.add(ex.texto)
            out.append(ex)
    return out


def build_splits(
    n: int = 6000, seed: int = 0, val_frac: float = 0.1, test_frac: float = 0.1
) -> dict[str, list[Example]]:
    ds = generate_dataset(n, seed)
    n_test, n_val = round(n * test_frac), round(n * val_frac)
    return {
        "test": ds[:n_test],
        "val": ds[n_test : n_test + n_val],
        "train": ds[n_test + n_val :],
    }


def to_chat(ex: Example, with_answer: bool = True) -> list[dict[str, str]]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ex.texto},
    ]
    if with_answer:
        msgs.append({"role": "assistant", "content": ex.pedido.model_dump_json()})
    return msgs
