"""Gerador sintético determinístico: texto e rótulo nascem juntos (sem ruído de anotação)."""

from __future__ import annotations

import random
import re
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

# Perfil v2 (augment=True): cobre os padrões em que o v1 falhou no conjunto difícil.
# Os marcadores e as frases abaixo são propositalmente diferentes dos textos de data/hard_test.jsonl.
CORRECAO_MARCADORES = ["corrigindo:", "desculpa, quis dizer", "opa, na verdade", "ops, troca pra"]
CANCELA_VERBOS = ["esquece", "pode tirar", "retira"]
_EXTENSO = {12: "uma dúzia de", 6: "meia dúzia de", 24: "duas dúzias de", 2: "um par de"}
_NUM_EXT = {5: "cinco", 7: "sete", 8: "oito", 10: "dez"}
_FONETICA = [("ss", "s"), ("ç", "c"), ("s", "z"), ("c", "k"), ("z", "s"), ("ch", "x")]


@dataclass(frozen=True)
class Example:
    texto: str
    pedido: Pedido


def _sem_acento(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def qtd_extenso(q: int) -> str:
    """Quantidade escrita por extenso: dúzia, meia dúzia, par ou número a partir de cinco."""
    return _EXTENSO.get(q) or _NUM_EXT[q]


def _item_text(
    rng: random.Random, prato: Prato, qtd: int, obs: list[str], extenso: bool = False
) -> str:
    num = NUM_F if prato.genero == "f" else NUM_M
    if extenso:
        q = qtd_extenso(qtd)
        nome = prato.plural
    else:
        q = num[qtd] if qtd <= 4 and rng.random() < 0.4 else str(qtd)
        nome = prato.plural if qtd > 1 and rng.random() < 0.7 else prato.nome
    falas = [
        rng.choice(VARIANTES_OBS[o]) if o in VARIANTES_OBS and rng.random() < 0.35 else o
        for o in obs
    ]
    return " ".join([q, nome, *falas])


def texto_correcao(
    rng: random.Random, prato: Prato, q_errada: int, q_final: int, obs: list[str]
) -> str:
    """O cliente diz uma quantidade e corrige em seguida; só a final vale."""
    erro = _item_text(rng, prato, q_errada, [])
    certo = _item_text(rng, prato, q_final, obs)
    return f"{erro}{rng.choice([', ', '. ', ' '])}{rng.choice(CORRECAO_MARCADORES)} {certo}"


def aplicar_typos(rng: random.Random, texto: str, taxa: float = 0.12) -> str:
    """Erros de digitação em palavras de 5+ letras; números e pontuação ficam intactos."""

    def _erra(m: re.Match[str]) -> str:
        w = m.group(0)
        if rng.random() >= taxa:
            return w
        i = rng.randrange(1, len(w) - 1)
        op = rng.choice(["drop", "double", "swap", "fonetica"])
        if op == "fonetica":
            for a, b in rng.sample(_FONETICA, k=len(_FONETICA)):
                if a in w:
                    return w.replace(a, b, 1)
            op = "drop"
        if op == "drop":
            return w[:i] + w[i + 1 :]
        if op == "double":
            return w[:i] + w[i] + w[i:]
        return w[: i - 1] + w[i] + w[i - 1] + w[i + 1 :]

    return re.sub(r"[^\W\d_]{5,}", _erra, texto)


def _artigo(prato: Prato) -> str:
    return "a" if prato.genero == "f" else "o"


def _make_one(rng: random.Random, augment: bool = False) -> Example:
    escolhidos = rng.sample(_PRATOS, k=rng.choice([1, 1, 2, 2, 3, 4]))
    extras = [p for p in _PRATOS if p not in escolhidos]
    itens, partes = [], []
    for prato in escolhidos:
        qtd = rng.choice([1, 1, 1, 2, 2, 3, 4])
        extenso = augment and rng.random() < 0.12
        if extenso:
            qtd = rng.choice([2, 5, 6, 7, 8, 10, 12, 24])
        obs = rng.sample(prato.obs, k=min(len(prato.obs), rng.choice([0, 0, 1, 2])))
        itens.append(Item(nome=prato.nome, quantidade=qtd, observacoes=obs))
        if augment and not extenso and rng.random() < 0.12:
            q_errada = rng.choice([q for q in range(1, 6) if q != qtd])
            partes.append(texto_correcao(rng, prato, q_errada, qtd, obs))
        elif augment and not extenso and rng.random() < 0.06:
            errado = rng.choice(extras)
            partes.append(
                f"{_item_text(rng, prato, qtd, obs)}, no lugar d{_artigo(errado)} {errado.nome}"
            )
        else:
            partes.append(_item_text(rng, prato, qtd, obs, extenso))
    cancelado = rng.choice(extras) if augment and rng.random() < 0.08 else None
    if cancelado:  # o item é pedido e depois retirado: não entra no rótulo
        partes.append(_item_text(rng, cancelado, rng.choice([1, 1, 2]), []))

    texto = rng.choice(ABERTURAS) + partes[0]
    for p in partes[1:]:
        texto += rng.choice(LIGACOES) + p
    if cancelado:
        texto += f"{rng.choice([', ', '. '])}{rng.choice(CANCELA_VERBOS)} {_artigo(cancelado)} {cancelado.nome}"

    restricoes: list[str] = []
    if rng.random() < 0.2:
        canon, frases = rng.choice(RESTRICOES)
        restricoes.append(canon)
        texto += rng.choice([". ", ", "]) + rng.choice(frases)

    pagamento = None
    if rng.random() < 0.6:
        pagamento, frases = rng.choice(PAGAMENTOS)
        texto += rng.choice([". ", ", "]) + rng.choice(frases)

    if augment and rng.random() < 0.35:
        texto = aplicar_typos(rng, texto)
    # ruído de digitação: o rótulo mantém a forma canônica, o texto pode perder acento/caixa
    if rng.random() < 0.3:
        texto = _sem_acento(texto)
    if rng.random() < 0.4:
        texto = texto.lower()
    return Example(texto=texto, pedido=Pedido(itens=itens, restricoes=restricoes, pagamento=pagamento))


def generate_dataset(n: int, seed: int = 0, augment: bool = False) -> list[Example]:
    rng = random.Random(seed)
    vistos: set[str] = set()
    out: list[Example] = []
    while len(out) < n:
        ex = _make_one(rng, augment)
        if ex.texto not in vistos:
            vistos.add(ex.texto)
            out.append(ex)
    return out


def build_splits(
    n: int = 6000,
    seed: int = 0,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    augment: bool = False,
) -> dict[str, list[Example]]:
    ds = generate_dataset(n, seed, augment)
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
