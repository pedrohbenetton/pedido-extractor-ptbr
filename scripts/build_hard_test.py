"""Fonte de verdade do conjunto difícil. Escrito à mão: cada caso testa uma dificuldade que
o gerador sintético não produz. Convenções: nome no singular; observações canônicas
("sem X"/"com X"); restrições canônicas; dúzia=12, par=2. Rode: python scripts/build_hard_test.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pedido_extractor.schema import Item, Pedido

CASOS: list[tuple] = []


def H(tag, texto, itens, restricoes=(), pagamento=None):
    CASOS.append((tag, texto, itens, list(restricoes), pagamento))


# correção no meio da frase
H("correcao", "quero 2 coxinhas, ah não, 3 coxinhas", [("coxinha", 3, [])])
H("correcao", "me vê um x-bacon e uma coca... na real troca a coca por suco de laranja", [("x-bacon", 1, []), ("suco de laranja", 1, [])])
H("correcao", "2 pizzas margherita. pera, só uma mesmo. pago no pix", [("pizza margherita", 1, [])], pagamento="pix")
H("correcao", "1 lasanha e 1 batata frita, cancela a batata", [("lasanha", 1, [])])
H("correcao", "3 pastéis de queijo, melhor 4", [("pastel de queijo", 4, [])])
H("correcao", "uma marmita de frango. não, de carne", [("marmita de carne", 1, [])])
H("correcao", "quero 2 esfihas de carne e 1 refrigerante lata, errei, são 3 esfihas", [("esfiha de carne", 3, []), ("refrigerante lata", 1, [])])
H("correcao", "açaí 500ml, oops quero 2 açaís 500ml, pago dinheiro", [("açaí 500ml", 2, [])], pagamento="dinheiro")
# gíria e abreviação
H("giria", "mano manda 2 x-bacon aí, pago no pix", [("x-bacon", 2, [])], pagamento="pix")
H("giria", "e aí, bora de pizza de calabresa sem a cebola, vlw", [("pizza de calabresa", 1, ["sem cebola"])])
H("giria", "vc tem como mandar 1 hamburguer artesanal no ponto? vr", [("hambúrguer artesanal", 1, ["ao ponto"])], pagamento="vale_refeicao")
H("giria", "manda um suco de laranja sem gelo pfv, pago em dinheiro", [("suco de laranja", 1, ["sem gelo"])], pagamento="dinheiro")
H("giria", "bão, 2 coxinhas e 2 pastéis de queijo, pix", [("coxinha", 2, []), ("pastel de queijo", 2, [])], pagamento="pix")
H("giria", "preciso de 3 marmitas de frango p hj, vale-refeição", [("marmita de frango", 3, [])], pagamento="vale_refeicao")
H("giria", "tô com fome kkk 1 lasanha e 1 milkshake de morango sem chantilly", [("lasanha", 1, []), ("milkshake de morango", 1, ["sem chantilly"])])
H("giria", "me ve 1 sorvete de chocolate c/ calda extra", [("sorvete de chocolate", 1, ["com calda extra"])])
# erros de digitação
H("typo", "2 pizas de calabreza", [("pizza de calabresa", 2, [])])
H("typo", "1 hamburger artesanal sem cebolla, pago no pixx", [("hambúrguer artesanal", 1, ["sem cebola"])], pagamento="pix")
H("typo", "uma lazanha e dois refrigerante lata", [("lasanha", 1, []), ("refrigerante lata", 2, [])])
H("typo", "3 esfiha de karne", [("esfiha de carne", 3, [])])
H("typo", "2 açai 500 ml", [("açaí 500ml", 2, [])])
H("typo", "1 x bacon e 1 x salada pagamento debito", [("x-bacon", 1, []), ("x-salada", 1, [])], pagamento="debito")
H("typo", "quero batata frira 2 e suco de laranja", [("batata frita", 2, []), ("suco de laranja", 1, [])])
H("typo", "milk shake de morango x2 sem chantili", [("milkshake de morango", 2, ["sem chantilly"])])
# ordem confusa
H("ordem_confusa", "pago no pix. quero 2 coxinhas e 1 pastel de queijo", [("coxinha", 2, []), ("pastel de queijo", 1, [])], pagamento="pix")
H("ordem_confusa", "sem cebola, 2 esfihas de carne", [("esfiha de carne", 2, ["sem cebola"])])
H("ordem_confusa", "sou alérgico a amendoim, manda 1 açaí 500ml", [("açaí 500ml", 1, [])], restricoes=["alergia a amendoim"])
H("ordem_confusa", "débito. 1 lasanha. sem pimenta.", [("lasanha", 1, ["sem pimenta"])], pagamento="debito")
H("ordem_confusa", "2 x-salada, 1 x-bacon (esse sem tomate), pago em dinheiro", [("x-salada", 2, []), ("x-bacon", 1, ["sem tomate"])], pagamento="dinheiro")
H("ordem_confusa", "o pastel de queijo é pra mim, 2 deles. a água sem gás é pra minha mãe, 1", [("pastel de queijo", 2, []), ("água sem gás", 1, [])])
H("ordem_confusa", "tenho vale refeição. vou querer a marmita de carne x3", [("marmita de carne", 3, [])], pagamento="vale_refeicao")
H("ordem_confusa", "preciso de opção sem glúten, 1 marmita de frango e 1 suco de laranja", [("marmita de frango", 1, []), ("suco de laranja", 1, [])], restricoes=["sem glúten"])
# negação, distratores e variações de observação
H("negacao", "1 hambúrguer artesanal bem passado, não precisa de guardanapo", [("hambúrguer artesanal", 1, ["bem passado"])])
H("negacao", "2 pizzas de calabresa, não coloca cebola", [("pizza de calabresa", 2, ["sem cebola"])])
H("negacao", "1 suco de laranja, sem açúcar e sem gelo", [("suco de laranja", 1, ["sem açúcar", "sem gelo"])])
H("negacao", "uma batata frita sem sal com cheddar e bacon", [("batata frita", 1, ["sem sal", "com cheddar e bacon"])])
H("negacao", "3 refrigerantes lata bem gelados", [("refrigerante lata", 3, ["bem gelado"])])
H("negacao", "1 lasanha. não tenho restrição alimentar nenhuma", [("lasanha", 1, [])])
H("negacao", "sou vegetariano então 2 pizzas margherita", [("pizza margherita", 2, [])], restricoes=["vegetariano"])
H("negacao", "intolerante a lactose aqui. 1 sorvete de chocolate sem cobertura", [("sorvete de chocolate", 1, ["sem cobertura"])], restricoes=["sem lactose"])
# quantidade por extenso
H("quantidade_extenso", "meia dúzia de coxinhas", [("coxinha", 6, [])])
H("quantidade_extenso", "uma dúzia de pastéis de queijo, pix", [("pastel de queijo", 12, [])], pagamento="pix")
H("quantidade_extenso", "quero um par de x-bacon", [("x-bacon", 2, [])])
H("quantidade_extenso", "dois hambúrgueres artesanais e três milkshakes de morango", [("hambúrguer artesanal", 2, []), ("milkshake de morango", 3, [])])
H("quantidade_extenso", "cinco esfihas de carne", [("esfiha de carne", 5, [])])
H("quantidade_extenso", "seis marmitas de frango, vale refeição", [("marmita de frango", 6, [])], pagamento="vale_refeicao")
H("quantidade_extenso", "duas dúzias de coxinhas", [("coxinha", 24, [])])
H("quantidade_extenso", "uma coxinha e um pastel de queijo, só isso", [("coxinha", 1, []), ("pastel de queijo", 1, [])])
# itens que o gerador nunca viu
H("fora_do_cardapio", "2 tapiocas de coco e 1 café", [("tapioca de coco", 2, []), ("café", 1, [])])
H("fora_do_cardapio", "1 sanduíche natural de frango, pago no pix", [("sanduíche natural de frango", 1, [])], pagamento="pix")
H("fora_do_cardapio", "quero 3 empadas de palmito", [("empada de palmito", 3, [])])
H("fora_do_cardapio", "1 pão de queijo e 1 cappuccino, dinheiro", [("pão de queijo", 1, []), ("cappuccino", 1, [])], pagamento="dinheiro")
H("fora_do_cardapio", "1 salada caesar sem croutons", [("salada caesar", 1, ["sem croutons"])])
H("fora_do_cardapio", "2 temakis de salmão com cream cheese extra", [("temaki de salmão", 2, ["com cream cheese extra"])])
# várias restrições e pagamento junto
H("restricao_pagamento", "alérgico a amendoim e intolerante a lactose, 1 açaí 500ml sem granola, crédito", [("açaí 500ml", 1, ["sem granola"])], restricoes=["alergia a amendoim", "sem lactose"], pagamento="credito")
H("restricao_pagamento", "vegetariano, não como carne. 2 pizzas margherita, 1 batata frita. débito", [("pizza margherita", 2, []), ("batata frita", 1, [])], restricoes=["vegetariano"], pagamento="debito")
H("restricao_pagamento", "1 lasanha, pago com cartão de crédito mesmo", [("lasanha", 1, [])], pagamento="credito")
H("restricao_pagamento", "2 coxinhas, pago tudo no pix mesmo", [("coxinha", 2, [])], pagamento="pix")
H("restricao_pagamento", "quero o combo sushi 20 peças, sem wasabi, vale-refeição", [("combo sushi 20 peças", 1, ["sem wasabi"])], pagamento="vale_refeicao")
H("restricao_pagamento", "1 marmita de carne sem cebola, preciso de opção sem glúten, pago em dinheiro", [("marmita de carne", 1, ["sem cebola"])], restricoes=["sem glúten"], pagamento="dinheiro")


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data" / "hard_test.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for i, (tag, texto, itens, restr, pag) in enumerate(CASOS, 1):
            pedido = Pedido(
                itens=[Item(nome=n, quantidade=q, observacoes=o) for n, q, o in itens],
                restricoes=restr,
                pagamento=pag,
            )
            row = {"id": f"hard-{i:03d}", "tag": tag, "texto": texto, "pedido": pedido.model_dump(mode="json")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(CASOS)} casos -> {out}")


if __name__ == "__main__":
    main()
