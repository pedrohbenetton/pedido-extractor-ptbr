<!-- RASCUNHO LOCAL. Não publicado em nenhum lugar. Revise tudo antes de postar. -->

# Fine-tuning de um modelo de 1,5B para pedidos de delivery em PT-BR: o que funcionou, o que errei e o que ainda não sei

**Resumo.** Ajustei com QLoRA um Qwen2.5-1.5B-Instruct para transformar mensagens livres de pedido de delivery em JSON validado. No conjunto difícil que montei, o exact match foi de 3,3% (modelo base) para 71,7% (v1) e 90,0% (v2), contra 68,3% a 75,0% de um Claude Haiku 5.5 zero-shot (a API não permite fixar a aleatoriedade, e as duas execuções deram valores diferentes). Depois montei um pequeno benchmark de PT-BR informal para ver o que o fine-tuning custava nas outras tarefas. Pelo caminho, cometi três erros de medição, e conto cada um aqui, porque eles ensinaram mais do que os acertos.

Código, dados e testes: [pedido-extractor-ptbr](https://github.com/pedrohbenetton/pedido-extractor-ptbr) · Benchmark: [ptbr-llm-bench](https://github.com/pedrohbenetton/ptbr-llm-bench) · Adapter: [Hugging Face](https://huggingface.co/pedrohbenetton/pedido-extractor-ptbr-lora)

## O problema

Num app de delivery, o cliente escreve como fala:

> "2 pizzas de calabresa sem cebola, ah não, 3. pago no pix"

O sistema precisa de uma estrutura, não de texto: itens (nome canônico, quantidade, observações), restrições alimentares e forma de pagamento.

```json
{"itens":[{"nome":"pizza de calabresa","quantidade":3,"observacoes":["sem cebola"]}],"restricoes":[],"pagamento":"pix"}
```

O desafio não é o formato. É a língua: correção no meio da frase ("ah não, 3"), gíria, erro de digitação, quantidade por extenso ("meia dúzia", "um par"), negação ("esquece o refri") e itens que não existem no cardápio.

## Os dados: dois conjuntos com papéis diferentes

**Treino e teste sintético.** Um gerador determinístico cria o texto e o rótulo ao mesmo tempo, então não há ruído de anotação. Gerei 6.000 exemplos: 4.800 para treino, 600 para validação e 600 para teste. O problema desse teste é que ele vem do mesmo gerador do treino: perto de 100% mostra que o formato foi aprendido, não que o problema está resolvido.

**Conjunto difícil.** 60 casos escritos por mim, em 8 categorias (correção, gíria, typo, ordem confusa, negação, quantidade por extenso, item fora do cardápio, restrição com pagamento). Eles nunca entram no treino, e um teste automático garante que nenhum texto deles é gerado. Dois limites que preciso dizer: os gabaritos foram escritos por mim com ajuda de IA e **sem revisão externa**, e alguns casos são ambíguos (observações compostas, "bem gelado"). Convenções como "dúzia = 12" são decisões minhas.

O número que importa é o do conjunto difícil.

## v1: o formato aprendeu, a língua não

QLoRA (4 bits, r=16, alpha=32, 2 épocas, lr 2e-4), numa T4 do Colab.

| | JSON válido (difícil) | Exact match (difícil) |
|---|---|---|
| Qwen2.5-1.5B base | 68,3% | 3,3% |
| + LoRA v1 | 100% | 71,7% |

No teste sintético o v1 chegou a 99,7%. No difícil, 71,7%. Essa distância é a distância entre dado de template e linguagem real, e se concentrava onde o gerador não cobria: **correção no meio da frase** (25% de acerto), typo e quantidade por extenso (62,5% cada).

## v2: mudar só os dados

Mantive modelo, hiperparâmetros e teste. Só mudei o gerador para cobrir correções, itens pedidos e depois cancelados ou trocados, typos (letra trocada, omitida, dobrada, erro fonético) e quantidade por extenso, com marcadores e frases propositalmente diferentes dos do conjunto difícil.

Exact match no difícil: **71,7% → 90,0%**. Uma categoria piorou: `fora_do_cardapio` caiu de 100% para 83,3%, que é um caso a menos entre seis.

Uma ressalva que precisa ficar junto do número: o v2 ensina de propósito conceitos que o teste cobra. Parte do ganho em correção, typo e extenso é o que foi ensinado, não generalização livre.

## E o Claude Haiku?

Rodei o Claude Haiku 5.5 via API, zero-shot (só o prompt de sistema, sem exemplos, esforço baixo). Ele deu 68,3% de exact match na primeira execução e 75,0% numa segunda; a API não permite fixar a aleatoriedade, então esses dois números são a faixa real. Ele erra sobretudo em convenções do rótulo (nome no singular, observação canônica "sem X"), que o fine-tuning ensina e o prompt sozinho não. Por isso a comparação mede em parte **aderência à convenção**, não "quem entende melhor português". Uma versão do Haiku com exemplos no prompt seria mais justa, e eu ainda não a rodei.

## O benchmark: o que o fine-tuning custou?

Um adapter treinado para uma tarefa pode piorar nas outras. Para olhar isso, montei o [ptbr-llm-bench](https://github.com/pedrohbenetton/ptbr-llm-bench): três tarefas escritas com ajuda de IA e revisadas com apoio externo (intenção de mensagem de cliente, 48 itens; interpretação ambígua com gíria, ironia e regionalismo, 40 itens; e a extração de pedido, 60 itens).

| Modelo | Média | extração | intenção | ambiguidade (rígida / tolerante) |
|---|---|---|---|---|
| Claude Haiku 5.5 | 95,6 | 89,4 | 100,0 | 97,5 / 97,5 |
| Qwen + LoRA v2 | 66,9 | 96,1 | 64,6 | 40,0 / 70,0 |
| Qwen + LoRA v1 | 62,3 | 89,4 | 50,0 | 47,5 / 65,0 |
| Qwen base | 50,2 | 40,6 | 62,5 | 47,5 / 77,5 |

(Na extração essa nota é por campo e mais estrita com ordem e acento que o exact match; com a métrica do projeto, as mesmas respostas dão 3,3% (base), 71,7% (v1), 90,0% (v2) e 75,0% (Haiku).)

O que eu leio daqui, com cuidado:

- O v1 perdia cerca de 12 pontos na intenção e na ambiguidade contra o base. O v2 **recupera a intenção** (64,6 contra 62,5, uma diferença de um item) e ainda perde na ambiguidade tolerante (três itens). Ou seja, o custo de especialização não é uniforme, e eu não sei explicar por que o v2 preserva mais.
- Intenção e ambiguidade não separam modelos fortes: o Haiku passa de 97%. Hoje elas só distinguem modelos pequenos.
- São uma execução, 40 a 60 itens por tarefa. Diferenças de um a três itens são ruído.

## Três erros que cometi

1. **Publiquei o adapter errado.** Subi no Hugging Face os pesos do v1 com o model card do v2. Percebi ao repontuar as respostas do benchmark com a métrica do projeto: deu 71,7% em vez de 90,0%, e as oito categorias batiam uma a uma com as do v1. Reenviei os pesos do v2 e repeti a rodada, e os números voltaram ao esperado. A lição prática: conferir o artefato publicado pelos números, não pelo nome da pasta.
2. **Meu primeiro scorer era rígido demais.** Na ambiguidade, respostas como `b) elogio sincero` valiam zero mesmo com a letra certa, porque eu comparava a linha inteira. Criei uma nota tolerante, mas **depois de ver o resultado**, e por isso mostro as duas notas lado a lado, sem trocar uma pela outra em silêncio.
3. **Minha primeira explicação estava errada.** Achei que o LoRA estava respondendo JSON em vez da etiqueta, o que explicaria a queda na intenção. Gravei as respostas brutas e testei: 0 de 88 respostas começavam com `{`. A hipótese foi descartada, e a causa da queda continua em aberto.

## O que eu não sei

- Se a vantagem do v2 sobre o Haiku sobrevive a um Haiku com exemplos no prompt.
- Quanto do ganho do v2 é generalização e quanto é o que o gerador ensinou.
- Por que o v2 preserva a intenção melhor que o v1.
- Como ficam os números com mais de uma seed e com gabaritos revisados por outra pessoa.

## Reproduzir

Os dois repositórios têm testes e CI. O treino roda numa T4 gratuita do Colab (`notebooks/train_colab.ipynb`), e as respostas brutas de todos os modelos estão em `results/published/` do benchmark.

---

*Pedro Benetton, estudante de Ciência da Computação na UFSCar Sorocaba.*
