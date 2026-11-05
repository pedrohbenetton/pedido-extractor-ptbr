from pedido_extractor.data import build_splits, to_chat
from pedido_extractor.evaluation import render_report, run_eval
from pedido_extractor.hard import load_hard_test


def _oraculo(respostas: dict[str, str]):
    def gen(batch):
        return [respostas.get(msgs[1]["content"], "nao sei") for msgs in batch]

    return gen


def _tudo():
    teste = build_splits(n=100, seed=0)["test"]
    hard = load_hard_test()
    respostas = {e.texto: e.pedido.model_dump_json() for e in teste}
    respostas |= {h.texto: h.pedido.model_dump_json() for h in hard}
    return teste, hard, respostas


def test_run_eval_oraculo_acerta_tudo():
    teste, hard, respostas = _tudo()
    r = run_eval(_oraculo(respostas), teste, hard)
    assert r["synthetic"]["exact_match"] == 1.0
    assert r["hard"]["all"]["exact_match"] == 1.0


def test_run_eval_modelo_ruim_zera():
    teste, hard, _ = _tudo()
    r = run_eval(_oraculo({}), teste, hard)
    assert r["synthetic"]["valid_json_rate"] == 0.0 and r["hard"]["all"]["item_f1"] == 0.0


def test_run_eval_envia_mensagens_sem_resposta_e_em_lotes():
    teste, hard, respostas = _tudo()
    vistos = []

    def gen(batch):
        vistos.append(batch)
        assert all(m[-1]["role"] == "user" for m in batch)
        return [respostas[m[1]["content"]] for m in batch]

    run_eval(gen, teste, hard, batch_size=16)
    assert max(len(b) for b in vistos) <= 16
    assert sum(len(b) for b in vistos) == len(teste) + len(hard)
    assert to_chat(teste[0], with_answer=False)[1]["content"] == vistos[0][0][1]["content"]


def test_render_report_tabela_markdown():
    teste, hard, respostas = _tudo()
    bom = run_eval(_oraculo(respostas), teste, hard)
    ruim = run_eval(_oraculo({}), teste, hard)
    md = render_report({"base": ruim, "lora": bom})
    assert "| base |" in md and "| lora |" in md
    assert "100.0%" in md and "0.0%" in md
    assert "correcao" in md and "quantidade_extenso" in md
