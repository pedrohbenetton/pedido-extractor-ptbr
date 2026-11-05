import threading
import time
from types import SimpleNamespace as NS

import pytest

from pedido_extractor.api_runtime import CallStats, make_anthropic_generate_fn

MSGS = [
    {"role": "system", "content": "SISTEMA"},
    {"role": "user", "content": "2 coxinhas"},
]


class FakeError(Exception):
    pass


class FakeClient:
    def __init__(self, reply=None, stop_reason="end_turn", fail_on=None, delay=0.0):
        self.calls: list[dict] = []
        self.reply = reply or (lambda kw: kw["messages"][0]["content"].upper())
        self.stop_reason = stop_reason
        self.fail_on = fail_on
        self.delay = delay
        self._lock = threading.Lock()
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        with self._lock:
            self.calls.append(kw)
        time.sleep(self.delay)
        if self.fail_on and self.fail_on in kw["messages"][0]["content"]:
            raise FakeError("boom")
        blocks = [NS(type="thinking", thinking="segredo"), NS(type="text", text=self.reply(kw))]
        return NS(
            content=blocks, stop_reason=self.stop_reason, usage=NS(input_tokens=100, output_tokens=20)
        )


def _gen(client, **kw):
    return make_anthropic_generate_fn(client, "m", error_types=(FakeError,), **kw)


def test_system_vai_separado_e_sem_sampling():
    c = FakeClient()
    gen, _ = _gen(c)
    gen([MSGS])
    kw = c.calls[0]
    assert kw["system"] == "SISTEMA"
    assert kw["messages"] == [{"role": "user", "content": "2 coxinhas"}]
    assert kw["model"] == "m" and kw["max_tokens"] >= 1024
    assert "temperature" not in kw and "top_p" not in kw
    assert kw["output_config"] == {"effort": "low"}


def test_effort_none_omite_output_config():
    c = FakeClient()
    _gen(c, effort=None)[0]([MSGS])
    assert "output_config" not in c.calls[0]


def test_so_blocos_de_texto_entram_na_resposta():
    gen, _ = _gen(FakeClient(reply=lambda kw: '{"x": 1}'))
    assert gen([MSGS]) == ['{"x": 1}']


def test_ordem_preservada_com_varios_workers():
    c = FakeClient(delay=0.01)
    lote = [[MSGS[0], {"role": "user", "content": f"p{i}"}] for i in range(20)]
    gen, _ = _gen(c, workers=8)
    assert gen(lote) == [f"P{i}" for i in range(20)]


def test_estatisticas_agregadas():
    gen, stats = _gen(FakeClient())
    gen([MSGS] * 5)
    s = stats.summary()
    assert s["requests"] == 5 and s["errors"] == 0 and s["refusals"] == 0
    assert s["input_tokens"] == 500 and s["output_tokens"] == 100
    assert 0 <= s["latency_mean_s"] <= s["latency_p95_s"]


def test_erro_da_api_vira_resposta_vazia_e_e_contado():
    lote = [[MSGS[0], {"role": "user", "content": t}] for t in ("ok", "FALHA", "ok")]
    gen, stats = _gen(FakeClient(fail_on="FALHA"))
    out = gen([[m[0], {"role": "user", "content": m[1]["content"]}] for m in lote])
    assert out[1] == "" and out[0] == "OK"
    assert stats.summary()["errors"] == 1


def test_erro_inesperado_nao_e_engolido():
    class Bug(Exception):
        pass

    c = FakeClient(reply=lambda kw: (_ for _ in ()).throw(Bug()))
    gen, _ = _gen(c)
    with pytest.raises(Bug):
        gen([MSGS])


def test_refusal_contado():
    gen, stats = _gen(FakeClient(stop_reason="refusal"))
    gen([MSGS])
    assert stats.summary()["refusals"] == 1


def test_custo_estimado():
    s = CallStats()
    s.record(1_000_000, 500_000, 1.0)
    assert s.cost(price_in=4.0, price_out=20.0) == pytest.approx(4.0 + 10.0)
    assert s.cost(None, None) is None
