"""Baseline com LLM de API (Anthropic): mesma interface `generate_fn` dos modelos locais,
mais medição de latência, tokens e custo. O SDK já faz retry de 429/5xx."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .evaluation import GenerateFn, Messages


class CallStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.latencies: list[float] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.errors = 0
        self.refusals = 0

    def record(self, input_tokens: int, output_tokens: int, latency: float, refusal: bool = False) -> None:
        with self._lock:
            self.latencies.append(latency)
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.refusals += refusal

    def record_error(self, latency: float) -> None:
        with self._lock:
            self.latencies.append(latency)
            self.errors += 1

    def cost(self, price_in: float | None, price_out: float | None) -> float | None:
        """Custo em US$ a partir de preços por 1M de tokens informados por quem roda."""
        if price_in is None or price_out is None:
            return None
        return self.input_tokens / 1e6 * price_in + self.output_tokens / 1e6 * price_out

    def summary(self) -> dict[str, float | int]:
        lat = sorted(self.latencies)
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0
        return {
            "requests": len(lat),
            "errors": self.errors,
            "refusals": self.refusals,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_mean_s": sum(lat) / len(lat) if lat else 0.0,
            "latency_p95_s": p95,
        }


def make_anthropic_generate_fn(
    client,
    model: str,
    *,
    effort: str | None = "low",
    max_tokens: int = 2048,
    workers: int = 8,
    error_types: tuple[type[Exception], ...] | None = None,
) -> tuple[GenerateFn, CallStats]:
    """`effort=None` omite `output_config` (modelos que não suportam). Sem `temperature`:
    foi removido nos modelos atuais. Erros da API viram resposta vazia (contam como JSON inválido)."""
    if error_types is None:
        import anthropic

        error_types = (anthropic.APIError,)
    stats = CallStats()

    def one(msgs: Messages) -> str:
        kw: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": msgs[0]["content"],
            "messages": msgs[1:],
        }
        if effort:
            kw["output_config"] = {"effort": effort}
        t0 = time.perf_counter()
        try:
            resp = client.messages.create(**kw)
        except error_types:
            stats.record_error(time.perf_counter() - t0)
            return ""
        stats.record(
            resp.usage.input_tokens,
            resp.usage.output_tokens,
            time.perf_counter() - t0,
            refusal=resp.stop_reason == "refusal",
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    def gen(batch: list[Messages]) -> list[str]:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(one, batch))

    return gen, stats
