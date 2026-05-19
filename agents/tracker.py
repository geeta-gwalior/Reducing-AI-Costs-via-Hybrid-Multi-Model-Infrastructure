"""
KifayatiMetrics — FinOps tracker for the hybrid routing system.
Tracks every inference call: model used, latency, cost, routing decision.
"""
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Literal

from .config import COST_GEMMA, COST_GEMINI, CACHE_MAX_SIZE


@dataclass
class InferenceLog:
    timestamp:  str
    model:      Literal["Gemma 3:4b", "Gemini 2.5 Flash", "Cache"]
    query:      str
    response_len: int
    latency_s:  float
    cost_usd:   float
    routing_reason: str   # e.g. "keyword_match", "token_count", "complex_keyword"
    cache_hit:  bool = False


class KifayatiMetrics:
    """
    Lightweight in-process metrics store.
    Stores last CACHE_MAX_SIZE logs (rolling window).
    """

    _COSTS = {
        "Gemma 3:4b":      COST_GEMMA,
        "Gemini 2.5 Flash": COST_GEMINI,
        "Cache":            0.0,
    }

    def __init__(self):
        self._logs: deque[InferenceLog] = deque(maxlen=CACHE_MAX_SIZE)

    # ── Public API ───────────────────────────────────────────────

    def log(
        self,
        model: str,
        query: str,
        response: str,
        latency_s: float,
        routing_reason: str,
        cache_hit: bool = False,
    ) -> InferenceLog:
        entry = InferenceLog(
            timestamp      = time.strftime("%H:%M:%S"),
            model          = model,
            query          = query[:60] + ("…" if len(query) > 60 else ""),
            response_len   = len(response),
            latency_s      = round(latency_s, 3),
            cost_usd       = self._COSTS.get(model, 0.0),
            routing_reason = routing_reason,
            cache_hit      = cache_hit,
        )
        self._logs.append(entry)
        return entry

    def all_logs(self) -> list[dict]:
        return [asdict(l) for l in self._logs]

    def summary(self) -> dict:
        if not self._logs:
            return {"total_requests": 0}

        total       = len(self._logs)
        gemma_calls = sum(1 for l in self._logs if l.model == "Gemma 3:4b")
        gemini_calls= sum(1 for l in self._logs if l.model == "Gemini 2.5 Flash")
        cache_hits  = sum(1 for l in self._logs if l.cache_hit)
        total_cost  = sum(l.cost_usd for l in self._logs)
        baseline    = total * COST_GEMINI          # if we'd used Gemini for everything
        saved       = baseline - total_cost
        avg_latency = sum(l.latency_s for l in self._logs) / total

        return {
            "total_requests":  total,
            "gemma_calls":     gemma_calls,
            "gemini_calls":    gemini_calls,
            "cache_hits":      cache_hits,
            "total_cost_usd":  round(total_cost, 6),
            "baseline_cost_usd": round(baseline, 6),
            "total_saved_usd": round(saved, 6),
            "avg_latency_s":   round(avg_latency, 3),
            "cost_reduction_pct": round((saved / baseline * 100) if baseline else 0, 1),
        }


# ── Singleton (shared across the app) ───────────────────────────
metrics_tracker = KifayatiMetrics()
