"""
Unit tests for Kifayati routing logic and metrics tracker.
Run with: pytest tests/ -v
"""
import sys
import pytest
from unittest.mock import MagicMock, patch

# Mock GCP libraries BEFORE any project import
for mod in [
    "vertexai",
    "vertexai.generative_models",
    "google",
    "google.cloud",
    "google.cloud.aiplatform",
    "google.cloud.aiplatform_v1",
]:
    sys.modules.setdefault(mod, MagicMock())


# ── Config tests ─────────────────────────────────────────────────
class TestConfig:
    def test_simple_keywords_not_empty(self):
        from agents.config import SIMPLE_KEYWORDS
        assert len(SIMPLE_KEYWORDS) > 0

    def test_complex_keywords_not_empty(self):
        from agents.config import COMPLEX_KEYWORDS
        assert len(COMPLEX_KEYWORDS) > 0

    def test_cost_gemma_cheaper_than_gemini(self):
        from agents.config import COST_GEMMA, COST_GEMINI
        assert COST_GEMMA < COST_GEMINI


# ── Tracker tests ────────────────────────────────────────────────
class TestKifayatiMetrics:
    def setup_method(self):
        from agents.tracker import KifayatiMetrics
        self.tracker = KifayatiMetrics()

    def test_empty_summary(self):
        s = self.tracker.summary()
        assert s["total_requests"] == 0

    def test_log_gemma_call(self):
        self.tracker.log("Gemma 3:4b", "hi", "Hello!", 0.5, "simple_keyword")
        s = self.tracker.summary()
        assert s["total_requests"] == 1
        assert s["gemma_calls"] == 1
        assert s["gemini_calls"] == 0

    def test_log_gemini_call(self):
        self.tracker.log("Gemini 2.5 Flash", "explain transformers", "...", 1.2, "complex_keyword")
        s = self.tracker.summary()
        assert s["gemini_calls"] == 1

    def test_cache_hit_zero_cost(self):
        self.tracker.log("Cache", "hi", "Hello!", 0.0, "cache_hit", cache_hit=True)
        s = self.tracker.summary()
        assert s["cache_hits"] == 1
        assert s["total_cost_usd"] == 0.0

    def test_cost_savings_calculated(self):
        self.tracker.log("Gemma 3:4b", "hi",    "Hello!",   0.2, "simple_keyword")
        self.tracker.log("Gemma 3:4b", "hello", "Hi there!", 0.2, "simple_keyword")
        s = self.tracker.summary()
        assert s["total_saved_usd"] > 0
        assert s["cost_reduction_pct"] > 0


# ── LRU Cache tests ───────────────────────────────────────────────
class TestLRUCache:
    def setup_method(self):
        from agents.agent import LRUCache
        self.cache = LRUCache(max_size=3)

    def test_miss_returns_none(self):
        assert self.cache.get("missing") is None

    def test_set_and_get(self):
        self.cache.set("hello", "world")
        assert self.cache.get("hello") == "world"

    def test_evicts_oldest(self):
        self.cache.set("a", "1")
        self.cache.set("b", "2")
        self.cache.set("c", "3")
        self.cache.set("d", "4")   # "a" should be evicted
        assert self.cache.get("a") is None
        assert self.cache.get("d") == "4"


# ── Routing logic tests ───────────────────────────────────────────
class TestRoutingLogic:
    def setup_method(self):
        from agents.agent import KifayatiRouter
        with patch("agents.agent.validate_config"), \
             patch("agents.agent.vertexai.init"), \
             patch("agents.agent.aiplatform.Endpoint"), \
             patch("agents.agent.GenerativeModel"):
            self.router = KifayatiRouter()

    def test_hi_routes_gemma(self):
        model, reason = self.router._route("hi")
        assert model == "Gemma 3:4b"
        assert reason == "simple_keyword"

    def test_explain_routes_gemini(self):
        model, reason = self.router._route("explain how transformers work in detail")
        assert model == "Gemini 2.5 Flash"
        assert reason == "complex_keyword"

    def test_short_query_routes_gemma(self):
        model, reason = self.router._route("what time")
        assert model == "Gemma 3:4b"
        assert reason == "token_count"

    def test_complex_keyword_beats_simple_keyword(self):
        model, reason = self.router._route("hello, can you explain neural networks?")
        assert model == "Gemini 2.5 Flash"
        assert reason == "complex_keyword"

    def test_long_unknown_query_defaults_gemini(self):
        q = "the quick brown fox jumps over the lazy dog many many times"
        model, reason = self.router._route(q)
        assert model == "Gemini 2.5 Flash"
        assert reason == "default_complex"
