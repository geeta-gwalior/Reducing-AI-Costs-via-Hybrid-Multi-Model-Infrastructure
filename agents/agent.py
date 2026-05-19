
import time
import logging
from collections import OrderedDict

import vertexai
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel

from .config import (
    PROJECT_ID, LOCATION, GEMMA_ENDPOINT_ID,
    SIMPLE_KEYWORDS, COMPLEX_KEYWORDS,
    SIMPLE_TOKEN_THRESHOLD, CACHE_MAX_SIZE,
    MAX_RETRIES, RETRY_DELAY, validate_config,
)
from .tracker import metrics_tracker

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Vertex AI init ───────────────────────────────────────────────
# validate_config() called inside KifayatiRouter.__init__
vertexai.init(project=PROJECT_ID, location=LOCATION)


# ── LRU Cache (simple in-process) ───────────────────────────────
class LRUCache:
    def __init__(self, max_size: int = CACHE_MAX_SIZE):
        self._store: OrderedDict[str, str] = OrderedDict()
        self._max   = max_size

    def get(self, key: str):
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: str, value: str):
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self._max:
            self._store.popitem(last=False)

    def __len__(self):
        return len(self._store)


# ── Agent wrapper ────────────────────────────────────────────────
class LlmAgent:
    def __init__(self, name: str, model_type: str, instruction: str):
        self.name        = name
        self.model_type  = model_type
        self.instruction = instruction

        if model_type == "gemini":
            self.model = GenerativeModel("gemini-2.5-flash")

        elif model_type == "gemma":
            self.endpoint = aiplatform.Endpoint(
                endpoint_name=(
                    f"projects/{PROJECT_ID}/locations/{LOCATION}"
                    f"/endpoints/{GEMMA_ENDPOINT_ID}"
                )
            )

    def ask(self, query: str) -> str:
        prompt = f"{self.instruction}\n\nUser: {query}\nAnswer:"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if self.model_type == "gemini":
                    resp = self.model.generate_content(prompt)
                    return resp.text

                elif self.model_type == "gemma":
                    instances = [{"prompt": prompt, "max_tokens": 300}]
                    resp      = self.endpoint.predict(instances=instances)
                    pred      = resp.predictions[0]
                    # Gemma returns either a string or a dict with 'content'
                    return pred if isinstance(pred, str) else pred.get("content", str(pred))

            except Exception as exc:
                wait = RETRY_DELAY * (2 ** (attempt - 1))
                log.warning(
                    "[%s] attempt %d/%d failed: %s. Retrying in %.1fs…",
                    self.name, attempt, MAX_RETRIES, exc, wait,
                )
                if attempt == MAX_RETRIES:
                    raise RuntimeError(
                        f"{self.name} failed after {MAX_RETRIES} attempts: {exc}"
                    ) from exc
                time.sleep(wait)


# ── Router ───────────────────────────────────────────────────────
class KifayatiRouter:
    def __init__(self):
        validate_config()
        self._cache = LRUCache()

        self.gemma_agent = LlmAgent(
            name        = "Gemma_Worker",
            model_type  = "gemma",
            instruction = (
                "You are a helpful, concise assistant. "
                "Keep answers short and friendly."
            ),
        )
        self.gemini_agent = LlmAgent(
            name        = "Gemini_Expert",
            model_type  = "gemini",
            instruction = (
                "You are an expert AI assistant. "
                "Provide accurate, detailed, well-structured explanations."
            ),
        )

    # ── Routing decision ────────────────────────────────────────
    def _route(self, query: str) -> tuple[str, str]:
        """
        Returns (model_label, routing_reason).
        model_label: "Gemma 3:4b" | "Gemini 2.5 Flash"
        """
        q = query.lower().strip()

        # Complex keywords win first (highest priority)
        words = set(q.split())
        if any(kw in words for kw in COMPLEX_KEYWORDS):
            return "Gemini 2.5 Flash", "complex_keyword"

        # Simple keyword check
        if any(kw in words for kw in SIMPLE_KEYWORDS):
            return "Gemma 3:4b", "simple_keyword"

        # Token count heuristic (rough: split on spaces)
        token_count = len(query.split())
        if token_count < SIMPLE_TOKEN_THRESHOLD:
            return "Gemma 3:4b", "token_count"

        # Default → Gemini for anything else
        return "Gemini 2.5 Flash", "default_complex"

    # ── Main entry point ────────────────────────────────────────
    def route_and_execute(self, user_input: str) -> dict:
        """
        Returns a dict with:
          response, model, routing_reason, latency_s, cost_usd, cache_hit
        """
        cache_key = user_input.lower().strip()

        # ── Cache check ──────────────────────────────────────────
        cached = self._cache.get(cache_key)
        if cached:
            log.info("[CACHE HIT] '%s…'", cache_key[:30])
            entry = metrics_tracker.log(
                model          = "Cache",
                query          = user_input,
                response       = cached,
                latency_s      = 0.0,
                routing_reason = "cache_hit",
                cache_hit      = True,
            )
            return {
                "response":       cached,
                "model":          "Cache",
                "routing_reason": "cache_hit",
                "latency_s":      0.0,
                "cost_usd":       0.0,
                "cache_hit":      True,
            }

        # ── Route & call ─────────────────────────────────────────
        model_label, reason = self._route(user_input)
        agent = (
            self.gemma_agent
            if model_label == "Gemma 3:4b"
            else self.gemini_agent
        )

        log.info("[ROUTING] %s → %s (%s)", cache_key[:30], model_label, reason)

        t0       = time.time()
        response = agent.ask(user_input)
        latency  = round(time.time() - t0, 3)

        # ── Cache store ──────────────────────────────────────────
        self._cache.set(cache_key, response)

        # ── Log metrics ──────────────────────────────────────────
        entry = metrics_tracker.log(
            model          = model_label,
            query          = user_input,
            response       = response,
            latency_s      = latency,
            routing_reason = reason,
            cache_hit      = False,
        )

        return {
            "response":       response,
            "model":          model_label,
            "routing_reason": reason,
            "latency_s":      latency,
            "cost_usd":       entry.cost_usd,
            "cache_hit":      False,
        }
