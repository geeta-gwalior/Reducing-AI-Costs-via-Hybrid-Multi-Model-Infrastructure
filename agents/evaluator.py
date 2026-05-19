"""
QueryEvaluator — Scores incoming queries for complexity.

Instead of a simple keyword check, this module assigns a
complexity score (0.0 → 1.0) using multiple signals:

  Signal 1 — Token count          (longer = more complex)
  Signal 2 — Complex keyword hit  (technical terms)
  Signal 3 — Simple keyword hit   (greetings / trivial)
  Signal 4 — Question depth       (nested clauses, "why", "how")
  Signal 5 — Code indicators      (backticks, language names)

Score ≥ COMPLEX_THRESHOLD → Gemini 2.5 Flash
Score <  COMPLEX_THRESHOLD → Gemma 3:4b
"""

import re
import logging
from dataclasses import dataclass

from .config import (
    SIMPLE_KEYWORDS, COMPLEX_KEYWORDS,
    SIMPLE_TOKEN_THRESHOLD,
)

log = logging.getLogger(__name__)

# Tune this to shift more/less traffic to Gemma
COMPLEX_THRESHOLD = 0.40

# Regex for code indicators
_CODE_PATTERN = re.compile(
    r"```|`[^`]+`|\b(python|javascript|typescript|java|golang|rust|sql|bash|curl)\b",
    re.IGNORECASE,
)

# Deep-question words that imply reasoning
_REASONING_WORDS = {
    "why", "how", "explain", "compare", "difference",
    "analyse", "analyze", "evaluate", "describe", "elaborate",
    "what is the impact", "pros and cons", "trade-off",
}


@dataclass
class EvalResult:
    score           : float   # 0.0 (trivial) → 1.0 (very complex)
    routed_to       : str     # "Gemma 3:4b" | "Gemini 2.5 Flash"
    routing_reason  : str     # human-readable explanation
    token_count     : int
    has_code        : bool
    has_complex_kw  : bool
    has_simple_kw   : bool
    has_reasoning   : bool


class QueryEvaluator:
    """
    Stateless complexity scorer.
    Call evaluate(query) → EvalResult.
    """

    def evaluate(self, query: str) -> EvalResult:
        q      = query.strip()
        q_low  = q.lower()
        words  = set(q_low.split())
        tokens = len(q.split())

        # ── Signal 1: Token count ────────────────────────────────
        # Normalise: 0 tokens = 0.0, 30+ tokens = 0.5 max from this signal
        token_score = min(tokens / 30, 1.0) * 0.5

        # ── Signal 2: Complex keyword ────────────────────────────
        has_complex_kw = any(kw in words for kw in COMPLEX_KEYWORDS)
        complex_score  = 0.50 if has_complex_kw else 0.0

        # ── Signal 3: Simple keyword (negative signal) ───────────
        has_simple_kw = any(kw in words for kw in SIMPLE_KEYWORDS)
        simple_penalty = -0.35 if has_simple_kw else 0.0

        # ── Signal 4: Reasoning words ────────────────────────────
        has_reasoning  = any(rw in q_low for rw in _REASONING_WORDS)
        reasoning_score = 0.30 if has_reasoning else 0.0

        # ── Signal 5: Code indicators ────────────────────────────
        has_code   = bool(_CODE_PATTERN.search(q))
        code_score = 0.40 if has_code else 0.0

        # ── Final score ──────────────────────────────────────────
        raw_score = (
            token_score
            + complex_score
            + simple_penalty
            + reasoning_score
            + code_score
        )
        score = max(0.0, min(raw_score, 1.0))   # clamp to [0, 1]

        # ── Routing decision ─────────────────────────────────────
        if score >= COMPLEX_THRESHOLD:
            routed_to      = "Gemini 2.5 Flash"
            routing_reason = self._reason(
                has_complex_kw, has_reasoning, has_code, tokens, is_complex=True
            )
        else:
            routed_to      = "Gemma 3:4b"
            routing_reason = self._reason(
                has_complex_kw, has_reasoning, has_code, tokens, is_complex=False
            )

        result = EvalResult(
            score          = round(score, 3),
            routed_to      = routed_to,
            routing_reason = routing_reason,
            token_count    = tokens,
            has_code       = has_code,
            has_complex_kw = has_complex_kw,
            has_simple_kw  = has_simple_kw,
            has_reasoning  = has_reasoning,
        )

        log.info(
            "[Evaluator] score=%.3f → %s | reason=%s",
            score, routed_to, routing_reason,
        )
        return result

    @staticmethod
    def _reason(
        has_complex_kw: bool,
        has_reasoning : bool,
        has_code      : bool,
        tokens        : int,
        is_complex    : bool,
    ) -> str:
        if has_code:
            return "code_detected"
        if has_complex_kw:
            return "complex_keyword"
        if has_reasoning:
            return "reasoning_required"
        if tokens >= SIMPLE_TOKEN_THRESHOLD and is_complex:
            return "token_count_high"
        if tokens < SIMPLE_TOKEN_THRESHOLD and not is_complex:
            return "token_count_low"
        return "default_simple" if not is_complex else "default_complex"


# ── Singleton ────────────────────────────────────────────────────
evaluator = QueryEvaluator()
