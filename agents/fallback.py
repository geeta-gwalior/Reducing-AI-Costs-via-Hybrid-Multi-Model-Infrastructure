"""
FallbackHandler — Circuit Breaker pattern for Kifayati AI.

If Gemma endpoint fails repeatedly, automatically switch all
traffic to Gemini until Gemma recovers. This ensures 100% uptime
even when the Vertex AI endpoint is unhealthy.

States:
  CLOSED   → Normal. Gemma is healthy, routing works as usual.
  OPEN     → Gemma has failed too many times. All traffic → Gemini.
  HALF_OPEN→ Testing if Gemma has recovered. 1 trial request sent.
"""

import time
import logging
from enum import Enum
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED    = "closed"      # healthy — normal routing
    OPEN      = "open"        # unhealthy — all to Gemini
    HALF_OPEN = "half_open"   # testing recovery


@dataclass
class CircuitBreaker:
    """
    Tracks Gemma endpoint health and opens/closes the circuit.

    Parameters
    ----------
    failure_threshold : int
        How many consecutive failures before opening the circuit.
    recovery_timeout  : float
        Seconds to wait before trying Gemma again (HALF_OPEN).
    success_threshold : int
        How many successes in HALF_OPEN to close the circuit again.
    """
    failure_threshold : int   = 3
    recovery_timeout  : float = 30.0
    success_threshold : int   = 2

    _state            : CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count    : int          = field(default=0,    init=False)
    _success_count    : int          = field(default=0,    init=False)
    _last_failure_time: float        = field(default=0.0,  init=False)

    # ── State queries ────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                log.info("[CircuitBreaker] Recovery timeout passed → HALF_OPEN")
                self._state = CircuitState.HALF_OPEN
        return self._state

    @property
    def is_gemma_available(self) -> bool:
        """True if Gemma should be tried (CLOSED or HALF_OPEN)."""
        return self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    # ── Event recording ──────────────────────────────────────────

    def record_success(self):
        """Call after a successful Gemma response."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                log.info("[CircuitBreaker] Gemma recovered → CLOSED")
                self._reset()
        else:
            self._failure_count = 0

    def record_failure(self):
        """Call after a Gemma failure."""
        self._failure_count    += 1
        self._last_failure_time = time.time()
        self._success_count     = 0

        if self._failure_count >= self.failure_threshold:
            log.warning(
                "[CircuitBreaker] %d failures → OPEN. Routing all to Gemini.",
                self._failure_count,
            )
            self._state = CircuitState.OPEN

    # ── Helpers ──────────────────────────────────────────────────

    def _reset(self):
        self._state         = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0

    def status_dict(self) -> dict:
        return {
            "circuit_state"  : self.state.value,
            "failure_count"  : self._failure_count,
            "gemma_available": self.is_gemma_available,
        }


# ── Singleton ────────────────────────────────────────────────────
circuit_breaker = CircuitBreaker()
