"""
api.py — FastAPI REST backend for Kifayati AI.

Endpoints:
  POST /chat          → main inference endpoint
  GET  /health        → liveness probe (used by GKE)
  GET  /metrics       → FinOps dashboard data
  GET  /circuit       → circuit breaker status
  POST /cache/clear   → flush the LRU cache

Run locally:
  uvicorn api:app --reload --port 8080
"""

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agents.agent    import KifayatiRouter
from agents.tracker  import metrics_tracker
from agents.fallback import circuit_breaker
from agents.evaluator import evaluator

log = logging.getLogger(__name__)

# ── Startup / shutdown ───────────────────────────────────────────
router: KifayatiRouter | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global router
    log.info("Kifayati API starting up…")
    router = KifayatiRouter()
    yield
    log.info("Kifayati API shutting down.")

# ── App ──────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Kifayati AI API",
    description = "Cost-optimised hybrid AI routing: Gemma 3 + Gemini 2.5 Flash",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Request / Response schemas ───────────────────────────────────
class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length = 1,
        max_length = 4000,
        example    = "Explain how transformers work",
    )

class ChatResponse(BaseModel):
    response       : str
    model          : str
    routing_reason : str
    latency_s      : float
    cost_usd       : float
    cache_hit      : bool
    complexity_score: float

class HealthResponse(BaseModel):
    status  : str
    uptime_s: float

class MetricsResponse(BaseModel):
    summary: dict
    logs   : list[dict]

# ── Uptime tracking ───────────────────────────────────────────────
_start_time = time.time()

# ── Routes ───────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health():
    """GKE liveness & readiness probe."""
    return {"status": "ok", "uptime_s": round(time.time() - _start_time, 1)}


@app.post("/chat", response_model=ChatResponse, tags=["Inference"])
async def chat(req: ChatRequest):
    """
    Main inference endpoint.
    Routes query to Gemma or Gemini based on complexity score.
    """
    if router is None:
        raise HTTPException(status_code=503, detail="Router not initialised")

    # Pre-evaluate complexity for the response payload
    eval_result = evaluator.evaluate(req.query)

    try:
        result = router.route_and_execute(req.query)
    except Exception as exc:
        log.error("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(
        response         = result["response"],
        model            = result["model"],
        routing_reason   = result["routing_reason"],
        latency_s        = result["latency_s"],
        cost_usd         = result["cost_usd"],
        cache_hit        = result["cache_hit"],
        complexity_score = eval_result.score,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["FinOps"])
async def metrics():
    """Real-time FinOps data — cost savings, latency, model usage."""
    return {
        "summary": metrics_tracker.summary(),
        "logs"   : metrics_tracker.all_logs(),
    }


@app.get("/circuit", tags=["Ops"])
async def circuit_status():
    """Circuit breaker state — shows if Gemma is healthy."""
    return circuit_breaker.status_dict()


@app.post("/cache/clear", tags=["Ops"])
async def clear_cache():
    """Flush the LRU response cache."""
    if router is None:
        raise HTTPException(status_code=503, detail="Router not initialised")
    router._cache._store.clear()
    return {"message": "Cache cleared", "timestamp": time.strftime("%H:%M:%S")}


# ── Global error handler ─────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code = 500,
        content     = {"detail": "Internal server error", "error": str(exc)},
    )
