import os
from dotenv import load_dotenv

load_dotenv()

# ── Google Cloud Config ──────────────────────────────────────────
PROJECT_ID       = os.getenv("PROJECT_ID")
LOCATION         = os.getenv("LOCATION", "us-central1")
GEMMA_ENDPOINT_ID = os.getenv("GEMMA_ENDPOINT_ID")

# ── Routing Thresholds ───────────────────────────────────────────
# Queries with token count BELOW this go to Gemma (cheap)
SIMPLE_TOKEN_THRESHOLD = 6

# Keywords that always route to Gemma (greetings / trivial)
SIMPLE_KEYWORDS = [
    'hi', 'hello', 'hey', 'thanks', 'thank you', 'bye',
    'weather', 'joke', 'name', 'who are you'
]

# Keywords that always route to Gemini (force complex)
COMPLEX_KEYWORDS = [
    'code', 'explain', 'debug', 'analyse', 'analyze',
    'compare', 'summarise', 'summarize', 'write', 'generate',
    'difference', 'how does', 'why does', 'architecture'
]

# ── Cost Assumptions (USD per request) ──────────────────────────
COST_GEMMA   = 0.00001   # Vertex AI managed endpoint
COST_GEMINI  = 0.0001    # Gemini 2.5 Flash API

# ── Cache Config ────────────────────────────────────────────────
CACHE_MAX_SIZE = 500      # max entries in LRU cache
CACHE_TTL_SEC  = 3600     # 1 hour TTL

# ── Retry Config ────────────────────────────────────────────────
MAX_RETRIES    = 3
RETRY_DELAY    = 1.5      # seconds (exponential backoff base)

def validate_config():
    """Raise early if critical env vars are missing."""
    missing = []
    if not PROJECT_ID:
        missing.append("PROJECT_ID")
    if not GEMMA_ENDPOINT_ID:
        missing.append("GEMMA_ENDPOINT_ID")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Please check your .env file."
        )
