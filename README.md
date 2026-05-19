# Kifayati AI — Cost-Optimized Hybrid Multi-Model Infrastructure

> **"Why pay for a jet when you only need a bicycle?"**
> Kifayati AI intelligently routes every query to the right model — saving up to 90% on AI inference costs without sacrificing quality.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![GCP](https://img.shields.io/badge/Google_Cloud-Vertex_AI-4285F4?logo=googlecloud)
![GKE](https://img.shields.io/badge/GKE-Kubernetes-326CE5?logo=kubernetes)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit)
![Tests](https://img.shields.io/badge/Tests-16_passing-brightgreen)

---

## The Problem

In today's GenAI landscape, developers default to using the most powerful (and expensive) models for **every** query — including simple greetings, one-liners, and trivial lookups. This leads to:

- **Unsustainable API costs** at scale
- **Unnecessary latency** for simple queries
- **Wasted compute** on tasks that don't need high reasoning

---

## The Solution — Hybrid Intelligent Routing

Kifayati AI acts as a **smart traffic controller** sitting in front of your AI models. Before calling any model, it evaluates query complexity using a **5-signal scoring engine** and routes accordingly:

```
User Query
    │
    ▼
┌─────────────────────────────┐
│      QueryEvaluator         │  ← 5-signal complexity score (0.0 → 1.0)
│  • Token count              │
│  • Complex keywords         │
│  • Reasoning depth          │
│  • Code detection           │
│  • Simple keyword penalty   │
└────────────┬────────────────┘
             │
      ┌──────┴──────┐
      │             │
  score < 0.4   score >= 0.4
      │             │
      ▼             ▼
┌──────────┐  ┌──────────────────┐
│ Gemma 3  │  │ Gemini 2.5 Flash │
│  (Edge)  │  │    (Cloud)       │
│ $0.00001 │  │   $0.0001        │
└──────────┘  └──────────────────┘
      │             │
      └──────┬──────┘
             ▼
      ┌─────────────┐
      │   tracker   │  ← logs cost, latency, routing reason
      └─────────────┘
```

---

## Key Features

### 1. 5-Signal Query Evaluator
Not just keyword matching — a proper complexity scorer using token count, technical keywords, reasoning depth, code detection, and greeting penalties.

### 2. LRU Cache
Identical queries return instantly at **zero cost**. Cache stores up to 500 entries with automatic eviction of oldest entries.

### 3. Circuit Breaker (Fallback Handler)
If Gemma fails 3 times consecutively, it **automatically routes all traffic to Gemini**. Retries Gemma after 30 seconds. Zero downtime, 100% uptime.

### 4. Real-time FinOps Dashboard
Built into the Streamlit UI — tracks cost per request, total savings vs Gemini-only baseline, latency comparison, and routing decisions live.

### 5. Production REST API (FastAPI)
Full REST backend alongside the Streamlit UI:
- `POST /chat` — inference endpoint
- `GET /health` — GKE liveness probe
- `GET /metrics` — FinOps data
- `GET /circuit` — circuit breaker status
- `POST /cache/clear` — flush cache

### 6. GKE Deployment with Auto-scaling
Kubernetes manifests with Horizontal Pod Autoscaler — scales from 1 to 5 pods based on CPU load. Workload Identity for secure GCP access (no key files needed).

### 7. CI/CD Pipeline
GitHub Actions → Cloud Build → GKE auto-deploy on every push to `main`.

---

## Cost Impact

| Scenario | Cost per 1000 requests |
|---|---|
| Gemini-only (baseline) | $0.10 |
| Kifayati Hybrid (70% Gemma) | $0.037 |
| **Savings** | **~63%** |

*Real savings depend on query distribution. Simple query-heavy workloads can save up to 90%.*

---

## Project Structure

```
kifayati-ai/
├── agents/
│   ├── agent.py        # KifayatiRouter — main routing engine + LRU cache
│   ├── evaluator.py    # 5-signal complexity scorer
│   ├── fallback.py     # Circuit breaker pattern
│   ├── tracker.py      # FinOps metrics logger
│   ├── config.py       # Centralised config + env validation
│   └── __init__.py
├── k8s/
│   └── deployment.yaml # GKE Deployment + HPA + Service
├── tests/
│   └── test_routing.py # 16 unit tests (run without GCP)
├── .github/
│   └── workflows/
│       └── deploy.yml  # CI/CD pipeline
├── app.py              # Streamlit frontend
├── api.py              # FastAPI REST backend
├── Dockerfile          # Multi-stage production build
├── requirements.txt
└── .env.example
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Models | Gemma 3:4b (Vertex AI), Gemini 2.5 Flash |
| Frontend | Streamlit |
| Backend API | FastAPI + Uvicorn |
| Cloud | Google Cloud Platform |
| Compute | Google Kubernetes Engine (GKE) |
| ML Platform | Vertex AI Online Prediction |
| CI/CD | GitHub Actions + Cloud Build |
| Language | Python 3.11 |

---

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/geeta-gwalior/Reducing-AI-Costs-via-Hybrid-Multi-Model-Infrastructure.git
cd Reducing-AI-Costs-via-Hybrid-Multi-Model-Infrastructure

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:
```env
PROJECT_ID=your-gcp-project-id
LOCATION=europe-west4
GEMMA_ENDPOINT_ID=your-vertex-ai-endpoint-id   # leave blank to use Gemini fallback
```

### 3. Run Locally

```bash
# Streamlit UI
streamlit run app.py

# FastAPI (new terminal)
uvicorn api:app --reload --port 8080
```

- Streamlit → `http://localhost:8501`
- API Docs → `http://localhost:8080/docs`

### 4. Run Tests

```bash
pytest tests/ -v
# 16 passed
```

---

## GKE Deployment

```bash
# 1. Build & push Docker image
docker build -t gcr.io/YOUR_PROJECT_ID/kifayati:latest .
docker push gcr.io/YOUR_PROJECT_ID/kifayati:latest

# 2. Create GKE cluster
gcloud container clusters create kifayati-cluster \
  --zone=europe-west4-a --num-nodes=2

# 3. Deploy
kubectl apply -f k8s/deployment.yaml

# 4. Get external IP
kubectl get service kifayati-service -n kifayati
```

---

## Architecture Decision — Why Hybrid?

| Query Type | Example | Best Model | Why |
|---|---|---|---|
| Greeting | "Hi!", "Thanks" | Gemma 3:4b | No reasoning needed |
| Simple fact | "What is Python?" | Gemma 3:4b | Short, direct answer |
| Complex reasoning | "Explain transformer architecture" | Gemini 2.5 Flash | Deep knowledge needed |
| Code generation | "Write a FastAPI endpoint" | Gemini 2.5 Flash | Code accuracy critical |
| Cached query | Any repeated query | Cache (free) | Already answered |

---

## Developed By

**Geeta Kakrani**
Google Developer Expert (AI/ML)


