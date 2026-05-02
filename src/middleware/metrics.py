"""
TIER 3: Prometheus Metrics
============================
Production equivalent:
  LlmMetricsService     (Micrometer Counters/Histograms)
  + CoreSpanAttribute   (OpenTelemetry span attributes)
  + TokenMetricsService (token usage counters)

METRICS EXPOSED AT /metrics (Prometheus scrape endpoint):
  rag_requests_total{endpoint, method, status}  ← request count
  rag_request_duration_seconds{endpoint}        ← latency histogram
  rag_active_requests                           ← in-flight gauge
  rag_agent_selections_total{agent, match_type} ← routing decisions
  rag_facts_retrieved_count{agent}              ← RAG retrieval histogram
  rag_token_usage{component}                    ← tokens per prompt component

PRODUCTION PARALLEL:
  LlmMetricsService.recordContentFilterViolation()
    → Counter("tech_assistant_content_filter_total")
  TokenMetricsService.recordTokenUsage()
    → records per-component token counts
  Both use io.micrometer MeterRegistry. We use prometheus_client directly.

HOW TO VIEW:
  curl http://localhost:8000/metrics
  Or point a Prometheus server at http://localhost:8000/metrics
"""

import time
from fastapi import FastAPI, Request, Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# ── Metric definitions ────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total HTTP requests handled",
    ["endpoint", "method", "status"],
)

REQUEST_LATENCY = Histogram(
    "rag_request_duration_seconds",
    "HTTP request duration in seconds",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

ACTIVE_REQUESTS = Gauge(
    "rag_active_requests",
    "Number of requests currently being processed",
)

AGENT_SELECTIONS = Counter(
    "rag_agent_selections_total",
    "Agent selection events by agent name and match strategy",
    ["agent", "match_type"],
)

FACTS_RETRIEVED = Histogram(
    "rag_facts_retrieved_count",
    "Number of RAG facts retrieved per request",
    ["agent"],
    buckets=[0, 1, 2, 3, 5, 10],
)

TOKEN_USAGE = Histogram(
    "rag_token_usage",
    "Approximate token count per prompt component",
    ["component"],
    buckets=[100, 250, 500, 1000, 2000, 4000, 8000],
)

LLM_ERRORS = Counter(
    "rag_llm_errors_total",
    "LLM call errors by type",
    ["error_type"],
)


# ── Middleware + endpoint registration ────────────────────────────────────────


def register_metrics(app: FastAPI) -> None:
    """
    Attach the request-tracking middleware and /metrics endpoint to the app.
    Call this in main.py before the server starts.
    """

    @app.middleware("http")
    async def track_requests(request: Request, call_next):
        ACTIVE_REQUESTS.inc()
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            raise
        finally:
            elapsed = time.perf_counter() - start
            endpoint = request.url.path
            REQUEST_COUNT.labels(
                endpoint=endpoint,
                method=request.method,
                status=str(status_code),
            ).inc()
            REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
            ACTIVE_REQUESTS.dec()

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint():
        """Prometheus scrape endpoint. Not protected by API key (standard practice)."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
