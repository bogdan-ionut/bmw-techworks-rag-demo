from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from app.rag.service import rag_search_async
from app.rag.prompt import CITY_NORMALIZATION_MAP

router = APIRouter()
logger = logging.getLogger(__name__)

# -----------------------------
# DTO
# -----------------------------
class QueryBody(BaseModel):
    query: str = Field(..., min_length=1)
    with_llm: bool = True
    k: Optional[int] = None  # retrieval top_k override
    filters: Optional[Dict[str, Any]] = None  # optional explicit pinecone filter
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None  # To re-use retrieved sources (not used in new async flow yet)
    planner: bool = True  # Use LLM query planner


class BenchmarkPayload(BaseModel):
    queries: List[str]
    iterations: int = 1


# -----------------------------
# Helpers
# -----------------------------
def _format_source(cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map rag_search_async candidate (metadata) to API source format.
    """
    md = cand.get("metadata") or {}

    # Handle image url mapping
    image_url = (
        md.get("profile_image_s3_url")
        or md.get("profile_image_url")
        or md.get("image_url")
        or None
    )

    return {
        "id": md.get("id") or cand.get("id"),
        "score": cand.get("score"),
        "vmid": md.get("vmid"),
        "full_name": md.get("full_name"),
        "profile_url": md.get("profile_url"),
        "profile_image_url": image_url,
        "headline": md.get("headline"),
        "location": md.get("location"),
        "job_title": md.get("job_title"),
        "job_date_range": md.get("job_date_range"),
        "job_title_2": md.get("job_title_2"),
        "job_date_range_2": md.get("job_date_range_2"),
        "school": md.get("school"),
        "school_degree": md.get("school_degree"),
        "school_date_range": md.get("school_date_range"),
        "school_2": md.get("school_2"),
        "school_degree_2": md.get("school_degree_2"),
        "school_date_range_2": md.get("school_date_range_2"),
        # filters
        "eyewear_present": md.get("eyewear_present"),
        "beard_present": md.get("beard_present"),
        # new fields for LLM
        "company": md.get("company"),
        "tech_tokens": md.get("tech_tokens"),
        "minimum_estimated_years_of_exp": md.get("minimum_estimated_years_of_exp"),
        # best context
        "embedding_text": md.get("embedding_text") or "",
    }


def _hydrate_answer_matches(answer: Dict[str, Any], sources_map: Dict[str, Any]) -> None:
    """
    Fill in missing details in answer['top_matches'] using the full source data.
    """
    top_matches = answer.get("top_matches") or []

    for m in top_matches:
        mid = m.get("id")
        if mid and mid in sources_map:
            s = sources_map[mid]
            if not m.get("full_name"):
                m["full_name"] = s.get("full_name") or ""
            if not m.get("profile_url"):
                m["profile_url"] = s.get("profile_url") or ""
            if not m.get("image_url"):
                m["image_url"] = s.get("profile_image_url") or None

            # Inject reasoning back into source
            if m.get("why_match"):
                s["reasoning"] = m.get("why_match")


# -----------------------------
# Routes
# -----------------------------
@router.get("/health", include_in_schema=False)
def health() -> Dict[str, Any]:
    return {"ok": True}


@router.get("/search")
async def search_endpoint(
    request: Request,
    q: str = Query(..., min_length=1),
    k: int = Query(24, ge=1, le=100),
    filters: Optional[str] = Query(None, description="Optional JSON string Pinecone filter"),
    planner: bool = Query(True, description="Use LLM query planner"),
) -> Dict[str, Any]:
    """
    Async search endpoint using Two-Stage Retrieval via rag_search_async.
    Stage 1: Retrieve 60 (default in service)
    Stage 2: Rerank 15 (default in service)
    """
    explicit_filter = None
    if filters:
        try:
            explicit_filter = json.loads(filters)
        except Exception:
            explicit_filter = None

    # Call async service
    # Note: k from query is treated as retrieval target if explicit,
    # but rag_search_async defaults to 60 for retrieval.
    # If the user asks for k=100, we should probably respect it for top_k.
    # But usually k in UI means "how many results to show".
    # For now, we'll map k to top_k if it's larger than default 60, otherwise stick to 60?
    # Actually, the task says "Stage 1: top_k=60". So we should enforce that default logic
    # unless specifically overridden logic is needed.
    # We will pass k as top_k ONLY if it is larger than 60, otherwise let service use 60.
    # However, k is often used for pagination size or result limit.
    # rag_search_async returns 'results' (reranked list).

    search_top_k = max(60, k)

    result = await rag_search_async(
        user_query=q,
        explicit_filter=explicit_filter,
        top_k=search_top_k,
        rerank_top_n=15, # Fixed as per instructions
        use_planner=planner,
        with_llm=False
    )

    # Format sources
    candidates = result.get("results", [])
    sources = [_format_source(c) for c in candidates]

    # Ensure we return the requested number of items if available (although we reranked 15)
    # If k > 15, we might want to append non-reranked items?
    # The instructions for Two-Stage Retrieval say:
    # "Stage 2 (Cohere): Setează top_n (rerank) la 10-15."
    # This implies we only return 15 high quality results.
    # The previous implementation appended the rest.
    # If the UI expects k=24, returning 15 might be fine if quality is higher.
    # We will return what rag_search_async returns.

    return {
        "sources": sources,
        "retrieved_docs": len(candidates), # This might be total retrieved or after rerank? Service returns reranked list in 'results'.
        "unique_sources": len(sources),
        "k": k,
        "filters_applied": result.get("filters_applied"),
        "latency_sec": result.get("latency_sec"),
        "retrieval_sec": result.get("retrieval_sec"),
        "rerank_sec": result.get("rerank_sec"),
    }


@router.post("/query")
async def rag_query(request: Request, body: QueryBody) -> Dict[str, Any]:
    """
    Async RAG endpoint.
    Stage 3: LLM Generation (Top 8).
    """
    # Defaults from instruction
    retrieval_top_k = 60
    rerank_top_n = 15
    llm_top_n = 5

    if body.k and body.k > 60:
        retrieval_top_k = body.k

    result = await rag_search_async(
        user_query=body.query,
        explicit_filter=body.filters,
        top_k=retrieval_top_k,
        rerank_top_n=rerank_top_n,
        llm_top_n=llm_top_n,
        use_planner=body.planner,
        with_llm=body.with_llm
    )

    # Format sources
    candidates = result.get("results", [])
    sources = [_format_source(c) for c in candidates]

    # Process answer
    answer_obj = result.get("answer") or {}

    # Hydrate top_matches in answer with full source details
    sources_map = {s["id"]: s for s in sources}
    _hydrate_answer_matches(answer_obj, sources_map)

    return {
        "answer": answer_obj,
        "answer_text": str(answer_obj.get("answer") or ""),
        "sources": sources,
        "filters_applied": result.get("filters_applied"),
        # Metadata
        "llm_provider": body.llm_provider or "default",
        "llm_model": body.llm_model or "default",
        "retrieved_docs": len(candidates), # Note: this is the reranked count
        "unique_sources": len(sources),
        "k": body.k or retrieval_top_k,
        "latency_sec": result.get("latency_sec"),
        "retrieval_sec": result.get("retrieval_sec"),
        "rerank_sec": result.get("rerank_sec"),
        "llm_sec": result.get("llm_sec"),
    }


@router.post("/benchmark")
async def benchmark_endpoint(payload: BenchmarkPayload) -> Dict[str, Any]:
    """
    Temporary benchmarking endpoint to validate performance and classification logic.
    """
    report = []

    for query in payload.queries:
        query_metrics = {
            "query": query,
            "runs": []
        }

        for _ in range(payload.iterations):
            # Using default parameters for benchmarking as per instructions
            # rag_search_async defaults: top_k=60, rerank_top_n=15, llm_top_n=8, use_planner=True, with_llm=True
            result = await rag_search_async(user_query=query)

            run_data = {
                "classification": result.get("query_type", "UNKNOWN"),
                "total_latency": result.get("latency_sec", 0.0),
                "breakdown": {
                    "retrieval_sec": result.get("retrieval_sec", 0.0),
                    "rerank_sec": result.get("rerank_sec", 0.0),
                    "llm_sec": result.get("llm_sec", 0.0)
                },
                "results_count": len(result.get("results", []))
            }
            query_metrics["runs"].append(run_data)

        # Aggregate logic could be here if we want average, but list of runs is detailed enough
        # We can just return the last classification (should be same) and average latencies if needed.
        # But the instructions say "Returnează un JSON agregat care să arate clar diferența".
        # Let's compute average for the query.

        avg_latency = sum(r["total_latency"] for r in query_metrics["runs"]) / payload.iterations
        classification = query_metrics["runs"][0]["classification"] if query_metrics["runs"] else "UNKNOWN"

        report.append({
            "query": query,
            "classification": classification,
            "avg_latency": avg_latency,
            "runs": query_metrics["runs"]
        })

    return {"benchmark_results": report}


@router.get("/meta", include_in_schema=False)
def meta(request: Request) -> Dict[str, Any]:
    s = request.app.state.settings
    return {
        "pinecone_index": s.pinecone_index_name,
        "pinecone_namespace": s.pinecone_namespace,
        "embed_model": s.embed_model,
        "llm_provider": s.llm_provider,
        "openai_model": s.openai_model,
        "gemini_model": s.gemini_model,
        "available_llms": {
            "OPENAI": [
                {"name": "GPT-4o mini", "value": "gpt-4o-mini-2024-07-18"},
                {"name": "GPT-5 mini", "value": "gpt-5-mini-2025-08-07"},
                {"name": "GPT-5 nano", "value": "gpt-5-nano-2025-08-07"},
            ],
            "GEMINI": [
                {"name": "Gemini 3 Flash", "value": "gemini-3-flash-preview"},
                {"name": "Gemini 2.5 Flash", "value": "gemini-2.5-flash"},
            ],
        },
        "cohere_rerank_top_n": s.cohere_rerank_top_n,
        "cache_ttl_sec": s.cache_ttl_sec,
        "frontend": "app/web/index.html",
        "language": "en",
    }
