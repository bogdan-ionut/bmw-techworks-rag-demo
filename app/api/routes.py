# app/api/routes.py
from __future__ import annotations

import ast
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None

from app.rag.prompt import (
    ANSWER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    format_candidates_for_prompt,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Make the UI bulletproof even if the LLM fails/truncates
DEFAULT_FALLBACK_TOP_N = 5
# Keep completion budgets large enough to avoid truncation (e.g., long URLs in top_matches)
SAFE_MIN_LLM_MAX_TOKENS = 1200


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
    sources: Optional[List[Dict[str, Any]]] = None  # To re-use retrieved sources for a pure LLM call
    planner: bool = True  # Use LLM query planner


# -----------------------------
# Cache helpers
# -----------------------------
def _cache_get(
    cache: Dict[Tuple[Any, ...], Dict[str, Any]],
    key: Tuple[Any, ...],
    ttl_sec: int,
) -> Optional[Dict[str, Any]]:
    entry = cache.get(key)
    if not entry:
        return None
    if (time.time() - entry["t"]) > ttl_sec:
        cache.pop(key, None)
        return None
    return entry["v"]


def _cache_set(
    cache: Dict[Tuple[Any, ...], Dict[str, Any]],
    key: Tuple[Any, ...],
    value: Dict[str, Any],
) -> None:
    cache[key] = {"t": time.time(), "v": value}


# -----------------------------
# LLM usage helpers
# -----------------------------
def _extract_llm_usage(resp: Any) -> Tuple[int, int, int, Optional[str]]:
    meta = getattr(resp, "response_metadata", {}) or {}
    usage = meta.get("token_usage") or meta.get("usage_metadata") or {}

    # OpenAI style
    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    total_tokens = usage.get("total_tokens") or 0

    # Gemini 2.5/3 style
    prompt_tokens = usage.get("prompt_token_count") or prompt_tokens
    completion_tokens = usage.get("candidates_token_count") or completion_tokens
    total_tokens = usage.get("total_token_count") or total_tokens

    prompt_tokens = int(prompt_tokens or 0)
    completion_tokens = int(completion_tokens or 0)
    total_tokens = int(total_tokens or (prompt_tokens + completion_tokens))

    model_name = (
        meta.get("model_name")
        or meta.get("model")
        or usage.get("model")
        or usage.get("model_name")
    )

    return prompt_tokens, completion_tokens, total_tokens, model_name


def _resolve_llm_choice(settings, body: QueryBody) -> Tuple[str, str]:
    provider = (body.llm_provider or settings.llm_provider or "").strip().upper()
    if provider not in {"OPENAI", "GEMINI"}:
        raise HTTPException(status_code=400, detail="Invalid llm_provider. Use OPENAI or GEMINI.")

    if provider == "OPENAI":
        model = (body.llm_model or settings.openai_model or "").strip()
    else:
        model = (body.llm_model or settings.gemini_model or "").strip()

    if not model:
        raise HTTPException(status_code=400, detail="Missing llm_model for selected provider.")

    return provider, model


def _get_llm_client(request: Request, provider: str, model: str):
    settings = request.app.state.settings
    cache = getattr(request.app.state, "llm_cache", {}) or {}
    key = (provider, model, settings.temperature, settings.max_tokens)

    if key in cache:
        return cache[key]

    if provider == "GEMINI":
        if ChatGoogleGenerativeAI is None:
            raise HTTPException(status_code=400, detail="Gemini requested but not installed.")
        if not settings.google_api_key:
            raise HTTPException(status_code=400, detail="GOOGLE_API_KEY is required for Gemini.")

        llm = ChatGoogleGenerativeAI(
            api_key=settings.google_api_key,
            model=model,
            temperature=settings.temperature,
            response_mime_type="application/json",  # Gemini 2.5/3 reliably returns JSON
        )
    else:
        if not settings.openai_api_key:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY is required for OpenAI models.")
        try:
            llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model=model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                response_format={"type": "json_object"},
            )
        except TypeError:
            # Fallback for older client versions that do not support response_format
            llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model=model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )

    cache[key] = llm
    request.app.state.llm_cache = cache
    return llm


# -----------------------------
# JSON safety
# -----------------------------
def _coerce_llm_text(x: Any) -> str:
    """
    LangChain providers sometimes return content as:
    - str (OpenAI)
    - list[dict|str] (Gemini / multimodal parts)
    - dict (rare)
    Convert safely to a plain string.
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8", errors="ignore")
        except Exception:
            return str(x)

    # Gemini sometimes returns list of parts
    if isinstance(x, list):
        parts: List[str] = []
        for it in x:
            if it is None:
                continue
            if isinstance(it, str):
                parts.append(it)
                continue
            if isinstance(it, dict):
                if "text" in it:
                    parts.append(str(it.get("text") or ""))
                elif "content" in it:
                    parts.append(str(it.get("content") or ""))
                else:
                    parts.append(json.dumps(it, ensure_ascii=False))
                continue
            parts.append(str(it))
        return "\n".join([p for p in parts if p]).strip()

    if isinstance(x, dict):
        # Gemini 3 responses may include nested candidates/parts
        if "candidates" in x:
            parts: List[str] = []
            for cand in x.get("candidates") or []:
                content = cand.get("content") or {}
                parts.extend(
                    [p.get("text") or "" for p in content.get("parts", []) if isinstance(p, dict)]
                )
            flat = "\n".join([p for p in parts if p]).strip()
            if flat:
                return flat

        if "text" in x:
            return str(x.get("text") or "")

        return json.dumps(x, ensure_ascii=False)

    return str(x)


def _safe_json_loads(s: Any) -> Dict[str, Any]:
    """
    Tries to parse model output as JSON.
    If it contains extra text, tries to extract the outermost {...}.
    Accepts str/list/dict and coerces safely to string first.
    """
    s = _coerce_llm_text(s).strip()

    def _parse_obj(text: str) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(text)
        except Exception:
            try:
                obj = ast.literal_eval(text)
            except Exception:
                return None

        return obj if isinstance(obj, dict) else None

    # Strip common Markdown fences often returned by models
    fenced = re.search(r"```(?:json)?\s*(.*?)```", s, flags=re.DOTALL)
    if fenced:
        s = fenced.group(1).strip()

    obj = _parse_obj(s)
    if obj is not None:
        return obj

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        obj = _parse_obj(s[start : end + 1])
        if obj is not None:
            return obj
        raise ValueError("Extracted JSON is not an object.")

    raise ValueError("Could not parse JSON from model output.")


def _safe_stringify_for_log(val: Any, limit: int = 6000) -> str:
    """Convert arbitrary objects to a loggable string with truncation."""

    def default(o: Any) -> Any:
        if isinstance(o, (bytes, bytearray)):
            return o.decode("utf-8", errors="ignore")
        if hasattr(o, "__dict__"):
            try:
                return {k: v for k, v in vars(o).items() if not k.startswith("_")}
            except Exception:
                return str(o)
        return str(o)

    try:
        text = json.dumps(val, default=default, ensure_ascii=False)
    except Exception:
        try:
            text = str(val)
        except Exception:
            text = repr(val)

    if len(text) > limit:
        return text[:limit] + f"… (truncated {len(text)} chars)"
    return text


def _log_llm_raw_output(resp: Any, raw: str, provider: str, model: Optional[str]) -> None:
    """Log the LLM raw response content in a structured way for debugging."""

    try:
        content = getattr(resp, "content", None)
        snippet = raw if len(raw) <= 1500 else (raw[:1500] + "…")
        meta = getattr(resp, "response_metadata", None)

        response_debug = {
            "resp_type": type(resp).__name__,
            "content_type": type(content).__name__ if content is not None else None,
            "raw_len": len(raw),
            "content_preview": _safe_stringify_for_log(content, limit=3000),
            "response_metadata": meta,
            "resp_dict_keys": sorted(list(getattr(resp, "__dict__", {}).keys())),
        }
        response_dump = _safe_stringify_for_log(response_debug, limit=8000)

        logger.info(
            "LLM raw output: %s",
            snippet,
            extra={
                "llm_provider": provider,
                "llm_model": model,
                "llm_raw_snippet": snippet,
                "llm_content_type": type(content or resp).__name__,
                "llm_response_metadata": meta,
                "llm_response_dump": response_dump,
            },
        )
    except Exception:
        logger.debug("Failed to log LLM raw output", exc_info=True)


# -----------------------------
# Query planner (LLM-based filter extraction)
# -----------------------------
def _run_query_planner(
    request: Request,
    query: str,
    provider: str,
    model: str,
) -> Dict[str, Any]:
    """
    Uses an LLM to decompose a natural language query into:
    - a structured metadata filter
    - a clean semantic query for vector search
    """
    llm = _get_llm_client(request, provider, model)
    try:
        resp = llm.invoke(
            [
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                HumanMessage(content=f"User query: \"\"\"{query}\"\"\""),
            ]
        )
        raw = _coerce_llm_text(getattr(resp, "content", resp) or "")
        _log_llm_raw_output(resp, raw, provider, model)
        parsed = _safe_json_loads(raw)

        logger.info(
            "Query planner ran successfully",
            extra={
                "query": query,
                "llm_provider": provider,
                "llm_model": model,
                "planner_output": parsed,
            },
        )
        return parsed

    except Exception as e:
        logger.exception(
            "Query planner failed",
            extra={"query": query, "llm_provider": provider, "llm_model": model, "error": str(e)},
        )
        # Fallback to pure semantic search on failure
        return {"semantic_query": query, "filter": None}


# -----------------------------
# Utils
# -----------------------------
def _normalize_linkedin(url: Optional[str]) -> str:
    if not url:
        return ""
    clean = str(url).strip().split("?")[0].split("#")[0].rstrip("/")
    try:
        parsed = urlparse(clean)
        host = (parsed.netloc or "").lower()
        if host.endswith("linkedin.com"):
            host = "linkedin.com"
        path = parsed.path or ""
        if path.startswith("/in/"):
            slug = path[len("/in/") :].rstrip("/").lower()
        else:
            slug = path.lstrip("/").rstrip("/").lower()
        return f"https://{host}/in/{slug}" if slug else ""
    except Exception:
        return clean.lower()


def _source_id(s: Dict[str, Any]) -> str:
    return (
        str(s.get("vmid") or "").strip()
        or _normalize_linkedin(s.get("profile_url"))
        or str(s.get("id") or "").strip()
        or str(s.get("full_name") or "").strip().lower()
    )


def _looks_like_short_search(query: str) -> bool:
    q = query.strip()
    return len(q) <= 22 and "?" not in q


def _source_to_text(s: Dict[str, Any]) -> str:
    # Prefer embedding_text if present (best semantic)
    emb = (s.get("embedding_text") or "").strip()
    if emb:
        return emb
    parts = [
        f"Employee: {s.get('full_name') or 'N/A'}",
        f"Headline: {s.get('headline') or 'N/A'}",
        f"Current role: {s.get('job_title') or 'N/A'} ({s.get('job_date_range') or ''})",
        f"Previous role: {s.get('job_title_2') or 'N/A'} ({s.get('job_date_range_2') or ''})",
        f"Location: {s.get('location') or 'N/A'}",
        f"Education 1: {s.get('school') or ''} – {s.get('school_degree') or ''} – {s.get('school_date_range') or ''}",
        f"Education 2: {s.get('school_2') or ''} – {s.get('school_degree_2') or ''} – {s.get('school_date_range_2') or ''}",
        f"LinkedIn: {s.get('profile_url') or ''}",
        f"VMID: {s.get('vmid') or ''}",
        f"Eyewear present: {s.get('eyewear_present')}",
        f"Beard present: {s.get('beard_present')}",
    ]
    return "\n".join([p for p in parts if p.strip()]).strip()


def _build_top_matches_from_sources(
    sources: List[Dict[str, Any]],
    top_n: int,
    *,
    why: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s0 in (sources or [])[: max(0, top_n)]:
        out.append(
            {
                "full_name": s0.get("full_name") or "",
                "profile_url": s0.get("profile_url") or "",
                "image_url": s0.get("profile_image_url") or None,
                "score": s0.get("score"),
                "why_match": why,
            }
        )
    return out


def _fallback_answer_obj(
    query: str,
    flt: Optional[Dict[str, Any]],
    sources: List[Dict[str, Any]],
    *,
    top_n: int,
    reason: str,
    raw: Optional[str] = None,
) -> Dict[str, Any]:
    fallback_top = (sources or [])[: max(0, top_n)]
    names = [s.get("full_name") or "N/A" for s in fallback_top]
    base = f"Top matches for '{query}' ({reason}): " + (", ".join(names) if names else "No matches.")
    if raw:
        # keep it short (UI-friendly)
        raw_snip = _coerce_llm_text(raw).strip().replace("\n", " ")
        if len(raw_snip) > 240:
            raw_snip = raw_snip[:240] + "..."
        base = base + f"\n\n(LLM output was invalid/truncated.)"

    return {
        "answer": base,
        "filters_applied": flt,
        "top_matches": _build_top_matches_from_sources(
            sources,
            top_n,
            why="Retrieved match (LLM output invalid/truncated).",
        ),
    }


def _answer_from_unstructured_raw(
    raw: Any,
    query: str,
    flt: Optional[Dict[str, Any]],
    sources: List[Dict[str, Any]],
    *,
    top_n: int,
) -> Optional[Dict[str, Any]]:
    """
    Build a best-effort answer object when the LLM returns plain text instead of JSON.

    This prevents the UI from surfacing a scary "invalid/unstructured" message when
    we still have a meaningful answer string we can show to the user.
    """

    text = _coerce_llm_text(raw).strip()
    if not text:
        return None

    return {
        "answer": text,
        "filters_applied": flt,
        "top_matches": _build_top_matches_from_sources(
            sources,
            top_n,
            why="Retrieved match (LLM output unstructured).",
        ),
    }


def _coerce_answer_obj(
    query: str,
    flt: Optional[Dict[str, Any]],
    sources: List[Dict[str, Any]],
    answer_obj: Any,
    *,
    top_n: int,
) -> Dict[str, Any]:
    """
    Ensure the final answer object always matches the expected schema and
    top_matches is never empty (unless there are no sources).
    """
    if not isinstance(answer_obj, dict):
        return _fallback_answer_obj(query, flt, sources, top_n=top_n, reason="fallback")

    # Ensure required keys exist
    # If the model returned "summary" instead of "answer", rename it for consistency
    if "summary" in answer_obj and "answer" not in answer_obj:
        answer_obj["answer"] = answer_obj.pop("summary")

    if "answer" not in answer_obj or not str(answer_obj.get("answer") or "").strip():
        fallback = _fallback_answer_obj(
            query=query,
            flt=flt,
            sources=sources,
            top_n=top_n,
            reason="LLM returned an empty answer",
        )
        answer_obj["answer"] = fallback["answer"]
        answer_obj.setdefault("top_matches", fallback.get("top_matches", []))
    if "filters_applied" not in answer_obj:
        answer_obj["filters_applied"] = flt
    if "top_matches" not in answer_obj or not isinstance(answer_obj.get("top_matches"), list):
        answer_obj["top_matches"] = []

    # Cap top_matches
    answer_obj["top_matches"] = answer_obj["top_matches"][: max(0, top_n)]

    # If empty, fill with fallback from sources
    if not answer_obj["top_matches"] and (sources or []):
        answer_obj["top_matches"] = _build_top_matches_from_sources(
            sources,
            top_n,
            why="Retrieved match (LLM did not return structured matches).",
        )

    return answer_obj


# -----------------------------
# Pinecone search (semantic) with optional metadata filter
# -----------------------------
def _pinecone_query(
    request: Request,
    query: str,
    top_k: int,
    flt: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    embeddings = request.app.state.embeddings
    index = request.app.state.pinecone_index
    namespace = request.app.state.settings.pinecone_namespace

    vec = embeddings.embed_query(query)
    res = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
        filter=flt,
    )

    matches = res.get("matches") if isinstance(res, dict) else getattr(res, "matches", []) or []
    sources: List[Dict[str, Any]] = []

    for m in matches:
        if isinstance(m, dict):
            md = m.get("metadata") or {}
            mid = m.get("id")
            score = m.get("score")
        else:
            md = getattr(m, "metadata", {}) or {}
            mid = getattr(m, "id", None)
            score = getattr(m, "score", None)

        if mid and not md.get("id"):
            md["id"] = mid

        # IMPORTANT: ingestion uses profile_image_s3_url, not profile_image_url
        image_url = (
            md.get("profile_image_s3_url")
            or md.get("profile_image_url")
            or md.get("image_url")
            or None
        )

        sources.append(
            {
                "id": md.get("id"),
                "score": score,
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
                # best context (already built in ingestion)
                "embedding_text": md.get("embedding_text") or "",
            }
        )

    return sources, len(matches)


# -----------------------------
# Cohere rerank (top_k -> top_n), reorder cards
# -----------------------------
def _cohere_rerank_sources(request: Request, query: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    reranker = request.app.state.cohere_reranker
    if not reranker or not sources:
        return sources
    docs = [Document(page_content=_source_to_text(s), metadata=s) for s in sources]
    reranked_docs = reranker.compress_documents(documents=docs, query=query)  # already top_n
    top_sources = [d.metadata for d in reranked_docs]
    top_ids = {_source_id(s) for s in top_sources}
    rest = [s for s in sources if _source_id(s) and _source_id(s) not in top_ids]
    return top_sources + rest


# -----------------------------
# Routes
# -----------------------------
@router.get("/health", include_in_schema=False)
def health() -> Dict[str, Any]:
    return {"ok": True}


def _merge_filters(
    a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if not a and not b:
        return None
    if a and not b:
        return a
    if b and not a:
        return b
    return {"$and": [a, b]}


@router.get("/search")
def search_endpoint(
    request: Request,
    q: str = Query(..., min_length=1),
    k: int = Query(24, ge=1, le=32),
    filters: Optional[str] = Query(None, description="Optional JSON string Pinecone filter"),
    planner: bool = Query(True, description="Use LLM query planner"),
) -> Dict[str, Any]:
    s = request.app.state.settings
    cache = request.app.state.cache
    ttl = int(s.cache_ttl_sec)
    selected_provider, selected_model = _resolve_llm_choice(s, QueryBody(query=q))

    query = q.strip()
    start = time.time()

    explicit_filter = None
    if filters:
        try:
            explicit_filter = json.loads(filters)
        except Exception:
            explicit_filter = None

    if planner:
        planned = _run_query_planner(request, query, selected_provider, selected_model)
        semantic_query = planned.get("semantic_query") or query
        inferred_filter = planned.get("filter")
    else:
        semantic_query = query
        inferred_filter = None

    flt = _merge_filters(explicit_filter, inferred_filter)

    cache_key = (
        "search",
        query.lower(),
        int(k),
        s.embed_model,
        s.pinecone_index_name,
        json.dumps(flt, sort_keys=True) if flt else None,
    )
    cached = _cache_get(cache, cache_key, ttl)
    if cached:
        return cached

    t0 = time.time()
    sources, retrieved_docs = _pinecone_query(request, semantic_query, top_k=int(k), flt=flt)
    t1 = time.time()

    rerank_sec = 0.0
    if bool(s.rerank_in_search):
        rerank_start = time.time()
        sources = _cohere_rerank_sources(request, query, sources)
        rerank_end = time.time()
        rerank_sec = round(rerank_end - rerank_start, 2)

    result = {
        "sources": sources,
        "retrieved_docs": retrieved_docs,
        "unique_sources": len(sources),
        "k": int(k),
        "filters_applied": flt,
        "latency_sec": round(time.time() - start, 2),
        "retrieval_sec": round(t1 - t0, 2),
        "rerank_sec": rerank_sec,
    }
    _cache_set(cache, cache_key, result)
    return result


@router.post("/query")
def rag_query(request: Request, body: QueryBody) -> Dict[str, Any]:
    s = request.app.state.settings
    cache = request.app.state.cache
    ttl = int(s.cache_ttl_sec)

    query = body.query.strip()
    with_llm = bool(body.with_llm)
    selected_provider, selected_model = _resolve_llm_choice(s, body)

    k = int(body.k or s.search_top_k)
    k = max(1, min(32, k))

    start = time.time()

    if body.planner:
        planned = _run_query_planner(request, query, selected_provider, selected_model)
        semantic_query = planned.get("semantic_query") or query
        inferred_filter = planned.get("filter")
    else:
        semantic_query = query
        inferred_filter = None

    flt = _merge_filters(body.filters, inferred_filter)

    # 1) retrieve + rerank (if sources are not passed in)
    if body.sources and len(body.sources) > 0:
        sources = body.sources
        retrieved_docs = len(sources)
        t0 = t1 = rerank_start = rerank_end = time.time()
        rerank_sec = 0.0
    else:
        t0 = time.time()
        sources, retrieved_docs = _pinecone_query(request, semantic_query, top_k=k, flt=flt)
        t1 = time.time()

        rerank_start = time.time()
        sources = _cohere_rerank_sources(request, semantic_query, sources)
        rerank_end = time.time()
        rerank_sec = round(rerank_end - rerank_start, 2)

    # Effective top_matches count for UI/answer payload
    top_n_cfg = int(getattr(s, "context_max_people", DEFAULT_FALLBACK_TOP_N) or DEFAULT_FALLBACK_TOP_N)
    top_n = max(0, min(top_n_cfg, len(sources or [])))

    # Build candidates for the answer prompt
    candidates: List[Dict[str, Any]] = []
    for s0 in sources:
        candidates.append(
            {
                "id": s0.get("id"),
                "score": s0.get("score"),
                "text": _source_to_text(s0),
                "metadata": {
                    "full_name": s0.get("full_name"),
                    "profile_url": s0.get("profile_url"),
                    "profile_image_s3_url": s0.get("profile_image_url"),  # prompt expects this key
                },
            }
        )

    # 3) skip LLM (optional) -> still return useful AI Insights payload
    if not with_llm or (s.skip_llm_for_short and _looks_like_short_search(query)):
        answer_obj = _fallback_answer_obj(
            query=query,
            flt=flt,
            sources=sources,
            top_n=top_n,
            reason="LLM skipped",
        )
        return {
            "answer": answer_obj,
            "answer_text": str(answer_obj.get("answer") or ""),
            "sources": sources,
            "filters_applied": flt,
            "llm_provider": selected_provider,
            "llm_used": selected_provider,
            "llm_model": selected_model or "SKIPPED",
            "llm_fallback_reason": None,
            "retrieved_docs": retrieved_docs,
            "unique_sources": len(sources),
            "k": k,
            "latency_sec": round(time.time() - start, 2),
            "retrieval_sec": round(t1 - t0, 2),
            "pinecone_sec": round(t1 - t0, 2),
            "rerank_sec": rerank_sec,
            "llm_sec": 0.0,
            "llm_prompt_tokens": 0,
            "llm_completion_tokens": 0,
            "llm_total_tokens": 0,
        }

    # 4) cache LLM answer (include filters)
    cache_key = (
        "rag",
        query.lower(),
        int(k),
        json.dumps(flt, sort_keys=True) if flt else None,
        selected_provider,
        selected_model,
        s.temperature,
        s.max_tokens,
        s.context_max_people,
        s.cohere_rerank_top_n,
    )
    cached = _cache_get(cache, cache_key, ttl)
    if cached:
        cached = dict(cached)
        cached["sources"] = sources
        cached["retrieved_docs"] = retrieved_docs
        cached["unique_sources"] = len(sources)
        cached["k"] = k
        cached["filters_applied"] = flt
        cached["retrieval_sec"] = round(t1 - t0, 2)
        cached["pinecone_sec"] = cached.get("pinecone_sec", round(t1 - t0, 2))
        cached["rerank_sec"] = rerank_sec
        cached["latency_sec"] = round(time.time() - start, 2)
        cached.setdefault("llm_sec", 0.0)
        cached.setdefault("llm_prompt_tokens", 0)
        cached.setdefault("llm_completion_tokens", 0)
        cached.setdefault("llm_total_tokens", 0)
        cached.setdefault("llm_provider", selected_provider)
        cached.setdefault("llm_model", selected_model)
        cached.setdefault("llm_fallback_reason", None)
        return cached

    # 5) LLM -> JSON answer
    llm = _get_llm_client(request, selected_provider, selected_model)
    t2 = time.time()

    user_msg = {
        "query": query,
        "filters_applied": flt,
        "candidates_text": format_candidates_for_prompt(candidates),
    }

    # Make output robust:
    # - enforce JSON output (OpenAI JSON mode when available)
    # - avoid truncation by using a safer max_tokens floor
    effective_max_tokens = max(int(getattr(s, "max_tokens", 0) or 0), SAFE_MIN_LLM_MAX_TOKENS)

    llm_call = llm
    fallback_reason: Optional[str] = None
    if selected_provider == "GEMINI":
        try:
            # Gemini uses max_output_tokens and rejects unknown fields
            llm_call = llm_call.bind(max_output_tokens=effective_max_tokens)
        except Exception:
            pass
        # Do NOT pass response_format to Gemini (raises validation errors)
    else:
        try:
            llm_call = llm_call.bind(max_tokens=effective_max_tokens)
        except Exception:
            pass

        try:
            # Works for OpenAI ChatCompletions JSON mode in langchain (ignored by non-OpenAI providers)
            llm_call = llm_call.bind(response_format={"type": "json_object"})
        except Exception:
            pass

    tight_system = (
        ANSWER_SYSTEM_PROMPT
        + "\n\nHard constraints:\n"
        + f"- Return at most {max(0, top_n)} top_matches.\n"
        + "- Keep 'answer' <= 1200 chars.\n"
        + "- Keep each 'why_match' <= 180 chars.\n"
        + "- Return JSON ONLY (no Markdown fences).\n"
    )

    try:
        resp = llm_call.invoke(
            [
                SystemMessage(content=tight_system),
                HumanMessage(content=json.dumps(user_msg, ensure_ascii=False)),
            ]
        )
    except Exception as e:
        logger.exception(
            "LLM invoke failed; using fallback",
            extra={"query": query, "llm_provider": selected_provider, "llm_model": selected_model, "llm_error": str(e)},
        )
        fallback_reason = "llm_invoke_failed"
        answer_obj = _fallback_answer_obj(
            query=query,
            flt=flt,
            sources=sources,
            top_n=top_n,
            reason=f"LLM invoke failed ({selected_provider})",
            raw=None,
        )
        return {
            "answer": answer_obj,
            "answer_text": str(answer_obj.get("answer") or ""),
            "sources": sources,
            "filters_applied": flt,
            "llm_provider": selected_provider,
            "llm_used": selected_provider,
            "llm_model": selected_model,
            "llm_fallback_reason": fallback_reason,
            "retrieved_docs": retrieved_docs,
            "unique_sources": len(sources),
            "k": k,
            "latency_sec": round(time.time() - start, 2),
            "retrieval_sec": round(t1 - t0, 2),
            "pinecone_sec": round(t1 - t0, 2),
            "rerank_sec": rerank_sec,
            "llm_sec": 0.0,
            "llm_prompt_tokens": 0,
            "llm_completion_tokens": 0,
            "llm_total_tokens": 0,
        }

    t3 = time.time()

    prompt_tokens, completion_tokens, total_tokens, usage_model = _extract_llm_usage(resp)
    llm_model_used = usage_model or selected_model or selected_provider

    raw = _coerce_llm_text(getattr(resp, "content", resp) or "")
    _log_llm_raw_output(resp, raw, selected_provider, selected_model)

    try:
        parsed = _safe_json_loads(raw)
        answer_obj = _coerce_answer_obj(query, flt, sources, parsed, top_n=top_n)
        logger.info("LLM parsing successful", extra={"query": query})
    except Exception as e:
        logger.exception(
            "LLM failed; attempting to salvage unstructured output",
            extra={
                "query": query,
                "llm_error": str(e),
                "llm_provider": selected_provider,
                "llm_model": selected_model,
                "llm_raw_snippet": raw[:1500],
            },
        )

        answer_obj = _answer_from_unstructured_raw(
            raw,
            query=query,
            flt=flt,
            sources=sources,
            top_n=top_n,
        )
        if answer_obj is not None:
            fallback_reason = "unstructured_output"
            logger.info("LLM output unstructured; surfaced raw text", extra={"query": query})
        else:
            fallback_reason = "llm_fallback"
            answer_obj = _fallback_answer_obj(
                query=query,
                flt=flt,
                sources=sources,
                top_n=top_n,
                reason="LLM fallback",
                raw=raw,
            )

    result = {
        "answer": answer_obj,
        "answer_text": str(answer_obj.get("answer") or ""),
        "sources": sources,
        "filters_applied": flt,
        "llm_provider": selected_provider,
        "llm_used": selected_provider,
        "llm_model": llm_model_used,
        "llm_fallback_reason": fallback_reason,
        "retrieved_docs": retrieved_docs,
        "unique_sources": len(sources),
        "k": k,
        "latency_sec": round(t3 - start, 2),
        "retrieval_sec": round(t1 - t0, 2),
        "pinecone_sec": round(t1 - t0, 2),
        "rerank_sec": rerank_sec,
        "llm_sec": round(t3 - t2, 2),
        "llm_prompt_tokens": prompt_tokens,
        "llm_completion_tokens": completion_tokens,
        "llm_total_tokens": total_tokens,
    }
    _cache_set(cache, cache_key, result)
    return result


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
                {"name": "GPT 5 Mini", "value": "gpt-5-mini-2025-08-07"},
                {"name": "GPT 5 Nano", "value": "gpt-5-nano-2025-08-07"},
                {"name": "GPT 4o Mini", "value": "gpt-4o-mini-2024-07-18"},
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
