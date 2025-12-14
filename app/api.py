# app/api.py

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from dotenv import load_dotenv
from fastapi import FastAPI, Query as FastQuery
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# -----------------------------
# Paths / Static UI
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="BMW TechWorks RAG Demo")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_homepage():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Static frontend not found. Create app/static/index.html first."}


@app.get("/health", include_in_schema=False)
def health():
    return {"ok": True}


# -----------------------------
# Secrets: AWS Secrets Manager + .env fallback
# -----------------------------
def load_secrets() -> Dict[str, str]:
    secret_name = os.getenv("AWS_SECRET_NAME")
    region = os.getenv("AWS_REGION", "eu-north-1")

    if not secret_name:
        print("[INFO] AWS_SECRET_NAME not set – using only .env for API keys.")
        return {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
        }

    try:
        session = boto3.session.Session(region_name=region)
        client = session.client("secretsmanager")
        resp = client.get_secret_value(SecretId=secret_name)
        secret_str = resp.get("SecretString") or "{}"
        secrets = json.loads(secret_str)
        print(f"[INFO] Loaded secrets from AWS Secrets Manager: {secret_name}")
    except (NoCredentialsError, BotoCoreError, ClientError) as e:
        print(f"[WARN] Cannot read AWS secret ({e}). Falling back to .env")
        secrets = {}

    return {
        "OPENAI_API_KEY": secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
        "ANTHROPIC_API_KEY": secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY", ""),
        "GOOGLE_API_KEY": secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY", ""),
    }


secrets = load_secrets()

# -----------------------------
# Config
# -----------------------------
RAG_K_DEFAULT = int(os.getenv("RAG_K", "8"))                 # used for LLM analysis
SEARCH_K_DEFAULT = int(os.getenv("SEARCH_K_DEFAULT", "24"))  # used for cards
CONTEXT_MAX_PEOPLE = int(os.getenv("CONTEXT_MAX_PEOPLE", "8"))
SKIP_LLM_FOR_SHORT = os.getenv("SKIP_LLM_FOR_SHORT", "0") == "1"

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "220"))

# Cache (nice for demo)
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "600"))
_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}


def _cache_get(key: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
    entry = _CACHE.get(key)
    if not entry:
        return None
    if (time.time() - entry["t"]) > CACHE_TTL_SEC:
        _CACHE.pop(key, None)
        return None
    return entry["v"]


def _cache_set(key: Tuple[Any, ...], value: Dict[str, Any]) -> None:
    _CACHE[key] = {"t": time.time(), "v": value}


# -----------------------------
# Vector store
# -----------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

embeddings = OpenAIEmbeddings(
    api_key=secrets.get("OPENAI_API_KEY", ""),
    model=EMBED_MODEL,
)

VECTORDIR = PROJECT_DIR / "chroma_db"

vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory=str(VECTORDIR),
)

# -----------------------------
# LLM provider switch
# -----------------------------
provider = os.getenv("LLM_PROVIDER", "OPENAI").upper()

if provider == "OPENAI":
    llm = ChatOpenAI(
        api_key=secrets.get("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
elif provider == "ANTHROPIC":
    llm = ChatAnthropic(
        api_key=secrets.get("ANTHROPIC_API_KEY", ""),
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        temperature=TEMPERATURE,
    )
elif provider == "GEMINI":
    llm = ChatGoogleGenerativeAI(
        api_key=secrets.get("GOOGLE_API_KEY", ""),
        model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        temperature=TEMPERATURE,
    )
else:
    raise ValueError(f"Invalid LLM_PROVIDER: {provider}")

# -----------------------------
# Prompt (short + strict)
# -----------------------------
SYSTEM_PROMPT = """You are an AI assistant that answers questions about BMW TechWorks Romania employees.

Rules:
- Use ONLY the provided context records.
- Do NOT output long lists of employees (the UI shows employee cards).
- Keep the answer concise: <= 120 words unless the user explicitly asks for details.
- If you don't know from context, say you don't know.
"""

USER_PROMPT = """Question: {question}

Context records:
{context}

Answer in Romanian, clear and concise. Focus on explaining *why* the suggested people match the query (rol, tehnologie, seniority, locație). Include 2-3 exemple concrete din context (nume + rol) pentru a justifica selecția și menționează rapid criteriile cheie (stack, experiență relevantă). Evită liste lungi; oferă un rezumat scurt și util.
"""

# -----------------------------
# Helpers: normalization + dedupe
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
            slug = path[len("/in/") :]
        else:
            slug = path.lstrip("/")
        slug = slug.rstrip("/").lower()
        if not slug:
            return ""
        return f"https://{host}/in/{slug}"
    except Exception:
        return clean.lower()


def _canonical_vmid(vmid: Optional[str]) -> str:
    if not vmid:
        return ""
    raw = str(vmid).strip().lower()
    return "".join(ch for ch in raw if ch.isalnum())


def _source_key(md: Dict[str, Any]) -> str:
    vmid = _canonical_vmid(md.get("vmid"))
    profile = _normalize_linkedin(md.get("profile_url"))
    full_name = str(md.get("full_name") or "").strip().lower()
    fallback_id = str(md.get("id") or "").strip().lower()
    return vmid or profile or full_name or fallback_id


def _completeness_score(rec: Dict[str, Any]) -> int:
    score = 0
    for k, weight in {
        "profile_image_url": 3,
        "profile_url": 2,
        "headline": 2,
        "job_title": 2,
        "job_date_range": 2,
        "location": 2,
        "job_title_2": 1,
        "job_date_range_2": 1,
        "school": 1,
        "school_2": 1,
    }.items():
        if rec.get(k):
            score += weight
    score += sum(1 for v in rec.values() if v not in (None, "", []))
    return score


def _merge_prefer_richer(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    left = _completeness_score(existing)
    right = _completeness_score(incoming)
    base, secondary = (incoming.copy(), existing) if right > left else (existing.copy(), incoming)
    for k, v in secondary.items():
        if base.get(k) in (None, "", []):
            base[k] = v
    return base


def _docs_to_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    unique: Dict[str, Dict[str, Any]] = {}
    for d in docs:
        md = d.metadata or {}

        s = {
            "source_id": md.get("id"),
            "vmid": md.get("vmid"),
            "full_name": md.get("full_name"),
            "profile_url": md.get("profile_url"),
            "profile_image_url": md.get("profile_image_url"),
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
        }

        key = _source_key(md)
        if not key:
            continue
        if key not in unique:
            unique[key] = s
        else:
            unique[key] = _merge_prefer_richer(unique[key], s)

    return list(unique.values())


def _sources_to_compact_context(sources: List[Dict[str, Any]], limit: int) -> str:
    """
    IMPORTANT: context compact (metadata only) => prompt mult mai mic => LLM mult mai rapid.
    """
    lines: List[str] = []
    for s in sources[: max(1, limit)]:
        lines.append(f"Employee: {s.get('full_name') or 'N/A'}")
        if s.get("headline"):
            lines.append(f"Headline: {s['headline']}")
        if s.get("job_title"):
            role = s["job_title"]
            period = s.get("job_date_range") or ""
            lines.append(f"Current role: {role}" + (f" ({period})" if period else ""))
        if s.get("job_title_2"):
            role2 = s["job_title_2"]
            period2 = s.get("job_date_range_2") or ""
            lines.append(f"Previous role: {role2}" + (f" ({period2})" if period2 else ""))
        if s.get("location"):
            lines.append(f"Location: {s['location']}")
        edu1 = " – ".join([x for x in [s.get("school"), s.get("school_degree"), s.get("school_date_range")] if x])
        edu2 = " – ".join([x for x in [s.get("school_2"), s.get("school_degree_2"), s.get("school_date_range_2")] if x])
        if edu1:
            lines.append(f"Education 1: {edu1}")
        if edu2:
            lines.append(f"Education 2: {edu2}")
        if s.get("profile_url"):
            lines.append(f"LinkedIn: {s['profile_url']}")
        if s.get("vmid"):
            lines.append(f"VMID: {s['vmid']}")
        lines.append("-----")
    return "\n".join(lines).strip()


def _looks_like_short_search(query: str) -> bool:
    q = query.strip()
    if len(q) <= 22 and "?" not in q:
        return True
    return False


# -----------------------------
# Core: SEARCH (fast cards)
# -----------------------------
def run_search(query: str, k: int) -> Dict[str, Any]:
    start = time.time()

    cache_key = ("search", query.strip().lower(), int(k), EMBED_MODEL)
    cached = _cache_get(cache_key)
    if cached:
        return cached

    t0 = time.time()
    docs = vectordb.similarity_search(query, k=k)
    t1 = time.time()

    sources = _docs_to_sources(docs)

    result = {
        "sources": sources,
        "retrieved_docs": len(docs),
        "unique_sources": len(sources),
        "latency_sec": round(time.time() - start, 2),
        "retrieval_sec": round(t1 - t0, 2),
        "k": k,
        "architecture": {
            "embedding_model": EMBED_MODEL,
            "vectordb": "Chroma",
            "llm_provider": provider,
        }
    }
    _cache_set(cache_key, result)
    return result


@app.get("/search")
def search_endpoint(
    q: str = FastQuery(..., min_length=1),
    k: int = FastQuery(SEARCH_K_DEFAULT, ge=1, le=200),
) -> Dict[str, Any]:
    return run_search(q, k)


# -----------------------------
# Core: RAG (LLM analysis)
# -----------------------------
def run_rag(query: str, k: int, with_llm: bool) -> Dict[str, Any]:
    start = time.time()

    # 1) Retrieve (same as search, but allow different k)
    t0 = time.time()
    docs = vectordb.similarity_search(query, k=k)
    t1 = time.time()

    sources = _docs_to_sources(docs)

    # 2) Optional: skip LLM for short "keyword search" queries
    if with_llm is False or (SKIP_LLM_FOR_SHORT and _looks_like_short_search(query)):
        answer = "Am găsit profile relevante. Vezi cardurile din dreapta."
        return {
            "answer": answer,
            "sources": sources,
            "llm_used": "SKIPPED",
            "retrieved_docs": len(docs),
            "unique_sources": len(sources),
            "k": k,
            "latency_sec": round(time.time() - start, 2),
            "retrieval_sec": round(t1 - t0, 2),
            "llm_sec": 0.0,
            "architecture": {
                "embedding_model": EMBED_MODEL,
                "vectordb": "Chroma",
                "llm_provider": provider,
            }
        }

    # Cache LLM answers (great for demos)
    cache_key = ("rag", query.strip().lower(), int(k), provider, MAX_TOKENS, TEMPERATURE, CONTEXT_MAX_PEOPLE)
    cached = _cache_get(cache_key)
    if cached:
        return cached

    # 3) Build SMALL context for LLM (metadata only)
    context_text = _sources_to_compact_context(sources, limit=min(CONTEXT_MAX_PEOPLE, len(sources)))

    # 4) Call LLM
    t2 = time.time()
    resp = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT.format(question=query, context=context_text)),
        ]
    )
    t3 = time.time()

    answer = getattr(resp, "content", str(resp))

    result = {
        "answer": answer,
        "sources": sources,
        "llm_used": provider,
        "retrieved_docs": len(docs),
        "unique_sources": len(sources),
        "k": k,
        "latency_sec": round(t3 - start, 2),
        "retrieval_sec": round(t1 - t0, 2),
        "llm_sec": round(t3 - t2, 2),
        "architecture": {
            "embedding_model": EMBED_MODEL,
            "vectordb": "Chroma",
            "llm_provider": provider,
        }
    }
    _cache_set(cache_key, result)
    return result


# -----------------------------
# FastAPI schema & endpoint
# -----------------------------
class QueryBody(BaseModel):
    query: str = Field(..., min_length=1)
    with_llm: bool = True
    k: Optional[int] = None


@app.post("/query")
def rag_query(q: QueryBody) -> Dict[str, Any]:
    k = int(q.k or RAG_K_DEFAULT)
    k = max(1, min(200, k))
    return run_rag(q.query, k=k, with_llm=q.with_llm)
