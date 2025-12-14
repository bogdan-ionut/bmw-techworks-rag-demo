import os
import time
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# --- Load .env explicitly ---
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DOTENV_PATH = PROJECT_ROOT / ".env"

if load_dotenv:
    if DOTENV_PATH.exists():
        load_dotenv(dotenv_path=DOTENV_PATH)
    else:
        load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage


STATIC_DIR = APP_DIR / "static"
INDEX_HTML = STATIC_DIR / "index.html"

CHROMA_DIR = os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "chroma_db"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "profiles")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

SEARCH_K_DEFAULT = int(os.getenv("SEARCH_K_DEFAULT", "16"))
RAG_K_DEFAULT = int(os.getenv("RAG_K_DEFAULT", "8"))
CONTEXT_MAX_PEOPLE = int(os.getenv("CONTEXT_MAX_PEOPLE", "8"))

SYSTEM_PROMPT = """Ești un asistent de analiză pentru o listă de profile (talent intelligence).
Reguli:
- Răspunde DOAR în limba română.
- NU lista nume complete sau URL-uri de profil în textul răspunsului (UI-ul afișează deja cardurile).
- Poți face referire la exemple anonimizate: „un profil cu rol GenAI Engineer…”.
- Dacă contextul nu conține suficiente informații, spune clar ce lipsește.
- Fii scurt, clar și util (max ~8-12 rânduri)."""

USER_PROMPT_TEMPLATE = """Întrebare: {query}

Context (profile-uri relevante, anonimizate):
{context}

Te rog:
1) Răspunde direct la întrebare.
2) Spune DE CE profile-urile din context se potrivesc (2-4 criterii).
3) Dă 2-3 exemple concrete din context (fără nume / fără URL), doar rol + 1-2 detalii relevante (ex: keywords/skills/locație/perioadă)."""

app = FastAPI(title="BMWTechWorks Talent Intelligence")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class SearchBody(BaseModel):
    query: str
    k: int = Field(default=SEARCH_K_DEFAULT, ge=1, le=200)


class QueryBody(BaseModel):
    query: str
    with_llm: bool = True
    k: Optional[int] = Field(default=None, ge=1, le=200)


_embeddings: Optional[OpenAIEmbeddings] = None
_vectordb: Optional[Chroma] = None
_ANALYSIS_CACHE: Dict[str, Dict[str, Any]] = {}


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)


def _require_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise HTTPException(
            status_code=500,
            detail=(
                "OPENAI_API_KEY lipsește. "
                "Asigură-te că ai un fișier .env în root și ai instalat python-dotenv "
                "(pip install python-dotenv), sau setează variabila de mediu OPENAI_API_KEY."
            ),
        )
    return key


def _get_vectordb() -> Chroma:
    global _embeddings, _vectordb
    _require_openai_key()

    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if _vectordb is None:
        _vectordb = Chroma(
            collection_name=CHROMA_COLLECTION,
            persist_directory=CHROMA_DIR,
            embedding_function=_embeddings,
        )

    return _vectordb


def _doc_to_source(doc: Any) -> Dict[str, Any]:
    md = getattr(doc, "metadata", {}) or {}
    return {
        "source_id": md.get("source_id") or md.get("vmid") or md.get("id") or "",
        "vmid": md.get("vmid") or md.get("source_id") or "",
        "full_name": md.get("full_name") or md.get("name") or "",
        "profile_url": md.get("profile_url") or md.get("url") or "",
        "profile_image_url": md.get("profile_image_url") or md.get("image") or "",
        "headline": md.get("headline") or "",
        "location": md.get("location") or "",
        "job_title": md.get("job_title") or "",
        "job_date_range": md.get("job_date_range") or "",
        "job_title_2": md.get("job_title_2") or "",
        "job_date_range_2": md.get("job_date_range_2") or "",
        "school": md.get("school") or "",
        "school_degree": md.get("school_degree") or "",
        "school_date_range": md.get("school_date_range") or "",
        "school_2": md.get("school_2") or "",
        "school_degree_2": md.get("school_degree_2") or "",
        "school_date_range_2": md.get("school_date_range_2") or "",
    }


def _docs_to_sources(docs: List[Any]) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for d in docs:
        s = _doc_to_source(d)
        sid = s.get("source_id") or s.get("vmid") or _safe_str(hash(d))
        if sid and sid not in seen:
            seen[sid] = s
    return list(seen.values())


def _compact_context(sources: List[Dict[str, Any]], max_people: int) -> str:
    lines: List[str] = []
    for i, s in enumerate(sources[:max_people], start=1):
        role = (s.get("job_title") or "").strip()
        headline = (s.get("headline") or "").strip()
        loc = (s.get("location") or "").strip()

        extra = []
        if role:
            extra.append(f"rol: {role}")
        if headline and headline != role:
            extra.append(f"headline: {headline}")
        if loc:
            extra.append(f"locație: {loc}")

        if not extra:
            extra.append("profil fără câmpuri standardizate (metadata incomplet)")
        lines.append(f"- Profil {i}: " + " | ".join(extra))

    return "\n".join(lines).strip()


def _extract_refusal(resp: Any) -> Optional[str]:
    add = getattr(resp, "additional_kwargs", None) or {}
    meta = getattr(resp, "response_metadata", None) or {}

    refusal = add.get("refusal") or add.get("refusal_reason")
    if refusal:
        return _safe_str(refusal).strip() or None

    msg = meta.get("message") if isinstance(meta, dict) else None
    if isinstance(msg, dict):
        r2 = msg.get("refusal")
        if r2:
            return _safe_str(r2).strip() or None

    r3 = meta.get("refusal") if isinstance(meta, dict) else None
    if r3:
        return _safe_str(r3).strip() or None

    return None


def _extract_llm_text(resp: Any) -> str:
    content = getattr(resp, "content", None)

    if isinstance(content, str):
        t = content.strip()
        if t:
            return t

    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if p.get("type") == "text" and p.get("text"):
                    parts.append(_safe_str(p["text"]))
                elif p.get("text"):
                    parts.append(_safe_str(p["text"]))
        t = "\n".join([x.strip() for x in parts if x and x.strip()]).strip()
        if t:
            return t

    refusal = _extract_refusal(resp)
    if refusal:
        return f"⚠️ Modelul a refuzat să răspundă: {refusal}"

    s = _safe_str(resp).strip()
    if s in ("AIMessage(content='')", "AIMessage(content=\"\")"):
        return ""
    return s


def _make_llm() -> ChatOpenAI:
    _require_openai_key()
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)


@app.get("/", response_class=HTMLResponse)
def home():
    if not INDEX_HTML.exists():
        return HTMLResponse("<h3>Missing static/index.html</h3>", status_code=500)
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {
        "ok": True,
        "dotenv_loaded": bool(load_dotenv),
        "dotenv_path": str(DOTENV_PATH),
        "dotenv_exists": DOTENV_PATH.exists(),
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "model": OPENAI_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "chroma_dir": CHROMA_DIR,
        "chroma_collection": CHROMA_COLLECTION,
    }


@app.get("/debug/db")
def debug_db():
    """
    Ajută enorm la diagnosticul "0 rezultate":
    - vezi câți vectori sunt în colecția curentă
    - vezi lista colecțiilor existente în folderul chroma_db
    """
    _require_openai_key()

    collections: List[str] = []
    try:
        import chromadb  # type: ignore
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collections = [c.name for c in client.list_collections()]
    except Exception:
        collections = []

    count = None
    try:
        vdb = _get_vectordb()
        count = vdb._collection.count()  # type: ignore[attr-defined]
    except Exception:
        count = None

    return {
        "chroma_dir": CHROMA_DIR,
        "dir_exists": Path(CHROMA_DIR).exists(),
        "collection_used_by_api": CHROMA_COLLECTION,
        "vectors_in_collection_used_by_api": count,
        "collections_found_in_dir": collections,
        "hint": (
            "Dacă ai vectori în 'langchain' dar API folosește 'profiles', "
            "setează CHROMA_COLLECTION=langchain SAU reconstruiește DB în 'profiles'."
        ),
    }


@app.post("/search")
def search_profiles(body: SearchBody):
    t0 = time.time()
    vectordb = _get_vectordb()

    docs = vectordb.similarity_search(body.query, k=body.k)
    sources = _docs_to_sources(docs)
    return {
        "query": body.query,
        "k": body.k,
        "count": len(sources),
        "latency_sec": round(time.time() - t0, 3),
        "sources": sources,
    }


@app.post("/query")
def query_rag(body: QueryBody):
    t0 = time.time()
    k = body.k or RAG_K_DEFAULT

    t_retrieval0 = time.time()
    vectordb = _get_vectordb()
    docs = vectordb.similarity_search(body.query, k=k)
    retrieval_sec = time.time() - t_retrieval0

    sources = _docs_to_sources(docs)

    top_ids = [s.get("source_id") or s.get("vmid") or "" for s in sources[:CONTEXT_MAX_PEOPLE]]
    cache_key = _sha256(
        json.dumps(
            {
                "provider": "OPENAI",
                "model": OPENAI_MODEL,
                "embedding": EMBEDDING_MODEL,
                "query": body.query,
                "k": k,
                "top_ids": top_ids,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    if body.with_llm and cache_key in _ANALYSIS_CACHE:
        cached = _ANALYSIS_CACHE[cache_key]
        cached["latency_sec"] = round(time.time() - t0, 3)
        cached["retrieval_sec"] = round(retrieval_sec, 3)
        cached["cache_hit"] = True
        return cached

    context = _compact_context(sources, CONTEXT_MAX_PEOPLE)
    if not context:
        answer = "Nu am găsit profile relevante în baza vectorială pentru această întrebare."
        payload = {
            "answer": answer,
            "sources": sources,
            "llm_used": "OPENAI",
            "retrieved_docs": len(docs),
            "unique_sources": len(sources),
            "k": k,
            "latency_sec": round(time.time() - t0, 3),
            "retrieval_sec": round(retrieval_sec, 3),
            "llm_sec": 0.0,
            "architecture": {
                "embedding_model": EMBEDDING_MODEL,
                "vectordb": "Chroma",
                "llm_provider": "OPENAI",
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "prompt_characters": 0,
            },
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "cache_hit": False,
        }
        _ANALYSIS_CACHE[cache_key] = payload
        return payload

    llm = _make_llm()
    user_prompt = USER_PROMPT_TEMPLATE.format(query=body.query, context=context)
    prompt_chars = len(SYSTEM_PROMPT) + len(user_prompt)

    t_llm0 = time.time()
    resp = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)])
    llm_sec = time.time() - t_llm0

    answer = _extract_llm_text(resp).strip()

    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    meta = getattr(resp, "response_metadata", None) or {}
    usage = meta.get("token_usage") if isinstance(meta, dict) else None
    if isinstance(usage, dict):
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)
        tt = int(usage.get("total_tokens") or (pt + ct))
        token_usage = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}

    if not answer:
        refusal = _extract_refusal(resp)
        if refusal:
            answer = f"⚠️ Modelul a refuzat să răspundă: {refusal}"
        else:
            answer = "Modelul nu a returnat conținut text pentru această cerere (posibil refuz sau răspuns gol)."

    payload = {
        "answer": answer,
        "sources": sources,
        "llm_used": "OPENAI",
        "retrieved_docs": len(docs),
        "unique_sources": len(sources),
        "k": k,
        "latency_sec": round(time.time() - t0, 3),
        "retrieval_sec": round(retrieval_sec, 3),
        "llm_sec": round(llm_sec, 3),
        "architecture": {
            "embedding_model": EMBEDDING_MODEL,
            "vectordb": "Chroma",
            "llm_provider": "OPENAI",
            "token_usage": token_usage,
            "prompt_characters": prompt_chars,
        },
        "token_usage": token_usage,
        "cache_hit": False,
    }

    _ANALYSIS_CACHE[cache_key] = payload
    return payload
