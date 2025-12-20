# app/core/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional

# Optional: load .env in local dev
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    # App
    env: str
    aws_region: str
    aws_secret_id: str

    # Providers / Models
    llm_provider: str  # OPENAI or GEMINI
    openai_api_key: str
    google_api_key: str

    openai_model: str
    gemini_model: str
    embed_model: str

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_namespace: Optional[str]

    # Cohere rerank
    cohere_api_key: str
    cohere_rerank_model: str
    cohere_rerank_top_n: int
    rerank_in_search: bool

    # RAG behavior
    search_top_k: int
    context_max_people: int
    skip_llm_for_short: bool

    # LLM params
    temperature: float
    max_tokens: int

    # Cache
    cache_ttl_sec: int

    # Logging
    log_level: str
    log_json: bool

    # Info
    data_jsonl_path: str

    def redacted(self) -> Dict[str, Any]:
        def mask(x: str) -> str:
            if not x:
                return ""
            if len(x) <= 8:
                return "********"
            return x[:4] + "..." + x[-4:]

        return {
            "env": self.env,
            "aws_region": self.aws_region,
            "aws_secret_id": self.aws_secret_id,
            "llm_provider": self.llm_provider,
            "openai_api_key": mask(self.openai_api_key),
            "google_api_key": mask(self.google_api_key),
            "openai_model": self.openai_model,
            "gemini_model": self.gemini_model,
            "embed_model": self.embed_model,
            "pinecone_api_key": mask(self.pinecone_api_key),
            "pinecone_index_name": self.pinecone_index_name,
            "pinecone_namespace": self.pinecone_namespace,
            "cohere_api_key": mask(self.cohere_api_key),
            "cohere_rerank_model": self.cohere_rerank_model,
            "cohere_rerank_top_n": self.cohere_rerank_top_n,
            "rerank_in_search": self.rerank_in_search,
            "search_top_k": self.search_top_k,
            "context_max_people": self.context_max_people,
            "skip_llm_for_short": self.skip_llm_for_short,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "cache_ttl_sec": self.cache_ttl_sec,
            "log_level": self.log_level,
            "log_json": self.log_json,
            "data_jsonl_path": self.data_jsonl_path,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    from app.core.secrets import load_secrets

    aws_region = _env("AWS_REGION", "us-east-1") or "us-east-1"
    aws_secret_id = (_env("AWS_SECRET_ID") or _env("AWS_SECRET_NAME") or "").strip()

    secrets = load_secrets(secret_id=aws_secret_id or None, region=aws_region)

    def pick(key: str, default: str = "") -> str:
        v = _env(key)
        if v:
            return str(v)
        return str(secrets.get(key) or default)

    llm_provider = (pick("LLM_PROVIDER", "OPENAI")).upper()

    # Backward-compat: accept PINECONE_INDEX too
    pinecone_index_name = _env("PINECONE_INDEX_NAME") or _env("PINECONE_INDEX") or secrets.get("PINECONE_INDEX_NAME") or ""

    pinecone_ns = pick("PINECONE_NAMESPACE", "profiles").strip()
    pinecone_ns_opt = pinecone_ns if pinecone_ns else None

    # Cohere model (English-first)
    cohere_model = pick("COHERE_RERANK_MODEL", "rerank-english-v3.0").strip()

    return Settings(
        env=pick("APP_ENV", "local"),
        aws_region=aws_region,
        aws_secret_id=aws_secret_id or "",

        llm_provider=llm_provider,
        openai_api_key=pick("OPENAI_API_KEY", ""),
        google_api_key=pick("GOOGLE_API_KEY", ""),

        openai_model=pick("OPENAI_MODEL", "gpt-4o-mini"),
        gemini_model=pick("GEMINI_MODEL", "gemini-1.5-flash"),
        embed_model=pick("EMBED_MODEL", "text-embedding-3-large"),

        pinecone_api_key=pick("PINECONE_API_KEY", ""),
        pinecone_index_name=str(pinecone_index_name),
        pinecone_namespace=pinecone_ns_opt,

        cohere_api_key=pick("COHERE_API_KEY", ""),
        cohere_rerank_model=cohere_model,
        cohere_rerank_top_n=int(pick("COHERE_RERANK_TOP_N", "6")),
        rerank_in_search=_env_bool("RERANK_IN_SEARCH", default=True),

        search_top_k=int(pick("SEARCH_TOP_K", "24")),
        context_max_people=int(pick("CONTEXT_MAX_PEOPLE", "8")),
        skip_llm_for_short=pick("SKIP_LLM_FOR_SHORT", "0") == "1",

        temperature=float(pick("TEMPERATURE", "0.1")),
        max_tokens=int(pick("MAX_TOKENS", "1000")),

        cache_ttl_sec=int(pick("CACHE_TTL_SEC", "600")),

        log_level=pick("LOG_LEVEL", "INFO"),
        log_json=_env_bool("LOG_JSON", default=False),

        data_jsonl_path=pick(
            "DATA_JSONL_PATH",
            "/Users/ionutbogdan/PycharmProjects/bmw-techworks-rag-demo/data/bmw_employees_cleaned_s3.jsonl",
        ),
    )
