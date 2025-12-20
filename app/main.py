# app/main.py
from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Gemini (optional)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None

# Cohere rerank – try both import paths for compatibility
try:
    from langchain_cohere import CohereRerank
except Exception:
    try:
        from langchain_community.document_compressors import CohereRerank  # older path
    except Exception:
        CohereRerank = None

from pinecone import Pinecone

from app.api.routes import router as api_router
from app.core.config import get_settings, Settings
from app.core.logging import init_logging

# init logging as early as possible
init_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json_logs=os.getenv("LOG_JSON", "0").strip().lower() in {"1", "true", "yes", "y", "on"},
)

logger = logging.getLogger(__name__)


def resolve_web_dir(base_dir: Path) -> Optional[Path]:
    web_dir = base_dir / "web"
    if (web_dir / "index.html").exists():
        return web_dir
    return None


def init_pinecone_index(pinecone_api_key: str, index_name: str):
    if not pinecone_api_key:
        raise RuntimeError("Missing PINECONE_API_KEY (env or AWS secret).")
    if not index_name:
        raise RuntimeError("Missing PINECONE_INDEX_NAME (env or AWS secret).")
    pc = Pinecone(api_key=pinecone_api_key)
    return pc.Index(index_name)


def init_cohere_reranker(cohere_api_key: str, model: str, top_n: int):
    """
    Newer langchain-cohere requires `model=...`.
    We'll try a small fallback list (English-first) to avoid blocking startup.
    """
    if not cohere_api_key:
        return None
    if CohereRerank is None:
        raise RuntimeError("COHERE_API_KEY set but CohereRerank import failed. Install langchain-cohere.")

    candidates = []
    if model:
        candidates.append(model)

    # Fallbacks (English-first, then multilingual)
    candidates += [
        "rerank-english-v3.0",
        "rerank-english-v2.0",
        "rerank-multilingual-v3.0",
        "rerank-multilingual-v2.0",
    ]

    seen = set()
    candidates = [m for m in candidates if not (m in seen or seen.add(m))]

    last_err: Optional[Exception] = None
    for m in candidates:
        try:
            rr = CohereRerank(
                cohere_api_key=cohere_api_key,
                model=m,          # <- REQUIRED
                top_n=int(top_n),
            )
            logger.info("Cohere reranker enabled (model=%s, top_n=%s)", m, top_n)
            return rr
        except Exception as e:
            last_err = e
            logger.warning("Failed to init CohereRerank with model=%s (%s). Trying next...", m, e)

    raise RuntimeError(f"Could not initialize CohereRerank with any known model. Last error: {last_err}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("App settings: %s", settings.redacted())

    app.state.settings = settings
    app.state.cache = {}
    app.state.llm_cache = {}

    # Embeddings (OpenAI)
    if not settings.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY (env or AWS secret).")
    app.state.embeddings = OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.embed_model,
    )

    # LLM provider
    llm_cache_key = None
    if settings.llm_provider == "OPENAI":
        app.state.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        llm_cache_key = (
            "OPENAI",
            settings.openai_model,
            settings.temperature,
            settings.max_tokens,
        )
    elif settings.llm_provider == "GEMINI":
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError("Gemini selected but langchain-google-genai not installed.")
        if not settings.google_api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY for Gemini (env or AWS secret).")
        app.state.llm = ChatGoogleGenerativeAI(
            api_key=settings.google_api_key,
            model=settings.gemini_model,
            temperature=settings.temperature,
        )
        llm_cache_key = (
            "GEMINI",
            settings.gemini_model,
            settings.temperature,
            settings.max_tokens,
        )
    else:
        raise RuntimeError(f"Invalid LLM_PROVIDER={settings.llm_provider}. Use OPENAI or GEMINI.")

    if llm_cache_key:
        app.state.llm_cache[llm_cache_key] = app.state.llm

    # Pinecone
    app.state.pinecone_index = init_pinecone_index(settings.pinecone_api_key, settings.pinecone_index_name)

    # Cohere reranker (optional)
    app.state.cohere_reranker = init_cohere_reranker(
        cohere_api_key=settings.cohere_api_key,
        model=settings.cohere_rerank_model,
        top_n=settings.cohere_rerank_top_n,
    )

    yield


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="BMW TechWorks RAG Demo", lifespan=lifespan)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Pass through health, root, web static files, and auth verification
    # Using startswith for /web to cover all static assets
    if (
        request.url.path in ["/", "/health", "/auth/verify"]
        or request.url.path.startswith("/web")
    ):
        return await call_next(request)

    # Check for access_token cookie
    token = request.cookies.get("access_token")
    if token == "authorized":
        return await call_next(request)

    return JSONResponse(status_code=403, content={"detail": "Not authenticated"})


class AuthRequest(BaseModel):
    password: str


@app.post("/auth/verify")
async def verify_password(
    body: AuthRequest,
    response: Response,
    settings: Settings = Depends(get_settings)
):
    if body.password.strip() == settings.demo_password:
        # Set HttpOnly cookie valid for session (or max_age)
        response.set_cookie(
            key="access_token",
            value="authorized",
            httponly=True,
            samesite="lax"
        )
        return {"status": "ok"}
    raise HTTPException(status_code=401, detail="Invalid password")


app.include_router(api_router)

WEB_DIR = resolve_web_dir(BASE_DIR)
if WEB_DIR:
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

    @app.get("/", include_in_schema=False)
    def serve_homepage():
        return FileResponse(str(WEB_DIR / "index.html"))
else:
    @app.get("/", include_in_schema=False)
    def serve_homepage():
        return {"message": "Frontend missing. Expected: app/web/index.html"}
