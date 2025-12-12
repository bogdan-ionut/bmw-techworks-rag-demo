# app/api.py

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# LangChain imports (v1+ style)
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

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
    """
    1) Încearcă să citească API keys din AWS Secrets Manager (dacă ai credențiale)
    2) Dacă nu are credențiale / secretul nu există, cade frumos pe .env
    """
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
# Vector store & retriever
# -----------------------------
RAG_K = int(os.getenv("RAG_K", "8"))

embeddings = OpenAIEmbeddings(
    api_key=secrets.get("OPENAI_API_KEY", ""),
    model="text-embedding-3-large",
)

VECTORDIR = PROJECT_DIR / "chroma_db"

vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory=str(VECTORDIR),
)

retriever = vectordb.as_retriever(search_kwargs={"k": RAG_K})


# -----------------------------
# LLM provider switch
# -----------------------------
provider = os.getenv("LLM_PROVIDER", "OPENAI").upper()

if provider == "OPENAI":
    llm = ChatOpenAI(
        api_key=secrets.get("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.1")),
    )
elif provider == "ANTHROPIC":
    llm = ChatAnthropic(
        api_key=secrets.get("ANTHROPIC_API_KEY", ""),
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        temperature=float(os.getenv("TEMPERATURE", "0.1")),
    )
elif provider == "GEMINI":
    llm = ChatGoogleGenerativeAI(
        api_key=secrets.get("GOOGLE_API_KEY", ""),
        model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        temperature=float(os.getenv("TEMPERATURE", "0.1")),
    )
else:
    raise ValueError(f"Invalid LLM_PROVIDER: {provider}")


# -----------------------------
# Prompt template
# -----------------------------
SYSTEM_PROMPT = """You are an AI assistant that answers questions about BMW TechWorks Romania employees.

You receive a user question and a set of employee records in the following text format:

Employee: <full_name>
Role: <job_title> (<job_date_range>)
Second Role: <job_title_2> (<job_date_range_2>)
Headline: <headline>
Location: <location>
Education 1: <school> – <school_degree> (<school_date_range>)
Education 2: <school_2> – <school_degree_2> (<school_date_range_2>)
LinkedIn: <profile_url>
VMID: <vmid>

Rules:
- Use ONLY the given context to answer.
- If some fields are missing (empty), just omit them.
- If you don't know the answer from the context, say you don't know.
- When listing employees, show as many useful fields as possible (roles, dates, location, education, LinkedIn).
- Keep answers concise but information-dense.
"""

USER_PROMPT = """Question: {question}

Context:
{context}

Now answer clearly in Romanian and, if it helps clarity, you can include bullet lists.
"""


# -----------------------------
# RAG core logic
# -----------------------------
def docs_to_rich_text(docs: List[Document]) -> str:
    return "\n\n-----\n\n".join(d.page_content for d in docs)


def _source_key(md: Dict[str, Any]) -> str:
    return (
        str(md.get("vmid") or "").strip()
        or str(md.get("profile_url") or "").strip()
        or str(md.get("full_name") or "").strip()
        or str(md.get("id") or "").strip()
    )


def _merge_missing(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if dst.get(k) in (None, "", []):
            dst[k] = v
    return dst


def run_rag(query: str) -> Dict[str, Any]:
    start = time.time()
    print(f"\n[RAG] New query: {query!r}")

    # 1) Retrieve
    t0 = time.time()
    docs = retriever.invoke(query)
    t1 = time.time()
    print(f"[RAG] Retrieved {len(docs)} docs in {t1 - t0:.2f}s")

    context_text = docs_to_rich_text(docs)

    # 2) Build full prompt
    full_prompt = f"{SYSTEM_PROMPT}\n\n" + USER_PROMPT.format(
        question=query,
        context=context_text,
    )

    # 3) Call LLM
    print("[RAG] Calling LLM...")
    t2 = time.time()
    resp = llm.invoke(full_prompt)
    t3 = time.time()
    total = t3 - start
    print(f"[RAG] LLM answered in {t3 - t2:.2f}s (total {total:.2f}s)")

    answer = getattr(resp, "content", str(resp))

    # 4) Build + DEDUP sources (important: multiple chunks can map to same person)
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
            unique[key] = _merge_missing(unique[key], s)

    sources = list(unique.values())

    return {
        "answer": answer,
        "sources": sources,
        "llm_used": provider,
        "retrieved_docs": len(docs),
        "k": RAG_K,
        "unique_sources": len(sources),
        "latency_sec": round(total, 2),
    }


# -----------------------------
# FastAPI schema & endpoint
# -----------------------------
class Query(BaseModel):
    query: str


@app.post("/query")
def rag_query(q: Query) -> Dict[str, Any]:
    """
    Body:
    {
        "query": "câți angajați cu numele de Iulia avem? și de unde sunt?"
    }
    """
    return run_rag(q.query)
