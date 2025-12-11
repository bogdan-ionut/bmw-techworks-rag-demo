# app/api.py

import os
import json
import time
from typing import Any, Dict, List

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from dotenv import load_dotenv

from fastapi import FastAPI
from pydantic import BaseModel

# LangChain imports (v1+ style)
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

load_dotenv()

app = FastAPI(title="BMW TechWorks RAG Demo")


# ---------- Helpers: load secrets (AWS Secrets Manager + .env fallback) ----------

def load_secrets() -> Dict[str, str]:
    """
    1) Încearcă să citească API keys din AWS Secrets Manager (dacă ai credențiale)
    2) Dacă nu are credențiale / secretul nu există, cade frumos pe .env
    """
    secret_name = os.getenv("AWS_SECRET_NAME")
    region = os.getenv("AWS_REGION", "eu-north-1")

    # Dacă nu e setat numele secretului, mergem direct pe .env
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

    # Fallback la .env dacă lipsesc anumite chei
    return {
        "OPENAI_API_KEY": secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
        "ANTHROPIC_API_KEY": secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY", ""),
        "GOOGLE_API_KEY": secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY", ""),
    }


secrets = load_secrets()


# ---------- Vector store & retriever ----------

# Același embeddings ca în main.py (text-embedding-3-large)
embeddings = OpenAIEmbeddings(
    api_key=secrets.get("OPENAI_API_KEY", ""),
    model="text-embedding-3-large",
)

# Încarcă Chroma DB creat de main.py
VECTORDIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")

vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory=VECTORDIR,
)

retriever = vectordb.as_retriever(search_kwargs={"k": 8})


# ---------- LLM provider switch ----------

provider = os.getenv("LLM_PROVIDER", "OPENAI").upper()

if provider == "OPENAI":
    llm = ChatOpenAI(
        api_key=secrets.get("OPENAI_API_KEY", ""),
        model="gpt-4o-mini",
        temperature=0.1,
    )
elif provider == "ANTHROPIC":
    llm = ChatAnthropic(
        api_key=secrets.get("ANTHROPIC_API_KEY", ""),
        model="claude-3-5-sonnet-20241022",
        temperature=0.1,
    )
elif provider == "GEMINI":
    llm = ChatGoogleGenerativeAI(
        api_key=secrets.get("GOOGLE_API_KEY", ""),
        model="gemini-1.5-flash",
        temperature=0.1,
    )
else:
    raise ValueError(f"Invalid LLM_PROVIDER: {provider}")


# ---------- Prompt template ----------

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


# ---------- RAG core logic ----------

def docs_to_rich_text(docs: List[Document]) -> str:
    """Concatenează documentele cu separatori, pentru context LLM."""
    return "\n\n-----\n\n".join(d.page_content for d in docs)


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
    print(f"[RAG] LLM answered in {t3 - t2:.2f}s (total {t3 - start:.2f}s)")

    # Chat models return a message object with `.content`
    answer = getattr(resp, "content", str(resp))

    # 4) Build sources metadata (to le poți afișa frumos în UI / JSON)
    sources = []
    for d in docs:
        md = d.metadata or {}
        sources.append(
            {
                "source_id": md.get("id"),
                "profile_url": md.get("profile_url"),
                "full_name": md.get("full_name"),
                "location": md.get("location"),
                "job_title": md.get("job_title"),
                "job_title_2": md.get("job_title_2"),
                "headline": md.get("headline"),
            }
        )

    return {
        "answer": answer,
        "sources": sources,
        "llm_used": provider,
        "retrieved_docs": len(docs),
        "latency_sec": round(t3 - start, 2),
    }


# ---------- FastAPI schema & endpoint ----------

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
