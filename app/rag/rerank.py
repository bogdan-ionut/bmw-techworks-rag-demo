# app/rag/rerank.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

try:
    from langchain_cohere import CohereRerank
except Exception:
    try:
        from langchain_community.document_compressors import CohereRerank  # type: ignore
    except Exception:
        CohereRerank = None  # type: ignore


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    *,
    cohere_api_key: str,
    model: str = "rerank-english-v3.0",
    top_n: int = 8,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    if not cohere_api_key:
        return candidates[:top_n]
    if CohereRerank is None:
        return candidates[:top_n]

    rr = CohereRerank(
        cohere_api_key=cohere_api_key,
        model=model,
        top_n=int(top_n),
    )

    docs = [
        Document(page_content=str(c.get("text") or ""), metadata=c)
        for c in candidates
    ]
    reranked_docs = rr.compress_documents(documents=docs, query=query)
    out = [d.metadata for d in reranked_docs]
    return out[:top_n]
