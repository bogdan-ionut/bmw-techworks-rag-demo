# app/rag/rerank.py
from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.documents import Document

try:
    from langchain_cohere import CohereRerank
except Exception:
    try:
        from langchain_community.document_compressors import CohereRerank  # type: ignore
    except Exception:
        CohereRerank = None  # type: ignore

logger = logging.getLogger(__name__)


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    *,
    cohere_api_key: str,
    model: str = "rerank-english-v3.0",
    top_n: int = 8,
) -> List[Dict[str, Any]]:
    """
    Reranks a list of candidate dictionaries using Cohere.
    Dynamic behavior:
      - If len(candidates) < 2, skips reranking (returns as-is).
      - If len(candidates) < top_n, adjusts top_n to len(candidates).
    """
    if not candidates:
        return []

    # Dynamic Optimization: Skip reranking if there's only 1 (or 0) candidate.
    # It preserves the vector score order (which is just 1 item).
    if len(candidates) < 2:
        logger.debug("Skipping rerank for single candidate.")
        return candidates

    # Cap top_n to the number of candidates available to avoid requesting more than we have
    # (though Cohere handles this, it's cleaner to be explicit).
    effective_top_n = min(len(candidates), int(top_n))

    if not cohere_api_key:
        return candidates[:effective_top_n]

    if CohereRerank is None:
        logger.warning("CohereRerank library not found, skipping rerank.")
        return candidates[:effective_top_n]

    try:
        rr = CohereRerank(
            cohere_api_key=cohere_api_key,
            model=model,
            top_n=effective_top_n,
        )

        docs = [
            Document(page_content=str(c.get("text") or ""), metadata=c)
            for c in candidates
        ]

        # compress_documents returns the top_n documents
        reranked_docs = rr.compress_documents(documents=docs, query=query)
        out = [d.metadata for d in reranked_docs]
        return out
    except Exception as e:
        logger.error(f"Cohere rerank failed: {e}", exc_info=True)
        # Fallback to original order
        return candidates[:effective_top_n]
