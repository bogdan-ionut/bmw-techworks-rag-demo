# app/rag/retrieve.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

try:
    from pinecone import Pinecone
except Exception as e:  # pragma: no cover
    raise RuntimeError("Missing dependency: pinecone. Install with: pip install pinecone") from e

from app.core.config import get_settings as get_app_settings


def get_pinecone_index(pc: Pinecone, index_name: str):
    desc = pc.describe_index(index_name)
    host = getattr(desc, "host", None)
    if not host:
        status = getattr(desc, "status", None)
        if isinstance(status, dict):
            host = status.get("host")
    return pc.Index(index_name, host=host) if host else pc.Index(index_name)


def _join_nonempty(items: List[str], sep: str = " | ") -> str:
    clean = [x.strip() for x in items if x and x.strip()]
    return sep.join(clean)


def profile_text_from_metadata(md: Dict[str, Any]) -> str:
    # Prefer the ingestion-built embedding_text if present (best coverage)
    emb = (md.get("embedding_text") or "").strip()
    if emb:
        return emb

    name = md.get("full_name") or ""
    headline = md.get("headline") or ""
    loc = md.get("location") or md.get("location_normalized") or ""

    job1 = _join_nonempty([md.get("job_title") or "", md.get("job_date_range") or ""], sep=" — ")
    job2 = _join_nonempty([md.get("job_title_2") or "", md.get("job_date_range_2") or ""], sep=" — ")

    edu1 = _join_nonempty(
        [md.get("school") or "", md.get("school_degree") or "", md.get("school_date_range") or ""],
        sep=" — ",
    )
    edu2 = _join_nonempty(
        [md.get("school_2") or "", md.get("school_degree_2") or "", md.get("school_date_range_2") or ""],
        sep=" — ",
    )

    tech_tokens = md.get("tech_tokens") or []
    tech_line = ", ".join([str(x) for x in tech_tokens][:40]) if isinstance(tech_tokens, list) else str(tech_tokens)

    lines: List[str] = []
    if name:
        lines.append(f"Name: {name}")
    if headline:
        lines.append(f"Headline: {headline}")
    if loc:
        lines.append(f"Location: {loc}")
    if job1.strip(" — "):
        lines.append(f"Current role: {job1}")
    if job2.strip(" — "):
        lines.append(f"Previous role: {job2}")
    if edu1:
        lines.append(f"Education: {edu1}")
    if edu2:
        lines.append(f"Education 2: {edu2}")
    if tech_line:
        lines.append(f"Keywords: {tech_line}")

    if md.get("profile_url"):
        lines.append(f"LinkedIn: {md['profile_url']}")
    if md.get("profile_image_s3_url"):
        lines.append(f"Image: {md['profile_image_s3_url']}")

    return "\n".join([l for l in lines if l.strip()])


def retrieve_profiles(
    query: str,
    *,
    top_k: int = 12,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    s = get_app_settings()

    embeddings = OpenAIEmbeddings(api_key=s.openai_api_key, model=s.embed_model)
    qvec = embeddings.embed_query(query)

    pc = Pinecone(api_key=s.pinecone_api_key)
    index = get_pinecone_index(pc, s.pinecone_index_name)

    resp = index.query(
        vector=qvec,
        top_k=top_k,
        namespace=s.pinecone_namespace,
        include_metadata=True,
        filter=metadata_filter or None,
    )

    matches = getattr(resp, "matches", None) or []
    docs: List[Document] = []

    for m in matches:
        md = dict(getattr(m, "metadata", None) or {})
        score = getattr(m, "score", None)
        _id = getattr(m, "id", None)
        md["_score"] = score
        md["_id"] = _id

        content = profile_text_from_metadata(md)
        docs.append(Document(page_content=content, metadata=md))

    return docs


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Test retrieval from Pinecone")
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--filter-json", type=str, default="", help='Pinecone filter JSON (use $eq/$and etc)')
    args = parser.parse_args()

    md_filter = json.loads(args.filter_json) if args.filter_json.strip() else None
    docs = retrieve_profiles(args.query, top_k=args.top_k, metadata_filter=md_filter)

    print(f"\nGot {len(docs)} results\n")
    for i, d in enumerate(docs, start=1):
        print("=" * 90)
        print(f"[{i}] score={d.metadata.get('_score')}  id={d.metadata.get('_id')}")
        print(d.page_content[:1200])
        print()


if __name__ == "__main__":
    _cli()
