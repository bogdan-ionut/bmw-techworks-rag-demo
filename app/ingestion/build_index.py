# app/ingestion/build_index.py
from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from app.core.secrets import load_secrets

try:
    # Pinecone SDK v3+
    from pinecone import Pinecone
    from pinecone_text.sparse import BM25Encoder
except Exception as e:
    raise RuntimeError(
        "Missing dependency: pinecone or pinecone-text. Install with: pip install pinecone pinecone-text") from e

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except Exception:
    boto3 = None  # optional (S3 fallback)

# -----------------------------------------------------------------------------
# Paths / defaults
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_JSONL_PATH = DATA_DIR / "bmw_employees_cleaned_s3.jsonl"

DEFAULT_AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
DEFAULT_AWS_SECRET_NAME = os.getenv("AWS_SECRET_NAME", "bmw-techworks-rag-demo/secrets")
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

DEFAULT_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "64"))


# -----------------------------------------------------------------------------
# Optional: S3 download if local JSONL missing
# -----------------------------------------------------------------------------
def maybe_download_jsonl_from_s3(local_path: Path, bucket: str, key: str) -> None:
    if local_path.exists():
        return
    if not bucket or not key:
        return
    if boto3 is None:
        print("[WARN] boto3 not installed; cannot download JSONL from S3.")
        return

    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3 = boto3.client("s3")
        print(f"[INFO] Downloading JSONL from S3: s3://{bucket}/{key} -> {local_path}")
        s3.download_file(bucket, key, str(local_path))
        print("[INFO] S3 download complete.")
    except (BotoCoreError, ClientError) as e:
        print(f"[WARN] Failed to download JSONL from S3 ({e}). Continuing...")


# -----------------------------------------------------------------------------
# JSONL loading & Helpers
# -----------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    records.append(rec)
            except Exception:
                pass
    print(f"[INFO] Loaded {len(records)} records from {path}")
    return records


def _norm_str(x: Any) -> str:
    return str(x).strip() if x is not None else ""


def _normalize_name(s: Any) -> str:
    if not s: return ""
    text = str(s).strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text


def _safe_get(d: Dict[str, Any], path: List[str]) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict): return None
        cur = cur.get(k)
    return cur


def _as_list_of_str(x: Any) -> List[str]:
    if not x: return []
    if isinstance(x, list): return [str(v).strip() for v in x if str(v).strip()]
    s = str(x).strip()
    return [s] if s else []


def extract_tech_tokens(headline: str) -> List[str]:
    h = headline or ""
    parts = []
    for chunk in re.split(r"[|]", h):
        parts.extend([p.strip() for p in chunk.split(",") if p.strip()])
    noise = {"@", "at", "bmw", "techworks", "romania"}
    tokens = []
    for p in parts:
        p2 = re.sub(r"\s+", " ", p).strip()
        if not p2 or p2.lower() in noise: continue
        if 2 <= len(p2) <= 60: tokens.append(p2)
    return list(dict.fromkeys(tokens))[:40]


def infer_bool_from_text(s: str, negative_markers: Tuple[str, ...]) -> Optional[bool]:
    if not s: return None
    low = s.lower()
    for m in negative_markers:
        if m in low: return False
    return True


def build_embedding_text(rec: Dict[str, Any]) -> str:
    # Build rich text representation for embedding
    lines = []
    full_name = _norm_str(rec.get("full_name"))
    if full_name: lines.append(f"Name: {full_name}")

    headline = _norm_str(rec.get("headline"))
    if headline: lines.append(f"Headline: {headline}")

    tech_tokens = extract_tech_tokens(headline)
    if tech_tokens: lines.append("Tech: " + ", ".join(tech_tokens))

    job_title = _norm_str(rec.get("job_title"))
    if job_title: lines.append(f"Role: {job_title}")

    location = _norm_str(rec.get("location"))
    if location: lines.append(f"Location: {location}")

    school = _norm_str(rec.get("school"))
    if school: lines.append(f"Education: {school}")

    summary = _norm_str(_safe_get(rec, ["image_caption", "summary"]))
    if summary: lines.append(f"Image: {summary}")

    return "\n".join(lines).strip()


def build_metadata(rec: Dict[str, Any], embedding_text: str) -> Dict[str, Any]:
    eyewear = _norm_str(_safe_get(rec, ["image_caption", "appearance", "eyewear"]))
    facial_hair = _norm_str(_safe_get(rec, ["image_caption", "appearance", "facial_hair"]))

    eyewear_present = infer_bool_from_text(eyewear, ("no eyewear", "without eyewear", "none visible"))
    beard_present = infer_bool_from_text(facial_hair, ("no visible", "none", "clean-shaven"))

    full_name_norm = _normalize_name(rec.get("full_name_normalized") or rec.get("full_name"))

    md = {
        "vmid": _norm_str(rec.get("vmid")),
        "full_name": _norm_str(rec.get("full_name")),
        "full_name_normalized": full_name_norm,
        "name_tokens": full_name_norm.split() if full_name_norm else [],
        "profile_url": _norm_str(rec.get("profile_url")),
        "profile_image_s3_url": _norm_str(rec.get("profile_image_s3_url")),
        "headline": _norm_str(rec.get("headline")),
        "job_title": _norm_str(rec.get("job_title")),
        "location": _norm_str(rec.get("location")),
        "location_normalized": _norm_str(rec.get("location_normalized")),
        "school": _norm_str(rec.get("school")),
        "minimum_estimated_years_of_exp": rec.get("minimum_estimated_years_of_exp"),
        "eyewear_present": eyewear_present,
        "beard_present": beard_present,
        "tech_tokens": extract_tech_tokens(rec.get("headline")),
        "embedding_text": embedding_text,
    }
    return {k: v for k, v in md.items() if v not in (None, "", [])}


def stable_id(rec: Dict[str, Any]) -> str:
    return _norm_str(rec.get("vmid")) or _norm_str(rec.get("profile_url")) or "unknown"


def batched(items: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), n):
        yield items[i: i + n]


def get_pinecone_index(pc: Pinecone, index_name: str):
    desc = pc.describe_index(index_name)
    host = getattr(desc, "host", None)
    if host:
        return pc.Index(index_name, host=host)
    return pc.Index(index_name)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main() -> None:
    load_dotenv(PROJECT_DIR / ".env")

    parser = argparse.ArgumentParser(description="Build Pinecone index (Hybrid) from jsonl")
    parser.add_argument("--jsonl", type=str, default=str(DEFAULT_JSONL_PATH))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    # Load Secrets
    secrets = load_secrets()
    openai_key = secrets.get("OPENAI_API_KEY")
    pinecone_key = secrets.get("PINECONE_API_KEY")
    index_name = secrets.get("PINECONE_INDEX_NAME")
    namespace = secrets.get("PINECONE_NAMESPACE")

    if not openai_key or not pinecone_key or not index_name:
        raise RuntimeError("Missing API Keys (OPENAI/PINECONE) or Index Name.")

    # Load Data
    jsonl_path = Path(args.jsonl).expanduser().resolve()
    maybe_download_jsonl_from_s3(
        local_path=jsonl_path,
        bucket=secrets.get("S3_BUCKET"),
        key=secrets.get("S3_KEY")
    )

    records = load_jsonl(jsonl_path)
    if not records:
        raise RuntimeError("No records found.")

    ids, texts, metadatas = [], [], []
    for rec in records:
        emb_text = build_embedding_text(rec)
        ids.append(stable_id(rec))
        texts.append(emb_text)
        metadatas.append(build_metadata(rec, emb_text))

    if args.dry_run:
        print("Dry run complete. First record text preview:", texts[0][:100])
        return

    # Init Models
    embeddings = OpenAIEmbeddings(api_key=openai_key, model=args.embed_model)

    # --- HYBRID SEARCH: BM25 INIT ---
    print(f"[INFO] Fitting BM25 on {len(texts)} text chunks...")
    bm25 = BM25Encoder()
    bm25.fit(texts)

    bm25_path = DATA_DIR / "bm25_params.json"
    bm25.dump(str(bm25_path))
    print(f"[INFO] Saved BM25 params to {bm25_path}")

    # Connect Pinecone
    pc = Pinecone(api_key=pinecone_key)
    index = get_pinecone_index(pc, index_name)

    if args.clear:
        print(f"[WARN] Clearing namespace {namespace}...")
        try:
            index.delete(delete_all=True, namespace=namespace)
        except Exception:
            pass

    print(f"[INFO] Upserting {len(texts)} vectors (Hybrid: Dense + Sparse)...")

    # Upsert Loop
    for i, batch in enumerate(batched(list(range(len(texts))), args.batch_size)):
        batch_ids = [ids[k] for k in batch]
        batch_texts = [texts[k] for k in batch]
        batch_mds = [metadatas[k] for k in batch]

        # 1. Generate Dense Vectors
        dense_vecs = embeddings.embed_documents(batch_texts)

        # 2. Generate Sparse Vectors (BM25)
        sparse_vecs = bm25.encode_documents(batch_texts)

        # 3. Combine
        vectors = []
        for _id, dense, sparse, md in zip(batch_ids, dense_vecs, sparse_vecs, batch_mds):
            vectors.append({
                "id": _id,
                "values": dense,
                "sparse_values": sparse,  # <--- CRITICAL FOR HYBRID
                "metadata": md
            })

        index.upsert(vectors=vectors, namespace=namespace)
        print(f"[INFO] Batch {i + 1} upserted.")

    print("✅ Indexing Complete.")


if __name__ == "__main__":
    main()