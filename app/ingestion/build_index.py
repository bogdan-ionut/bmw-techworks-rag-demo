# app/ingestion/build_index.py
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from app.core.secrets import load_secrets

try:
    # Pinecone SDK v3+
    from pinecone import Pinecone
except Exception as e:  # pragma: no cover
    raise RuntimeError("Missing dependency: pinecone. Install with: pip install pinecone") from e

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except Exception:
    boto3 = None  # optional (S3 fallback)


# -----------------------------------------------------------------------------
# Paths / defaults
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[2]  # .../bmw-techworks-rag-demo
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_JSONL_PATH = DATA_DIR / "bmw_employees.jsonl"

DEFAULT_AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
DEFAULT_AWS_SECRET_NAME = os.getenv("AWS_SECRET_NAME", "bmw-techworks-rag-demo/secrets")
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

DEFAULT_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "64"))


# -----------------------------------------------------------------------------
# Optional: S3 download if local JSONL missing
# -----------------------------------------------------------------------------
def maybe_download_jsonl_from_s3(local_path: Path, bucket: str, key: str) -> None:
    """
    Optional fallback: if local JSONL is missing, try downloading from S3.
    Requires AWS creds too. If boto3 isn't installed, this just does nothing.
    """
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
# JSONL loading
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
                else:
                    print(f"[WARN] Line {i}: JSON is not an object; skipping.")
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {i}: invalid JSON ({e}); skipping.")
    print(f"[INFO] Loaded {len(records)} records from {path}")
    return records


# -----------------------------------------------------------------------------
# Helpers: flatten + embedding text
# -----------------------------------------------------------------------------
def _norm_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _safe_get(d: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _as_list_of_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    s = str(x).strip()
    return [s] if s else []


def extract_tech_tokens(headline: str) -> List[str]:
    """
    Heuristic: split headline on separators to get "skills-ish" tokens.
    Example: "Software Engineer @ BMW | React.js, React Native, Vue.js, Expo"
    """
    h = headline or ""
    parts: List[str] = []
    for chunk in re.split(r"[|]", h):
        parts.extend([p.strip() for p in chunk.split(",") if p.strip()])

    noise = {"@", "at", "bmw", "techworks", "romania"}
    tokens: List[str] = []
    for p in parts:
        p2 = re.sub(r"\s+", " ", p).strip()
        p2_low = p2.lower()
        if not p2 or p2_low in noise:
            continue
        if 2 <= len(p2) <= 60:
            tokens.append(p2)

    seen = set()
    out = []
    for t in tokens:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            out.append(t)
    return out[:40]


def build_embedding_text(rec: Dict[str, Any]) -> str:
    """
    Text embedded in Pinecone (maximize semantic retrieval):
    - job titles, headline, location, education
    - image caption summary + appearance/clothing tags (for "with eyeglasses", etc.)
    """
    full_name = _norm_str(rec.get("full_name"))
    headline = _norm_str(rec.get("headline"))
    job_title = _norm_str(rec.get("job_title"))
    job_range = _norm_str(rec.get("job_date_range"))
    job_title_2 = _norm_str(rec.get("job_title_2"))
    job_range_2 = _norm_str(rec.get("job_date_range_2"))
    location = _norm_str(rec.get("location"))

    school = _norm_str(rec.get("school"))
    school_degree = _norm_str(rec.get("school_degree"))
    school_range = _norm_str(rec.get("school_date_range"))
    school_2 = _norm_str(rec.get("school_2"))
    school_degree_2 = _norm_str(rec.get("school_degree_2"))
    school_range_2 = _norm_str(rec.get("school_date_range_2"))

    img_summary = _norm_str(_safe_get(rec, ["image_caption", "summary"]))
    hair = _norm_str(_safe_get(rec, ["image_caption", "appearance", "hair"]))
    facial_hair = _norm_str(_safe_get(rec, ["image_caption", "appearance", "facial_hair"]))
    eyewear = _norm_str(_safe_get(rec, ["image_caption", "appearance", "eyewear"]))
    expression = _norm_str(_safe_get(rec, ["image_caption", "appearance", "expression"]))
    visible_features = _as_list_of_str(_safe_get(rec, ["image_caption", "appearance", "visible_features"]))

    clothing_style = _norm_str(_safe_get(rec, ["image_caption", "clothing", "style"]))
    clothing_items = _as_list_of_str(_safe_get(rec, ["image_caption", "clothing", "items"]))
    accessories = _as_list_of_str(_safe_get(rec, ["image_caption", "clothing", "accessories"]))

    background = _norm_str(_safe_get(rec, ["image_caption", "image_composition", "background"]))
    search_tags = _as_list_of_str(_safe_get(rec, ["image_caption", "search_tags"]))

    tech_tokens = extract_tech_tokens(headline)

    lines: List[str] = []
    if full_name:
        lines.append(f"Name: {full_name}")
    if headline:
        lines.append(f"Headline: {headline}")
    if tech_tokens:
        lines.append("Tech/keywords: " + ", ".join(tech_tokens))

    if job_title or job_range:
        lines.append(f"Current role: {job_title} ({job_range})".strip())
    if job_title_2 or job_range_2:
        lines.append(f"Previous role: {job_title_2} ({job_range_2})".strip())

    if location:
        lines.append(f"Location: {location}")

    edu1 = " – ".join([x for x in [school, school_degree, school_range] if x])
    edu2 = " – ".join([x for x in [school_2, school_degree_2, school_range_2] if x])
    if edu1:
        lines.append(f"Education: {edu1}")
    if edu2:
        lines.append(f"Education: {edu2}")

    # image semantics
    if img_summary:
        lines.append(f"Photo summary: {img_summary}")
    if eyewear:
        lines.append(f"Eyewear: {eyewear}")
    if facial_hair:
        lines.append(f"Facial hair: {facial_hair}")
    if hair:
        lines.append(f"Hair: {hair}")
    if expression:
        lines.append(f"Expression: {expression}")
    if visible_features:
        lines.append("Visible features: " + ", ".join(visible_features))
    if clothing_style:
        lines.append(f"Clothing style: {clothing_style}")
    if clothing_items:
        lines.append("Clothing items: " + ", ".join(clothing_items))
    if accessories:
        lines.append("Accessories: " + ", ".join(accessories))
    if background:
        lines.append(f"Background: {background}")
    if search_tags:
        lines.append("Search tags: " + ", ".join(search_tags))

    return "\n".join(lines).strip()


def infer_bool_from_text(s: str, negative_markers: Tuple[str, ...]) -> Optional[bool]:
    if not s:
        return None
    low = s.lower()
    for m in negative_markers:
        if m in low:
            return False
    return True


def build_metadata(rec: Dict[str, Any], embedding_text: str) -> Dict[str, Any]:
    """
    Pinecone metadata must be flat.
    Keep UI fields + filter-friendly fields + flattened image caption.
    Also store embedding_text for rerank/context building later.
    """
    headline = _norm_str(rec.get("headline"))
    tech_tokens = extract_tech_tokens(headline)

    eyewear = _norm_str(_safe_get(rec, ["image_caption", "appearance", "eyewear"]))
    facial_hair = _norm_str(_safe_get(rec, ["image_caption", "appearance", "facial_hair"]))

    eyewear_present = infer_bool_from_text(
        eyewear, negative_markers=("no eyewear", "without eyewear", "none visible")
    )
    beard_present = infer_bool_from_text(
        facial_hair, negative_markers=("no visible", "none", "clean-shaven", "clean shaven")
    )

    md: Dict[str, Any] = {
        # identity / UI
        "vmid": _norm_str(rec.get("vmid")),
        "full_name": _norm_str(rec.get("full_name")),
        "full_name_normalized": _norm_str(rec.get("full_name_normalized")),
        "profile_url": _norm_str(rec.get("profile_url")),
        "profile_image_s3_url": _norm_str(rec.get("profile_image_s3_url")),
        "profile_image_s3_key": _norm_str(rec.get("profile_image_s3_key")),

        # work
        "headline": headline,
        "job_title": _norm_str(rec.get("job_title")),
        "job_date_range": _norm_str(rec.get("job_date_range")),
        "job_title_2": _norm_str(rec.get("job_title_2")),
        "job_date_range_2": _norm_str(rec.get("job_date_range_2")),

        # location
        "location": _norm_str(rec.get("location")),
        "location_normalized": _norm_str(rec.get("location_normalized")),

        # education
        "school": _norm_str(rec.get("school")),
        "school_degree": _norm_str(rec.get("school_degree")),
        "school_date_range": _norm_str(rec.get("school_date_range")),
        "school_2": _norm_str(rec.get("school_2")),
        "school_degree_2": _norm_str(rec.get("school_degree_2")),
        "school_date_range_2": _norm_str(rec.get("school_date_range_2")),

        # numeric
        "minimum_estimated_years_of_exp": rec.get("minimum_estimated_years_of_exp"),

        # image caption (flattened)
        "image_summary": _norm_str(_safe_get(rec, ["image_caption", "summary"])),
        "image_hair": _norm_str(_safe_get(rec, ["image_caption", "appearance", "hair"])),
        "image_facial_hair": facial_hair,
        "image_eyewear": eyewear,
        "image_expression": _norm_str(_safe_get(rec, ["image_caption", "appearance", "expression"])),
        "image_head_pose": _norm_str(_safe_get(rec, ["image_caption", "appearance", "head_pose"])),
        "image_background": _norm_str(_safe_get(rec, ["image_caption", "image_composition", "background"])),

        # filter booleans
        "eyewear_present": eyewear_present,
        "beard_present": beard_present,

        # lists
        "tech_tokens": tech_tokens,
        "image_search_tags": _as_list_of_str(_safe_get(rec, ["image_caption", "search_tags"])),
        "image_clothing_items": _as_list_of_str(_safe_get(rec, ["image_caption", "clothing", "items"])),
        "image_clothing_colors": _as_list_of_str(_safe_get(rec, ["image_caption", "clothing", "colors"])),
        "image_accessories": _as_list_of_str(_safe_get(rec, ["image_caption", "clothing", "accessories"])),

        # useful for rerank/LLM later
        "embedding_text": embedding_text,
    }

    # compact
    compact: Dict[str, Any] = {}
    for k, v in md.items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        if isinstance(v, list) and len(v) == 0:
            continue
        compact[k] = v
    return compact


def stable_id(rec: Dict[str, Any]) -> str:
    vmid = _norm_str(rec.get("vmid"))
    if vmid:
        return vmid
    url = _norm_str(rec.get("profile_url"))
    if url:
        return url
    name = _norm_str(rec.get("full_name"))
    return name or "unknown"


def batched(items: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def get_pinecone_index(pc: Pinecone, index_name: str):
    """
    Pinecone SDK v3: best practice is to use host from describe_index.
    """
    desc = pc.describe_index(index_name)
    host = getattr(desc, "host", None)
    if host:
        return pc.Index(index_name, host=host)
    return pc.Index(index_name)


def main() -> None:
    load_dotenv(PROJECT_DIR / ".env")

    parser = argparse.ArgumentParser(description="Build Pinecone index from bmw_employees.jsonl")
    parser.add_argument("--jsonl", type=str, default=str(DEFAULT_JSONL_PATH), help="Path to JSONL file")
    parser.add_argument("--secret-name", type=str, default=DEFAULT_AWS_SECRET_NAME, help="AWS Secrets Manager secret name")
    parser.add_argument("--region", type=str, default=DEFAULT_AWS_REGION, help="AWS region for Secrets Manager/S3")
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL, help="OpenAI embedding model")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for embedding/upsert")
    parser.add_argument("--clear", action="store_true", help="DANGER: delete all vectors in namespace before upsert")
    parser.add_argument("--dry-run", action="store_true", help="Do not upsert, only print a sample")
    args = parser.parse_args()

    # Ensure secrets loader sees your chosen region/secret
    if args.secret_name:
        os.environ["AWS_SECRET_NAME"] = args.secret_name
    if args.region:
        os.environ["AWS_REGION"] = args.region

    secrets = load_secrets()

    openai_key = (secrets.get("OPENAI_API_KEY") or "").strip()
    pinecone_key = (secrets.get("PINECONE_API_KEY") or "").strip()
    index_name = (secrets.get("PINECONE_INDEX_NAME") or "").strip()
    namespace = (secrets.get("PINECONE_NAMESPACE") or "").strip()

    if not openai_key:
        raise RuntimeError("Missing OPENAI_API_KEY (AWS secret or env).")
    if not pinecone_key:
        raise RuntimeError("Missing PINECONE_API_KEY (AWS secret or env).")
    if not index_name:
        raise RuntimeError("Missing PINECONE_INDEX_NAME (AWS secret or env).")
    if not namespace:
        raise RuntimeError("Missing PINECONE_NAMESPACE (AWS secret or env).")

    jsonl_path = Path(args.jsonl).expanduser().resolve()

    # Optional S3 fallback if local file missing
    maybe_download_jsonl_from_s3(
        local_path=jsonl_path,
        bucket=(secrets.get("S3_BUCKET") or "").strip(),
        key=(secrets.get("S3_KEY") or "").strip(),
    )

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found at: {jsonl_path}")

    records = load_jsonl(jsonl_path)
    if not records:
        raise RuntimeError("No records loaded from JSONL.")

    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for rec in records:
        _id = stable_id(rec)
        emb_text = build_embedding_text(rec)
        md = build_metadata(rec, embedding_text=emb_text)

        ids.append(_id)
        texts.append(emb_text)
        metadatas.append(md)

    if args.dry_run:
        print("\n=== DRY RUN SAMPLE ===")
        print("ID:", ids[0])
        print("\n--- EMBED TEXT ---\n", texts[0][:1200], "\n...")
        print("\n--- METADATA ---\n", json.dumps(metadatas[0], ensure_ascii=False, indent=2)[:1200], "\n...")
        print("\n(DRY RUN: no upsert performed)")
        return

    embeddings = OpenAIEmbeddings(api_key=openai_key, model=args.embed_model)

    pc = Pinecone(api_key=pinecone_key)
    index = get_pinecone_index(pc, index_name)

    if args.clear:
        print(f"[WARN] Clearing Pinecone namespace '{namespace}' in index '{index_name}'...")
        index.delete(delete_all=True, namespace=namespace)
        print("[INFO] Namespace cleared.")

    total = len(texts)
    print(f"[INFO] Upserting {total} vectors into Pinecone index='{index_name}', namespace='{namespace}'")

    for batch_idx, batch in enumerate(batched(list(range(total)), args.batch_size), start=1):
        batch_ids = [ids[i] for i in batch]
        batch_texts = [texts[i] for i in batch]
        batch_mds = [metadatas[i] for i in batch]

        vecs = embeddings.embed_documents(batch_texts)

        vectors = []
        for _id, values, md in zip(batch_ids, vecs, batch_mds):
            vectors.append({"id": _id, "values": values, "metadata": md})

        index.upsert(vectors=vectors, namespace=namespace)
        print(f"[INFO] Batch {batch_idx}: upserted {len(vectors)} vectors")

    print("✅ Done. Pinecone index updated successfully.")


if __name__ == "__main__":
    main()
