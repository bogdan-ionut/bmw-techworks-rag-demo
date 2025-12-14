# app/main.py

import os
import json
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


# --- paths ---
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db")))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "profiles")

DOTENV_PATH = BASE_DIR / ".env"
if DOTENV_PATH.exists():
    load_dotenv(dotenv_path=DOTENV_PATH)
else:
    load_dotenv()

# Fișierul local JSONL
LOCAL_JSONL = DATA_DIR / "bmw_employees.jsonl"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")


def require_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY lipsește. Pune cheia în .env în root-ul proiectului."
        )
    return key


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Citește JSONL linie cu linie și întoarce o listă de dict-uri."""
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError as e:
                print(f"[WARN] Sar peste linie invalidă ({e})")
    print(f"[INFO] Loaded {len(records)} JSONL records")
    return records


def record_to_text(rec: Dict[str, Any]) -> str:
    """Transformă un record brut într-un blob de text bogat pentru RAG."""
    parts = [
        f"Employee: {rec.get('full_name', 'N/A')}",
        f"Headline: {rec.get('headline', 'N/A')}",
        "",
        "# Current role",
        f"Job title: {rec.get('job_title', 'N/A')}",
        f"Job period: {rec.get('job_date_range', 'N/A')}",
        "",
        "# Previous role",
        f"Job title 2: {rec.get('job_title_2', 'N/A')}",
        f"Job period 2: {rec.get('job_date_range_2', 'N/A')}",
        "",
        "# Location",
        f"Location: {rec.get('location', 'N/A')}",
        "",
        "# Education",
        f"School 1: {rec.get('school', 'N/A')} – "
        f"{rec.get('school_degree', 'N/A')} – "
        f"{rec.get('school_date_range', 'N/A')}",
        f"School 2: {rec.get('school_2', 'N/A')} – "
        f"{rec.get('school_degree_2', 'N/A')} – "
        f"{rec.get('school_date_range_2', 'N/A')}",
        "",
        f"LinkedIn: {rec.get('profile_url', 'N/A')}",
        f"VMID: {rec.get('vmid', 'N/A')}",
    ]
    return "\n".join(parts)


def build_documents(records: List[Dict[str, Any]]) -> List[Document]:
    """
    Construiește Document-e LangChain cu metadata RICH pentru UI (/sources).
    """
    docs: List[Document] = []
    for rec in records:
        content = record_to_text(rec)

        stable_id = rec.get("vmid") or rec.get("profile_url") or rec.get("full_name") or "unknown"

        meta = {
            "id": stable_id,
            "source_id": stable_id,  # util pt UI
            "vmid": rec.get("vmid"),
            "full_name": rec.get("full_name"),
            "profile_url": rec.get("profile_url"),
            "profile_image_url": rec.get("profile_image_url"),
            "headline": rec.get("headline"),
            "location": rec.get("location"),
            "job_title": rec.get("job_title"),
            "job_date_range": rec.get("job_date_range"),
            "job_title_2": rec.get("job_title_2"),
            "job_date_range_2": rec.get("job_date_range_2"),
            "school": rec.get("school"),
            "school_degree": rec.get("school_degree"),
            "school_date_range": rec.get("school_date_range"),
            "school_2": rec.get("school_2"),
            "school_degree_2": rec.get("school_degree_2"),
            "school_date_range_2": rec.get("school_date_range_2"),
        }

        docs.append(Document(page_content=content, metadata=meta))

    print(f"[INFO] Built {len(docs)} formatted Documents")
    return docs


def main() -> None:
    require_openai_key()

    if not LOCAL_JSONL.exists():
        raise FileNotFoundError(f"JSONL file not found at {LOCAL_JSONL}")

    records = load_jsonl(LOCAL_JSONL)
    docs = build_documents(records)

    # NU splităm: 1 profil = 1 Document
    splits = docs
    print(f"[INFO] Without splitting: {len(splits)} full documents")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # IMPORTANT: folosim aceeași colecție ca API-ul
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=CHROMA_COLLECTION,
    )

    try:
        vectordb.persist()
    except Exception:
        pass

    print(f"✅ Chroma DB built & saved in {CHROMA_DIR}")
    print(f"✅ Collection: {CHROMA_COLLECTION}")
    try:
        count = vectordb._collection.count()  # type: ignore[attr-defined]
        print(f"✅ Vectors count: {count}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
