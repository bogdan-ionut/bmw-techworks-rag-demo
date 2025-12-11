# app/main.py

import os
import json
from pathlib import Path
from typing import List, Dict, Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Fișierul tău local JSONL (ai deja 478 de rânduri acolo)
LOCAL_JSONL = DATA_DIR / "bmw_employees.jsonl"

# Opțional: dacă vrei să folosești S3
S3_BUCKET = os.getenv("S3_BUCKET")
S3_KEY = os.getenv("S3_KEY", "bmw_employees.jsonl")


def maybe_download_from_s3() -> None:
    """
    Dacă fișierul local NU există, încearcă să îl descarci din S3.
    Dacă nu ai credențiale / bucket, doar loghează un warning și continuă.
    """
    if LOCAL_JSONL.exists():
        return

    if not (S3_BUCKET and S3_KEY):
        print("[INFO] Local JSONL missing și S3 nu e configurat – sar peste download.")
        return

    try:
        s3 = boto3.client("s3")
        print(f"[INFO] Download din S3: bucket={S3_BUCKET}, key={S3_KEY}")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        s3.download_file(S3_BUCKET, S3_KEY, str(LOCAL_JSONL))
        print("[INFO] Download din S3 completat.")
    except (BotoCoreError, ClientError) as e:
        print(f"[WARN] Nu pot descărca din S3 ({e}); continui doar cu local dacă apare ulterior.")


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
    ]
    return "\n".join(parts)


def build_documents(records: List[Dict[str, Any]]) -> List[Document]:
    """Construiește Document-e LangChain cu metadata utilă pentru /sources."""
    docs: List[Document] = []
    for rec in records:
        content = record_to_text(rec)
        meta = {
            "full_name": rec.get("full_name"),
            "location": rec.get("location"),
            "job_title": rec.get("job_title"),
            "job_title_2": rec.get("job_title_2"),
            "profile_url": rec.get("profile_url"),
            "vmid": rec.get("vmid"),
        }
        docs.append(Document(page_content=content, metadata=meta))
    print(f"[INFO] Built {len(docs)} formatted Documents")
    return docs


def main() -> None:
    # 1. S3 (opțional) + verificare fișier local
    maybe_download_from_s3()
    if not LOCAL_JSONL.exists():
        raise FileNotFoundError(f"JSONL file not found at {LOCAL_JSONL}")

    # 2. Încărcăm JSONL
    records = load_jsonl(LOCAL_JSONL)

    # 3. Construim Document-ele
    docs = build_documents(records)

    # 4. Split în chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    splits = splitter.split_documents(docs)
    print(f"[INFO] After splitting: {len(splits)} chunks")

    # 5. Embeddings + Chroma
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-large",
    )

    vectordb = Chroma.from_documents(
        splits,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    vectordb.persist()
    print(f"✅ Chroma DB built & saved in {CHROMA_DIR}")


if __name__ == "__main__":
    main()
