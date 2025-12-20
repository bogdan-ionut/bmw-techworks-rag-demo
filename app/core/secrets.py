# app/core/secrets.py
from __future__ import annotations

import json
import os
import logging
from functools import lru_cache
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

# Optional: load .env in local dev
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v.strip() if v else default


def _env_fallback() -> Dict[str, str]:
    # Keep backward-compat names
    pinecone_index = _env("PINECONE_INDEX_NAME") or _env("PINECONE_INDEX")
    pinecone_ns = _env("PINECONE_NAMESPACE") or "profiles"

    return {
        "OPENAI_API_KEY": _env("OPENAI_API_KEY"),
        "GOOGLE_API_KEY": _env("GOOGLE_API_KEY"),
        "PINECONE_API_KEY": _env("PINECONE_API_KEY"),
        "COHERE_API_KEY": _env("COHERE_API_KEY"),
        "PINECONE_INDEX_NAME": pinecone_index,
        "PINECONE_NAMESPACE": pinecone_ns,
        "DEMO_PASSWORD": _env("DEMO_PASSWORD"),
        # --- LANGCHAIN / LANGSMITH KEYS ---
        "LANGCHAIN_TRACING_V2": _env("LANGCHAIN_TRACING_V2"),
        "LANGCHAIN_ENDPOINT": _env("LANGCHAIN_ENDPOINT"),
        "LANGCHAIN_API_KEY": _env("LANGCHAIN_API_KEY"),
        "LANGCHAIN_PROJECT": _env("LANGCHAIN_PROJECT"),
    }


@lru_cache(maxsize=1)
def load_secrets(secret_id: Optional[str] = None, region: Optional[str] = None) -> Dict[str, str]:
    """
    Loads secrets from AWS Secrets Manager (if configured), otherwise from env/.env.

    Supports env names:
      - AWS_SECRET_ID (preferred) OR AWS_SECRET_NAME (legacy)
      - AWS_REGION / AWS_DEFAULT_REGION
    """
    sid = (secret_id or _env("AWS_SECRET_ID") or _env("AWS_SECRET_NAME")).strip()
    reg = (region or _env("AWS_REGION") or _env("AWS_DEFAULT_REGION") or "us-east-1").strip()

    fb = _env_fallback()

    if not sid:
        logger.info("AWS secret not configured (AWS_SECRET_ID/AWS_SECRET_NAME not set) -> using env/.env only.")
        return fb

    try:
        session = boto3.session.Session(region_name=reg)
        client = session.client("secretsmanager")
        resp = client.get_secret_value(SecretId=sid)
        secret_str = resp.get("SecretString") or "{}"
        obj: Any = json.loads(secret_str)

        if not isinstance(obj, dict):
            raise ValueError("SecretString is not a JSON object")

        logger.info("Loaded secrets from AWS Secrets Manager: %s", sid)

        def pick(key: str) -> str:
            # prefer ENV if explicitly set; otherwise secret; otherwise fallback
            envv = _env(key)
            if envv:
                return envv
            return str(obj.get(key) or fb.get(key) or "")

        # Keep backward compat for index name env var
        pinecone_index = _env("PINECONE_INDEX_NAME") or _env("PINECONE_INDEX") or str(obj.get("PINECONE_INDEX_NAME") or "")

        return {
            "OPENAI_API_KEY": pick("OPENAI_API_KEY"),
            "GOOGLE_API_KEY": pick("GOOGLE_API_KEY"),
            "PINECONE_API_KEY": pick("PINECONE_API_KEY"),
            "COHERE_API_KEY": pick("COHERE_API_KEY"),
            "PINECONE_INDEX_NAME": pinecone_index or fb.get("PINECONE_INDEX_NAME", ""),
            "PINECONE_NAMESPACE": pick("PINECONE_NAMESPACE") or "profiles",
            "DEMO_PASSWORD": pick("DEMO_PASSWORD"),
            # --- LANGCHAIN / LANGSMITH KEYS ---
            "LANGCHAIN_TRACING_V2": pick("LANGCHAIN_TRACING_V2"),
            "LANGCHAIN_ENDPOINT": pick("LANGCHAIN_ENDPOINT"),
            "LANGCHAIN_API_KEY": pick("LANGCHAIN_API_KEY"),
            "LANGCHAIN_PROJECT": pick("LANGCHAIN_PROJECT"),
        }

    except (NoCredentialsError, BotoCoreError, ClientError, ValueError, json.JSONDecodeError) as e:
        logger.warning("Cannot read AWS secret (%s) -> falling back to env/.env", e)
        return fb