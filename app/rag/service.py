# app/rag/service.py
from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.rag.retrieve import retrieve_profiles
from app.rag.rerank import rerank_candidates
from app.rag.prompt import (
    PLANNER_SYSTEM_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    format_candidates_for_prompt,
    ALLOWED_FILTER_FIELDS,
)


GLASSES_POS = re.compile(r"\b(eyeglasses|eye-glasses|glasses|spectacles)\b", re.I)
GLASSES_NEG = re.compile(r"\b(no glasses|without glasses|no eyewear)\b", re.I)

BEARD_POS = re.compile(r"\b(beard|bearded|mustache|moustache)\b", re.I)
BEARD_NEG = re.compile(r"\b(clean[- ]shaven|no beard)\b", re.I)


def hard_filters_from_text(q: str) -> Tuple[Optional[Dict[str, Any]], str]:
    parts: List[Dict[str, Any]] = []
    semantic = q.strip()

    if GLASSES_NEG.search(semantic):
        parts.append({"eyewear_present": {"$eq": False}})
        semantic = GLASSES_NEG.sub("", semantic).strip()
    elif GLASSES_POS.search(semantic):
        parts.append({"eyewear_present": {"$eq": True}})
        semantic = GLASSES_POS.sub("", semantic).strip()

    if BEARD_NEG.search(semantic):
        parts.append({"beard_present": {"$eq": False}})
        semantic = BEARD_NEG.sub("", semantic).strip()
    elif BEARD_POS.search(semantic):
        parts.append({"beard_present": {"$eq": True}})
        semantic = BEARD_POS.sub("", semantic).strip()

    semantic = re.sub(r"\s+", " ", semantic).strip()

    if not parts:
        return None, semantic
    if len(parts) == 1:
        return parts[0], semantic
    return {"$and": parts}, semantic


def safe_json_loads(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(s[start : end + 1])
    raise ValueError("Could not parse JSON from model output.")


def merge_filters(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not a and not b:
        return None
    if a and not b:
        return a
    if b and not a:
        return b
    return {"$and": [a, b]}


def _sanitize_filter(flt: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Prevent Pinecone errors if planner outputs unknown metadata fields.
    Keeps only ALLOWED_FILTER_FIELDS (+ logical operators).
    """
    if not flt or not isinstance(flt, dict):
        return flt

    def keep(node: Any) -> Any:
        if isinstance(node, dict):
            out: Dict[str, Any] = {}
            for k, v in node.items():
                if k in ("$and", "$or"):
                    if isinstance(v, list):
                        vv = [keep(x) for x in v]
                        vv = [x for x in vv if x not in (None, {}, [])]
                        if vv:
                            out[k] = vv
                    continue
                if k.startswith("$"):
                    out[k] = v
                    continue
                if k in ALLOWED_FILTER_FIELDS:
                    out[k] = v
            return out
        return node

    cleaned = keep(flt)
    return cleaned if cleaned else None


def plan_query_with_llm(user_query: str) -> Dict[str, Any]:
    s = get_settings()

    llm = ChatOpenAI(
        api_key=s.openai_api_key,
        model=s.openai_model,
        temperature=0.0,
        max_tokens=600,
    )

    out = llm.invoke(
        [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=user_query),
        ]
    )
    plan = safe_json_loads(out.content)

    plan.setdefault("top_k", 30)
    plan.setdefault("rerank_top_n", s.cohere_rerank_top_n)
    plan.setdefault("semantic_query", user_query)
    plan.setdefault("filter", None)

    plan["filter"] = _sanitize_filter(plan.get("filter"))
    return plan


def generate_answer(user_query: str, candidates: List[Dict[str, Any]], filters_applied: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    s = get_settings()

    llm = ChatOpenAI(
        api_key=s.openai_api_key,
        model=s.openai_model,
        temperature=s.temperature,
        max_tokens=s.max_tokens,
    )

    user_msg = {
        "query": user_query,
        "filters_applied": filters_applied,
        "candidates_text": format_candidates_for_prompt(candidates),
    }

    out = llm.invoke(
        [
            SystemMessage(content=ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(user_msg, ensure_ascii=False)),
        ]
    )
    try:
        return safe_json_loads(out.content)
    except Exception:
        return {
            "answer": (out.content or "").strip() or "I couldn't generate an answer.",
            "filters_applied": filters_applied,
            "top_matches": [],
        }


def rag_search(user_query: str) -> Dict[str, Any]:
    s = get_settings()

    hard_filter, cleaned_semantic = hard_filters_from_text(user_query)
    plan = plan_query_with_llm(user_query)

    plan_filter = plan.get("filter")
    semantic_query = (plan.get("semantic_query") or cleaned_semantic or user_query).strip()

    merged_filter = _sanitize_filter(merge_filters(hard_filter, plan_filter))

    top_k = int(plan.get("top_k", 30))
    rerank_top_n = int(plan.get("rerank_top_n", s.cohere_rerank_top_n))

    docs = retrieve_profiles(
        query=semantic_query,
        top_k=top_k,
        metadata_filter=merged_filter,
    )

    candidates: List[Dict[str, Any]] = []
    for d in docs:
        md = dict(d.metadata or {})
        candidates.append(
            {
                "id": md.get("_id"),
                "score": md.get("_score"),
                "text": d.page_content,
                "metadata": md,
            }
        )

    # rerank (optional)
    if s.cohere_api_key:
        candidates = rerank_candidates(
            user_query,
            candidates,
            cohere_api_key=s.cohere_api_key,
            model=s.cohere_rerank_model,
            top_n=rerank_top_n,
        )
    else:
        candidates = candidates[:rerank_top_n]

    answer = generate_answer(user_query, candidates, merged_filter)

    return {
        "query": user_query,
        "semantic_query": semantic_query,
        "filters_applied": merged_filter,
        "top_k": top_k,
        "rerank_top_n": rerank_top_n,
        "results": candidates,
        "answer": answer,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="End-to-end RAG search (planner + filter + retrieve + rerank + answer)")
    parser.add_argument("query", type=str)
    args = parser.parse_args()

    out = rag_search(args.query)
    print(json.dumps(out["answer"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
