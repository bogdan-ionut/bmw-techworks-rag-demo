# app/rag/service.py
from __future__ import annotations

import argparse
import json
import asyncio
import copy
import re
from typing import Any, Dict, List, Optional, Tuple
from functools import partial

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
    sanitize_filter,
    CITY_NORMALIZATION_MAP,
)


SAFE_MIN_LLM_MAX_TOKENS = 1200


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

    # 1. Try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2. Try to find markdown code blocks
    code_blocks = re.findall(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if code_blocks:
        for block in reversed(code_blocks):
            try:
                return json.loads(block.strip())
            except Exception:
                pass

    # 3. Find all top-level JSON objects using raw_decode
    decoder = json.JSONDecoder()
    pos = 0
    candidates = []
    while True:
        match = s.find('{', pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(s[match:])
            candidates.append(result)
            pos = match + index
        except ValueError:
            pos = match + 1

    if candidates:
        dicts = [c for c in candidates if isinstance(c, dict)]
        if dicts:
            # Check for keys likely in our app
            expected_keys = {"answer", "semantic_query", "filter", "key_patterns", "top_matches", "top_k"}
            for c in reversed(dicts):
                if any(k in c for k in expected_keys):
                    return c
            # Fallback to last dict
            return dicts[-1]

    # 4. Last resort: simple start/end
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except Exception:
            pass

    # 5. Try cleaning trailing commas
    s_cleaned = re.sub(r",\s*([}\]])", r"\1", s)
    if s_cleaned != s:
        try:
            return safe_json_loads(s_cleaned)
        except RecursionError:
            pass
        except ValueError:
            pass

    raise ValueError("Could not parse JSON from model output.")


def merge_filters(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not a and not b:
        return None
    if a and not b:
        return a
    if b and not a:
        return b
    return {"$and": [a, b]}


def normalize_location_filter(f: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Normalize 'location_normalized' values using CITY_NORMALIZATION_MAP.
    """
    if not f:
        return f

    new_f = copy.deepcopy(f)

    def _visit(node: Any) -> Any:
        if not isinstance(node, dict):
            return node

        # If we see location_normalized: {...}
        if "location_normalized" in node:
            cond = node["location_normalized"]
            if isinstance(cond, dict):
                # Handle $eq
                if "$eq" in cond and isinstance(cond["$eq"], str):
                    raw = cond["$eq"].lower().strip()
                    cond["$eq"] = CITY_NORMALIZATION_MAP.get(raw, raw)
                # Handle $in
                if "$in" in cond and isinstance(cond["$in"], list):
                    normalized_list = []
                    for item in cond["$in"]:
                        if isinstance(item, str):
                            raw = item.lower().strip()
                            normalized_list.append(CITY_NORMALIZATION_MAP.get(raw, raw))
                        else:
                            normalized_list.append(item)
                    cond["$in"] = normalized_list

        # Recurse logic
        for k, v in node.items():
            if k in ("$and", "$or") and isinstance(v, list):
                node[k] = [_visit(item) for item in v]

        return node

    return _visit(new_f)


def expand_name_filter(f: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Relax 'full_name_normalized' strict equality to allow case variations.
    This helps if the DB contains 'Ionut Buraga' but the planner output 'ionut buraga'.
    """
    if not f:
        return f

    new_f = copy.deepcopy(f)

    def _visit(node: Any) -> Any:
        if not isinstance(node, dict):
            return node

        # If we see full_name_normalized: {$eq: "..."}
        if "full_name_normalized" in node:
            cond = node["full_name_normalized"]
            if isinstance(cond, dict) and "$eq" in cond:
                val = cond["$eq"]
                if isinstance(val, str) and val.strip():
                    base = val.strip()
                    # Generate common casing variations
                    variations = list({
                        base.lower(),
                        base.title(),
                        base.upper(),
                        base
                    })
                    # Replace strict match with set inclusion
                    node["full_name_normalized"] = {"$in": variations}

        # If we see name_tokens: ensure query values are lowercase
        if "name_tokens" in node:
            cond = node["name_tokens"]
            # e.g. {$in: ["Ionut", "IONUT"]} -> {$in: ["ionut"]}
            if isinstance(cond, dict):
                for op in ("$in", "$nin", "$all"):
                    if op in cond and isinstance(cond[op], list):
                        cond[op] = [str(x).lower() for x in cond[op] if x]
                for op in ("$eq", "$ne"):
                    if op in cond and isinstance(cond[op], str):
                        cond[op] = cond[op].lower()

        # Recurse logic
        for k, v in node.items():
            if k in ("$and", "$or") and isinstance(v, list):
                node[k] = [_visit(item) for item in v]

        return node

    return _visit(new_f)


def _classify_query_complexity(query: str) -> str:
    """
    Classifies the query complexity to determine if we can skip the LLM planner.
    """
    q_lower = query.lower()

    # 1. Visual keywords
    visual_keywords = ["glasses", "photo", "beard"]
    if any(k in q_lower for k in visual_keywords):
        return "VISUAL"

    # 2. Check length and complex connectors
    words = q_lower.split()
    is_short = len(words) < 6

    # Connectors: "and", "or", "who", "experience"
    connectors = [r"\band\b", r"\bor\b", r"\bwho\b", r"\bexperience\b"]
    has_complex = False
    for pattern in connectors:
        if re.search(pattern, q_lower):
            has_complex = True
            break

    if is_short and not has_complex:
        return "SIMPLE"

    return "COMPLEX"


async def plan_query_with_llm(user_query: str) -> Dict[str, Any]:
    s = get_settings()

    llm = ChatOpenAI(
        api_key=s.openai_api_key,
        model=s.openai_model,
        temperature=0.0,
        max_tokens=1500,
    )

    out = await llm.ainvoke(
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

    plan["filter"] = sanitize_filter(plan.get("filter"))
    return plan


async def generate_answer(user_query: str, candidates: List[Dict[str, Any]], filters_applied: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    s = get_settings()

    llm = ChatOpenAI(
        api_key=s.openai_api_key,
        model=s.openai_model,
        temperature=s.temperature,
        max_tokens=max(s.max_tokens, SAFE_MIN_LLM_MAX_TOKENS),
    )

    user_msg = {
        "query": user_query,
        "filters_applied": filters_applied,
        "candidates_text": format_candidates_for_prompt(candidates),
    }

    out = await llm.ainvoke(
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


async def rag_search_async(user_query: str, use_planner: bool = True) -> Dict[str, Any]:
    s = get_settings()

    complexity = _classify_query_complexity(user_query)
    should_use_planner = use_planner and (complexity != "SIMPLE")

    hard_filter, cleaned_semantic = hard_filters_from_text(user_query)

    if should_use_planner:
        plan = await plan_query_with_llm(user_query)
        plan_filter = plan.get("filter")
        semantic_query = (plan.get("semantic_query") or cleaned_semantic or user_query).strip()
        top_k = int(plan.get("top_k", 30))
        rerank_top_n = int(plan.get("rerank_top_n", s.cohere_rerank_top_n))
    else:
        # Planner disabled: use raw query + basic regex filters
        plan_filter = None
        semantic_query = cleaned_semantic or user_query
        top_k = 30
        rerank_top_n = s.cohere_rerank_top_n

    merged_filter = sanitize_filter(merge_filters(hard_filter, plan_filter))
    merged_filter = normalize_location_filter(merged_filter)
    merged_filter = expand_name_filter(merged_filter)

    loop = asyncio.get_running_loop()
    docs = await loop.run_in_executor(
        None,
        partial(
            retrieve_profiles,
            query=semantic_query,
            top_k=top_k,
            metadata_filter=merged_filter,
        )
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
        candidates = await loop.run_in_executor(
            None,
            partial(
                rerank_candidates,
                user_query,
                candidates,
                cohere_api_key=s.cohere_api_key,
                model=s.cohere_rerank_model,
                top_n=rerank_top_n,
            )
        )
    else:
        candidates = candidates[:rerank_top_n]

    answer = await generate_answer(user_query, candidates, merged_filter)

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
    parser.add_argument("--no-planner", action="store_true", help="Disable LLM query planner")
    args = parser.parse_args()

    out = asyncio.run(rag_search_async(args.query, use_planner=not args.no_planner))
    print(json.dumps(out["answer"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
