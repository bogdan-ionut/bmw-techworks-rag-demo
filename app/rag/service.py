# app/rag/service.py
from __future__ import annotations

import argparse
import json
import asyncio
import copy
import re
import time
from typing import Any, Dict, List, Optional, Tuple, AsyncGenerator
from functools import partial

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.rag.retrieve import retrieve_profiles
from app.rag.rerank import rerank_candidates
from app.rag.prompt import (
    PLANNER_SYSTEM_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    ANSWER_STREAM_SYSTEM_PROMPT,
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
            return json.loads(s[start: end + 1])
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
    try:
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
    except Exception:
        # Safety net: Fallback to full LLM processing on any error
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

    plan.setdefault("top_k", 60)
    plan.setdefault("rerank_top_n", 15)
    plan.setdefault("semantic_query", user_query)
    plan.setdefault("filter", None)

    plan["filter"] = sanitize_filter(plan.get("filter"))
    return plan


async def generate_answer(user_query: str, candidates: List[Dict[str, Any]],
                          filters_applied: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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


def _matches_filter(metadata: Dict[str, Any], filter_dict: Optional[Dict[str, Any]]) -> bool:
    """
    Checks if a candidate's metadata matches the MongoDB-style filter.
    Supports a subset of operators used in this app ($eq, $in, $gte, $and, $or).
    """
    if not filter_dict:
        return True

    # Recursive check
    for k, v in filter_dict.items():
        if k == "$and":
            if isinstance(v, list) and not all(_matches_filter(metadata, cond) for cond in v):
                return False
        elif k == "$or":
            if isinstance(v, list) and not any(_matches_filter(metadata, cond) for cond in v):
                return False
        elif k.startswith("$"):
            # Ignore other top-level operators if any
            pass
        else:
            # Field check
            val = metadata.get(k)

            if isinstance(v, dict):
                # Operator check
                for op, op_val in v.items():
                    if op == "$eq":
                        if val != op_val: return False
                    elif op == "$ne":
                        if val == op_val: return False
                    elif op == "$in":
                        if not isinstance(op_val, list): return False
                        if isinstance(val, list):
                            # List intersection
                            if not any(x in op_val for x in val): return False
                        else:
                            if val not in op_val: return False
                    elif op == "$nin":
                        if not isinstance(op_val, list): return False
                        if isinstance(val, list):
                            if any(x in op_val for x in val): return False
                        else:
                            if val in op_val: return False
                    elif op == "$gte":
                        if val is None or not isinstance(val, (int, float)) or val < op_val: return False
                    elif op == "$gt":
                        if val is None or not isinstance(val, (int, float)) or val <= op_val: return False
                    elif op == "$lte":
                        if val is None or not isinstance(val, (int, float)) or val > op_val: return False
                    elif op == "$lt":
                        if val is None or not isinstance(val, (int, float)) or val >= op_val: return False
            else:
                # Implicit equality
                if val != v: return False

    return True


async def prepare_search_context(
        user_query: str,
        explicit_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 60,
        rerank_top_n: int = 15,
        use_planner: bool = True
) -> Dict[str, Any]:
    """
    Executes the retrieval, planning, filtering, and reranking steps.
    Returns the context needed for answer generation.
    """
    s = get_settings()
    start_time = time.time()
    loop = asyncio.get_running_loop()

    complexity = _classify_query_complexity(user_query)
    should_use_planner = use_planner and (complexity != "SIMPLE")

    # 1. Speculative Retrieval (Parallel Task)
    speculative_filter = copy.deepcopy(explicit_filter)
    speculative_filter = normalize_location_filter(speculative_filter)
    speculative_filter = expand_name_filter(speculative_filter)

    task_retrieval = loop.run_in_executor(
        None,
        partial(
            retrieve_profiles,
            query=user_query,
            top_k=top_k,
            metadata_filter=speculative_filter,
        )
    )

    # 2. Planner (Parallel Task)
    task_planner = None
    if should_use_planner:
        task_planner = asyncio.create_task(plan_query_with_llm(user_query))

    # Wait for both
    docs = []
    plan = {}
    if task_planner:
        retrieval_res, planner_res = await asyncio.gather(task_retrieval, task_planner)
        docs = retrieval_res
        plan = planner_res
    else:
        docs = await task_retrieval
        plan = {}

    t1 = time.time()
    retrieval_sec = t1 - start_time

    # 3. Process Filters & In-Memory Filtering
    hard_filter, cleaned_semantic = hard_filters_from_text(user_query)
    plan_filter = plan.get("filter")
    semantic_query = (plan.get("semantic_query") or cleaned_semantic or user_query).strip()

    merged_filter = sanitize_filter(merge_filters(merge_filters(hard_filter, plan_filter), explicit_filter))
    merged_filter = normalize_location_filter(merged_filter)
    merged_filter = expand_name_filter(merged_filter)

    candidates: List[Dict[str, Any]] = []
    for d in docs:
        md = dict(d.metadata or {})
        score = md.get("_score")
        if score is None:
            score = 0.0

        candidates.append(
            {
                "id": md.get("_id"),
                "score": score,
                "text": d.page_content,
                "metadata": md,
            }
        )

    # Apply In-Memory Filtering
    candidates = [c for c in candidates if _matches_filter(c["metadata"], merged_filter)]

    # Rerank
    rerank_sec = 0.0
    if s.cohere_api_key and candidates:
        t_rerank_start = time.time()
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
        rerank_sec = time.time() - t_rerank_start
    else:
        candidates = candidates[:rerank_top_n]

    return {
        "user_query": user_query,
        "semantic_query": semantic_query,
        "filters_applied": merged_filter,
        "candidates": candidates,
        "complexity": complexity,
        "metrics": {
            "start_time": start_time,
            "retrieval_sec": retrieval_sec,
            "rerank_sec": rerank_sec,
        }
    }


async def rag_search_async(
        user_query: str,
        explicit_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 60,
        rerank_top_n: int = 15,
        llm_top_n: int = 8,
        use_planner: bool = True,
        with_llm: bool = True
) -> Dict[str, Any]:
    # Use helper
    ctx = await prepare_search_context(
        user_query, explicit_filter, top_k, rerank_top_n, use_planner
    )

    candidates = ctx["candidates"]
    metrics = ctx["metrics"]

    answer = None
    llm_sec = 0.0

    if with_llm:
        # Always generate AI answer if requested, regardless of complexity
        llm_candidates = candidates[:llm_top_n]
        t_llm_start = time.time()
        answer = await generate_answer(user_query, llm_candidates, ctx["filters_applied"])
        llm_sec = time.time() - t_llm_start
    else:
        answer = {
            "answer": "",
            "filters_applied": ctx["filters_applied"],
            "top_matches": []
        }

    return {
        "query": user_query,
        "semantic_query": ctx["semantic_query"],
        "filters_applied": ctx["filters_applied"],
        "top_k": top_k,
        "rerank_top_n": rerank_top_n,
        "results": candidates,
        "answer": answer,
        "latency_sec": time.time() - metrics["start_time"],
        "retrieval_sec": metrics["retrieval_sec"],
        "rerank_sec": metrics["rerank_sec"],
        "llm_sec": llm_sec,
        "query_type": ctx["complexity"]
    }


async def generate_answer_stream(
        user_query: str,
        candidates: List[Dict[str, Any]],
        filters_applied: Optional[Dict[str, Any]]
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Generates answer stream tokens and final metadata.
    Yields:
      {"type": "token", "content": "..."}
      {"type": "metadata", "content": dict}
    """
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

    stream = llm.astream(
        [
            SystemMessage(content=ANSWER_STREAM_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(user_msg, ensure_ascii=False)),
        ]
    )

    delimiter = "###METADATA###"
    buffer = ""
    metadata_mode = False

    async for chunk in stream:
        token = chunk.content
        if not metadata_mode:
            buffer += token
            if delimiter in buffer:
                # Split buffer
                pre, post = buffer.split(delimiter, 1)
                if pre:
                    yield {"type": "token", "content": pre}
                metadata_mode = True
                buffer = post  # Start accumulating JSON
            else:
                # Safety: Check if buffer ends with a partial delimiter.
                # If not, we can flush safe parts.
                # Simplification: Only hold if buffer ends with partial delimiter.
                # But delimiter is long.
                # Let's keep it simple: If buffer is very long and no delimiter, yield.
                if len(buffer) > len(delimiter) * 2:
                    safe_len = len(buffer) - len(delimiter)
                    yield {"type": "token", "content": buffer[:safe_len]}
                    buffer = buffer[safe_len:]
        else:
            buffer += token

    # Stream finished
    if not metadata_mode:
        # Delimiter never found? Just yield buffer as text.
        if buffer:
            yield {"type": "token", "content": buffer}
        # And empty metadata
        yield {"type": "metadata", "content": {"top_matches": []}}
    else:
        # Parse JSON from buffer
        try:
            meta = safe_json_loads(buffer)
        except Exception:
            meta = {"top_matches": []}
        yield {"type": "metadata", "content": meta}


async def rag_search_stream_generator(
        user_query: str,
        explicit_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 60,
        rerank_top_n: int = 15,
        llm_top_n: int = 5,
        use_planner: bool = True
) -> AsyncGenerator[str, None]:
    """
    Orchestrates RAG and yields SSE events.
    """
    # 1. Prepare context
    ctx = await prepare_search_context(
        user_query, explicit_filter, top_k, rerank_top_n, use_planner
    )

    candidates = ctx["candidates"]
    metrics = ctx["metrics"]
    llm_candidates = candidates[:llm_top_n]

    # --- NO COMPLEXITY CHECK: Always stream via LLM ---

    # 3. LLM Stream
    t_llm_start = time.time()
    async for chunk in generate_answer_stream(user_query, llm_candidates, ctx["filters_applied"]):
        if chunk["type"] == "token":
            safe_content = json.dumps(chunk["content"])
            yield f"data: {safe_content}\n\n"

        elif chunk["type"] == "metadata":
            metrics["llm_sec"] = time.time() - t_llm_start

            # Construct final metadata
            final_payload = {
                "answer": {"top_matches": chunk["content"].get("top_matches", [])},
                "candidates": candidates,  # needed for hydration
                "metrics": metrics,
                "filters_applied": ctx["filters_applied"]
            }
            yield f"event: metadata\ndata: {json.dumps(final_payload, default=str)}\n\n"


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end RAG search (planner + filter + retrieve + rerank + answer)")
    parser.add_argument("query", type=str)
    parser.add_argument("--no-planner", action="store_true", help="Disable LLM query planner")
    args = parser.parse_args()

    out = asyncio.run(rag_search_async(args.query, use_planner=not args.no_planner))
    print(json.dumps(out["answer"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()