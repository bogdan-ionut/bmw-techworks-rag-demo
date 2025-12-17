# app/rag/prompt.py

from __future__ import annotations

import json
from typing import Any, Dict, List

ALLOWED_FILTER_FIELDS = {
    "eyewear_present",
    "beard_present",
    "location_normalized",
    "location",
    "job_title",
    "minimum_estimated_years_of_exp",
    "tech_tokens",
    "image_search_tags",
    "full_name_normalized",
}

PLANNER_SYSTEM_PROMPT = """\
You are a query planner for a recruiting search engine.

Your job:
1) Extract HARD FILTERS that can be applied on metadata (boolean, exact match, numeric ranges).
2) Produce a clean semantic_query for vector search.

Rules:
- Output MUST be valid JSON only (no markdown).
- Use only these filter fields when possible:
  eyewear_present (bool)
  beard_present (bool)
  location_normalized (string)
  minimum_estimated_years_of_exp (number)
  tech_tokens (list of strings)
  image_search_tags (list of strings)

- If the user asks for eyeglasses/glasses/spectacles -> eyewear_present=true.
- If the user asks for "no glasses" -> eyewear_present=false.
- If the user asks for beard -> beard_present=true; clean-shaven -> beard_present=false.
- If you are unsure, omit the filter (do not guess).

Filter format must be Pinecone-compatible:
- Equality: {"field": {"$eq": value}}
- In: {"field": {"$in": [..]}}
- Numeric: {"field": {"$gte": n}} etc
- Combine with {"$and":[...]} or {"$or":[...]} when needed.

Return schema:
{
  "semantic_query": "string",
  "filter": { ... } | null,
  "top_k": number,
  "rerank_top_n": number
}

Defaults:
- top_k: 30
- rerank_top_n: 8
"""

ANSWER_SYSTEM_PROMPT = """
You are a recruiting intelligence assistant for BMW TechWorks.

Given a search query and candidate profiles, provide:
1. A concise 2-3 sentence summary of what you found
2. Key patterns across top matches (skills, locations, experience levels)
3. Specific reasons why each top candidate matches
4. Any gaps or recommendations for the search

Be practical, data-driven, and focused on actionable insights for recruiters.
"""


def format_candidates_for_prompt(candidates: List[Dict[str, Any]], max_chars: int = 1200) -> str:
    blocks: List[str] = []
    for i, c in enumerate(candidates, start=1):
        md = c.get("metadata", {}) or {}
        text = (c.get("text") or "").strip()
        score = c.get("score")

        # keep each candidate short to stay under context limits
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        blocks.append(
            "\n".join(
                [
                    f"Candidate #{i}",
                    f"id: {c.get('id')}",
                    f"score: {score}",
                    f"full_name: {md.get('full_name')}",
                    f"profile_url: {md.get('profile_url')}",
                    f"image_url: {md.get('profile_image_s3_url')}",
                    "text:",
                    text,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
