# app/rag/prompt.py

from __future__ import annotations

import json
from typing import Any, Dict, List

ALLOWED_FILTER_FIELDS = {
    "eyewear_present",
    "beard_present",
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
- Output MUST be valid JSON only (no markdown). Do NOT include conversational text, "thinking", or "explanation" outside the JSON object.
- Use only these filter fields when possible:
  eyewear_present (bool)
  beard_present (bool)
  minimum_estimated_years_of_exp (number)
  tech_tokens (list of strings)
  image_search_tags (list of strings)

- For visual attributes (e.g., "eyeglasses", "smiling", "beard"), use the boolean `eyewear_present`, `beard_present`, or `image_search_tags` filters.
- **IMPORTANT**: Remove visual descriptors from the `semantic_query` to avoid biasing the vector search. The `semantic_query` should only contain professional criteria (skills, roles, experience).
- **IMPORTANT**: Do NOT filter by location (city, country). Leave location terms in the `semantic_query` to allow partial matching.
- If a user asks for "python engineers with eyeglasses", the `semantic_query` should be "python engineers" and the filter should be `{"eyewear_present": {"$eq": true}}`.
- If a user asks for "people from Berlin", the `semantic_query` should be "people from Berlin" and the filter should be `null` (do not use location_normalized).
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

Your output MUST be a JSON object with the following schema:
{
  "answer": "A compelling, high-level executive summary of the talent pool found. Use 3-4 impactful sentences. Focus on the quality of the candidates, their shared strengths, and any notable location/skill trends.",
  "key_patterns": {
    "skills": ["..."],
    "locations": ["..."],
    "experience_levels": ["..."]
  },
  "top_matches": [
    {
      "id": "...",
      "full_name": "...",
      "profile_url": "...",
      "why_match": "A specific, data-driven justification (2-3 sentences). Mention the exact skills, companies, or projects that made this candidate stand out for the query. Be direct."
    }
  ]
}

- Output MUST be valid JSON only (no markdown). Do NOT include conversational text, "thinking", or "explanation" outside the JSON object.
- The "answer" field is your "elevator pitch" to the recruiter. Make it insightful.
- IMPORTANT: If the number of `top_matches` you return is less than the total number of candidate profiles reviewed, you MUST state this clearly in your "answer". For example: "From the 12 retrieved profiles, here is an analysis of the top 8 matches..."
- The "key_patterns" field identifies common themes among the candidates.
- For "top_matches", the `id` field MUST match the `id` provided in the candidate text block.
- The `why_match` explanation will be displayed directly on the candidate card. make it count!

Be practical and focus on actionable insights for recruiters. Your goal is to help them understand the search results at a glance.
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
