# app/rag/prompt.py

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

ALLOWED_FILTER_FIELDS = {
    "eyewear_present",
    "beard_present",
    "minimum_estimated_years_of_exp",
    "tech_tokens",
    "image_search_tags",
    "full_name_normalized",
    "name_tokens",
    "location_normalized",
    "school",
    "school_degree",
    "job_title",
}

CITY_NORMALIZATION_MAP = {
    "cluj": "cluj-napoca",
    "cluj napoca": "cluj-napoca",
    "bucuresti": "bucharest",
    "bucharest": "bucharest",
    "iasi": "iasi",
    "timisoara": "timisoara",
    "brasov": "brasov",
    "munich": "munich",
    "munchen": "munich",
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
  location_normalized (string, lowercase)
  school (string)
  school_degree (string)
  job_title (string)
  full_name_normalized (string, lowercase) - USE ONLY FOR EXACT FULL NAME MATCHES
  name_tokens (list of strings) - USE FOR PARTIAL NAMES / FIRST NAMES

- For visual attributes (e.g., "eyeglasses", "smiling", "beard"), use the boolean `eyewear_present`, `beard_present`, or `image_search_tags` filters.
- **IMPORTANT**: Remove visual descriptors from the `semantic_query` to avoid biasing the vector search. The `semantic_query` should only contain professional criteria (skills, roles, experience).
- **Location**: If the user explicitly asks for a city, use `location_normalized` (e.g. "Cluj", "Munich" -> "cluj-napoca", "munich").
- **Education**: If the user asks for "Bachelor" or "Master" or specific university, use `school` or `school_degree`.
- **Name**: If the user asks for a person by name:
  - If it looks like a full name (e.g. "Ionut Buraga"), use `full_name_normalized` (e.g. "ionut buraga").
  - If it is a partial name (e.g. "Ionut"), use `name_tokens` with the `$in` operator (e.g. `{"name_tokens": {"$in": ["ionut"]}}`).
- If a user asks for "python engineers with eyeglasses in Cluj", the `semantic_query` should be "python engineers" and the filter should be `{"$and": [{"eyewear_present": {"$eq": true}}, {"location_normalized": {"$eq": "cluj-napoca"}}]}`.
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
  "answer": "Generate a concise summary (max 2 sentences). Focus ONLY on hard skills and fit. No fluff. No intro.",
  "top_matches": [
    {
      "id": "...",
      "why_match": "A single line, data-driven justification (max 10-15 words). Mention key matching skills/companies."
    }
  ]
}

- Output MUST be valid JSON only (no markdown). Do NOT include conversational text, "thinking", or "explanation" outside the JSON object.
- The "answer" field is your "elevator pitch" to the recruiter. Make it insightful but concise (max 300 chars).
- IMPORTANT: If the number of `top_matches` you return is less than the total number of candidate profiles reviewed, you MUST state this clearly in your "answer".
- For "top_matches", the `id` field MUST match the `id` provided in the candidate text block.
- Do NOT return `full_name` or `profile_url` in `top_matches`; we will map them back by `id`.
"""

ANSWER_STREAM_SYSTEM_PROMPT = """
You are a recruiting intelligence assistant for BMW TechWorks.

Part 1: The Answer
Generate a concise summary (max 2 sentences). Focus ONLY on hard skills and fit. No fluff. No intro.
Directly output the text of the answer. Do not use JSON for this part.

Part 2: The Matches
Output the separator string: "###METADATA###"
Then output a JSON object with the following schema:
{
  "top_matches": [
    {
      "id": "...",
      "why_match": "A single line, data-driven justification (max 10-15 words). Mention key matching skills/companies."
    }
  ]
}

- For "top_matches", the `id` field MUST match the `id` provided in the candidate text block.
- Do NOT return `full_name` or `profile_url` in `top_matches`; we will map them back by `id`.
"""


def sanitize_filter(flt: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
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


def format_candidates_for_prompt(candidates: List[Dict[str, Any]], max_chars: int = 1200) -> str:
    """
    Optimized formatter that constructs a LEAN payload for the LLM.
    Strips raw embedding text (visual descriptions) and irrelevant URLs.
    """
    blocks: List[str] = []
    for i, c in enumerate(candidates, start=1):
        md = c.get("metadata", {}) or {}
        score = c.get("score")

        # Build structured lean profile from metadata
        lines = []
        if md.get("headline"):
            lines.append(f"Headline: {md['headline']}")
        if md.get("job_title"):
            lines.append(f"Role: {md['job_title']}")
        if md.get("company"):
            lines.append(f"Company: {md['company']}")

        # Tech tokens
        tech = md.get("tech_tokens", [])
        if tech:
            # limit tech tokens to avoid spam
            lines.append(f"Skills: {', '.join(tech[:25])}")

        if md.get("location"):
            lines.append(f"Location: {md['location']}")
        if md.get("minimum_estimated_years_of_exp"):
            lines.append(f"Exp: {md['minimum_estimated_years_of_exp']} yrs")

        # Education
        if md.get("school"):
            lines.append(f"School: {md['school']}")

        lean_text = "\n".join(lines)

        # Fallback if metadata is somehow empty, use truncated text but keep it short
        if not lean_text.strip():
             text = (c.get("text") or "").strip()
             lean_text = text[:600] + "..." if len(text) > 600 else text

        blocks.append(
            "\n".join(
                [
                    f"Candidate #{i}",
                    f"id: {c.get('id')}",
                    f"score: {score}",
                    f"full_name: {md.get('full_name', 'Unknown')}",
                    "Summary:",
                    lean_text,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
