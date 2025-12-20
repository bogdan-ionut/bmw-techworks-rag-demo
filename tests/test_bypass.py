import pytest
from unittest.mock import AsyncMock, patch
from app.rag.service import rag_search_async

@pytest.mark.asyncio
async def test_bypass_llm_for_simple_query():
    # Mock retrieval to return some dummy results
    with patch("app.rag.service.retrieve_profiles", return_value=[]) as mock_retrieve, \
         patch("app.rag.service.rerank_candidates", return_value=[]) as mock_rerank, \
         patch("app.rag.service.generate_answer", new_callable=AsyncMock) as mock_generate_answer, \
         patch("app.rag.service.plan_query_with_llm", new_callable=AsyncMock) as mock_plan, \
         patch("app.rag.service.get_settings") as mock_settings:

        # Setup mock settings
        mock_settings.return_value.cohere_api_key = "fake_key"
        mock_settings.return_value.cohere_rerank_model = "rerank-english-v3.0"

        # Simple query should bypass LLM
        query = "Java" # Short, no complex connectors -> SIMPLE
        result = await rag_search_async(user_query=query, with_llm=True)

        assert result["query_type"] == "SIMPLE"
        mock_generate_answer.assert_not_called()
        assert result["answer"]["answer"] == "Here are the top matches for your search. Use the filters to refine the results."
        assert result["llm_sec"] == 0.0

@pytest.mark.asyncio
async def test_full_llm_for_complex_query():
    with patch("app.rag.service.retrieve_profiles", return_value=[]) as mock_retrieve, \
         patch("app.rag.service.rerank_candidates", return_value=[]) as mock_rerank, \
         patch("app.rag.service.generate_answer", new_callable=AsyncMock) as mock_generate_answer, \
         patch("app.rag.service.plan_query_with_llm", new_callable=AsyncMock) as mock_plan, \
         patch("app.rag.service.get_settings") as mock_settings:

        mock_settings.return_value.cohere_api_key = "fake_key"
        mock_settings.return_value.cohere_rerank_model = "rerank-english-v3.0"
        # Mock planner response
        mock_plan.return_value = {"semantic_query": "Java developer", "filter": None}
        mock_generate_answer.return_value = {"answer": "AI generated", "top_matches": []}

        # Complex query should trigger LLM
        query = "Java developer with 5 years experience and aws skills"
        result = await rag_search_async(user_query=query, with_llm=True)

        assert result["query_type"] == "COMPLEX"
        mock_generate_answer.assert_called_once()
        assert result["answer"]["answer"] == "AI generated"
