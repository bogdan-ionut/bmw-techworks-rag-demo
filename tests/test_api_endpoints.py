
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from app.api import routes
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Create a test app
app = FastAPI()
app.include_router(routes.router)

client = TestClient(app)

@pytest.fixture
def mock_rag_search():
    with patch("app.api.routes.rag_search_async", new_callable=AsyncMock) as mock:
        yield mock

def test_search_endpoint_calls_service(mock_rag_search):
    # Setup mock return
    mock_rag_search.return_value = {
        "results": [
            {
                "id": "1",
                "score": 0.9,
                "metadata": {"id": "1", "full_name": "Test User", "profile_image_s3_url": "http://img.com"}
            }
        ],
        "latency_sec": 0.1,
        "retrieval_sec": 0.05,
        "rerank_sec": 0.01,
        "filters_applied": None
    }

    response = client.get("/search?q=test&k=24")

    assert response.status_code == 200
    data = response.json()
    assert "sources" in data
    assert len(data["sources"]) == 1
    assert data["sources"][0]["full_name"] == "Test User"

    # Verify service call parameters
    mock_rag_search.assert_called_once()
    call_kwargs = mock_rag_search.call_args[1]
    assert call_kwargs["user_query"] == "test"
    assert call_kwargs["top_k"] == 60 # Default min for Stage 1
    assert call_kwargs["rerank_top_n"] == 15
    assert call_kwargs["with_llm"] is False

def test_search_endpoint_respects_high_k(mock_rag_search):
    mock_rag_search.return_value = {"results": []}

    # If k > 60, it should be passed through
    response = client.get("/search?q=test&k=100")

    call_kwargs = mock_rag_search.call_args[1]
    assert call_kwargs["top_k"] == 100

def test_rag_query_endpoint(mock_rag_search):
    mock_rag_search.return_value = {
        "results": [
            {
                "id": "1",
                "metadata": {"id": "1", "full_name": "Test User"}
            }
        ],
        "answer": {
            "answer": "This is an answer",
            "top_matches": [
                {"id": "1", "why_match": "Good match"}
            ]
        },
        "filters_applied": {"some": "filter"}
    }

    payload = {
        "query": "find developers",
        "with_llm": True
    }

    response = client.post("/query", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["answer"]["answer"] == "This is an answer"
    assert len(data["sources"]) == 1

    # Verify hydration of top_matches
    match = data["answer"]["top_matches"][0]
    assert match["full_name"] == "Test User" # Hydrated from source

    # Verify service call
    call_kwargs = mock_rag_search.call_args[1]
    assert call_kwargs["user_query"] == "find developers"
    assert call_kwargs["llm_top_n"] == 8
    assert call_kwargs["with_llm"] is True
