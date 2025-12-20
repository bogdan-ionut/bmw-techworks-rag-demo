
import pytest
from unittest.mock import AsyncMock, patch
from app.api import routes
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Create a test app to bypass global dependencies in app.main
app = FastAPI()
app.include_router(routes.router)

client = TestClient(app)

@pytest.fixture
def mock_rag_search():
    with patch("app.api.routes.rag_search_async", new_callable=AsyncMock) as mock:
        yield mock

def test_smoke_search_endpoint(mock_rag_search):
    """
    Simple smoke test to ensure the search endpoint is reachable and returns 200.
    """
    # Setup minimal mock return
    mock_rag_search.return_value = {
        "results": [],
        "latency_sec": 0.1,
        "retrieval_sec": 0.05,
        "rerank_sec": 0.01,
        "filters_applied": None
    }

    response = client.get("/search?q=smoke_test")
    assert response.status_code == 200

def test_smoke_query_endpoint(mock_rag_search):
    """
    Simple smoke test to ensure the query endpoint is reachable and returns 200.
    """
    mock_rag_search.return_value = {
        "results": [],
        "answer": {"answer": "Smoke test", "top_matches": []},
        "filters_applied": None
    }

    payload = {
        "query": "smoke test",
        "with_llm": True
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
