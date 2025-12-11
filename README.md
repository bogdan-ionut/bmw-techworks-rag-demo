# BMW TechWorks Romania RAG Demo

Retrieval-Augmented Generation (RAG) app using LangChain, FastAPI, and AWS. Builds a knowledge base from BMW TechWorks employee profiles (public data) for querying roles, education, etc.

## Features
- Data ingestion from JSONL (S3 support).
- Embeddings: OpenAI text-embedding-3-large.
- Vector DB: Chroma.
- LLM: Switch between OpenAI, Anthropic, Gemini (secrets from AWS).
- FastAPI endpoint for queries.
- AWS: S3 for data, Secrets Manager for keys.

## Setup Local
1. Clone: `git clone https://github.com/YourUsername/bmw-techworks-rag-demo`
2. Venv: `python -m venv .venv` & activate.
3. Install: `pip install -r requirements.txt`
4. .env: Set keys.
5. Build DB: `python app/main.py`
6. Run API: `uvicorn app.api:app --reload`
7. Test: POST to /query {"query": "Java developers from UTCN"}

## AWS Deploy
- Upload JSONL to S3.
- Use Secrets Manager for keys.
- Deploy on EC2: See deploy/ec2_setup.sh

Built for GenAI Engineer interview at BMW TechWorks Romania. Demonstrates RAG, prompt eng, data pipelines, MLOps.