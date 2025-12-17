# BMW TechWorks Romania RAG Demo

Retrieval-Augmented Generation (RAG) app using LangChain, FastAPI, and AWS. Builds a knowledge base from BMW TechWorks employee
profiles (public data) for querying roles, education, etc.

## Features
- Data ingestion from JSONL (S3 support).
- Embeddings: OpenAI text-embedding-3-large.
- Vector DB: Pinecone.
- LLM: Switch between OpenAI, Gemini (secrets from AWS).
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

## Docker
- Build: `docker build -t bmw-techworks-rag-demo:local .`
- Run locally: `docker run --env-file .env -p 8000:8000 bmw-techworks-rag-demo:local`
- The image exposes FastAPI on port 8000.

## CI/CD with GitHub Actions
- `.github/workflows/ci.yml` installs dependencies on Python 3.11 and ensures the codebase compiles (no runtime secrets required).
- `.github/workflows/docker-publish.yml` builds the Docker image with Buildx and, on pushes to `main` or tags, publishes to GitHub Container Registry (GHCR) as `ghcr.io/<org>/bmw-techworks-rag-demo` with `latest` + SHA/tag labels.
- Both workflows run on GitHub-hosted Ubuntu runners and cache dependencies/build layers for speed.
- GHCR pushes authenticate using the built-in `GITHUB_TOKEN`; no extra secrets are required for publishing.

## AWS Deploy
- Upload JSONL to S3.
- Use Secrets Manager for keys.
- Deploy on EC2: See deploy/ec2_setup.sh
- For ECS, point your service/task definition to the GHCR image built by `docker-publish.yml` (e.g., `ghcr.io/<org>/bmw-techworks-rag-demo:latest`) and supply environment variables/secrets via task definitions or AWS Parameter Store.

Built for GenAI Engineer interview at BMW TechWorks Romania. Demonstrates RAG, prompt eng, data pipelines, MLOps.
