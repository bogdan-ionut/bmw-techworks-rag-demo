# Talent Intelligence RAG Demo

> **Context:** This application was built as a Proof of Concept (POC) and a hands-on learning project to prepare for my technical interview with Ionut and Alexandru. The goal was to explore the end-to-end lifecycle of a Generative AI product—from data engineering and vector search to cloud deployment.

## 💡 The Concept

Standard recruiting searches rely on rigid keyword matching. I wanted to build a prototype that allows for **semantic understanding** and **multimodal filtering**.

This app allows you to query a dataset of **~480 public profiles** using natural language, filtering candidates not just by their text CVs, but by visual attributes inferred from their profile pictures (e.g., *"Python engineers who wear glasses"*).

## 🎯 Core GenAI Features

*   **Semantic Search:** Uses dense vector embeddings (`text-embedding-3-large`) to find candidates based on meaning (e.g., "Cloud Architect" matches "DevOps Engineer").
*   **Multimodal Data Enrichment:** I used **GPT 5.2 (Vision)** to analyze profile images and generate textual descriptions (`image_captions`). This metadata allows the LLM to reasoning about visual traits alongside technical skills.
*   **Hybrid Retrieval & Reranking:** Combines **Pinecone** vector search with a **Cohere Reranker** (`rerank-english-v3.0`) to improve precision and fix the "lost in the middle" problem.
*   **LLM Reasoning:** Uses **OpenAI** (`gpt-4o-mini`) or **Google** (`gemini-2.5-flash`) to synthesize the retrieved data into a coherent executive summary for the recruiter.

## 🧠 The Data & AI Pipeline

As a Generative AI Engineer, I focused heavily on the quality of the data before it even reaches the vector database.

### 1. Data Engineering & Acquisition
Instead of using a generic dataset, I built a custom one to simulate a real-world scenario:
*   **Source:** Scraped public profiles ("People currently at BMW TechWorks Romania") using PhantomBuster.
*   **Normalization:** Unified 17 disparate JSON sources into a clean `jsonl` structure, standardizing locations and names.

### 2. Semantic Enrichment (The "Secret Sauce")
This is where the RAG pipeline differentiates itself. I didn't just chunk text; I enriched it:
*   **Visual Analysis:** Downloaded profile images and passed them through a Vision Model (GPT 5.2) to tag attributes (glasses, smiling, formal wear).
*   **Metadata Extraction:** Calculated fields like `minimum_estimated_years_of_exp` to enable hard filtering in the vector DB.

### 3. Retrieval Architecture
*   **Embedding Model:** OpenAI `text-embedding-3-large` (chosen for its higher dimensionality and performance on English text).
*   **Vector Database:** **Pinecone (Serverless)**. Used purely for similarity search with metadata filtering (e.g., `$and: [{ "tech": "python" }, { "glasses": true }]`).
*   **Reranking:** Top-k results from Pinecone are passed to a **Cross-Encoder (Cohere)** to score relevance based on the specific query nuance.

## ☁️ Infrastructure & Deployment (Learning Journey)

While my primary focus is GenAI, I treated this project as an opportunity to learn how to deploy AI apps securely in a cloud environment. I moved beyond localhost to build a production-like environment on AWS.

*   **Containerization:** The application (FastAPI + LangChain) is Dockerized (`python:3.11-slim`).
*   **Serverless Compute:** Deployed on **AWS ECS Fargate**. I chose this to avoid managing EC2 instances while ensuring scalability.
*   **Security:**
    *   **Secrets Management:** No keys are stored in the repo. The app fetches API keys (OpenAI, Pinecone, Cohere) at runtime from AWS Secrets Manager via IAM Roles.
    *   **Network:** Traffic is routed through an Application Load Balancer (ALB) with SSL/TLS termination.
*   **CI/CD:** A GitHub Actions pipeline handles testing, building, and deploying the container to AWS ECR/ECS automatically on every push.

## 🛠 Tech Stack

*   **Orchestration:** LangChain
*   **LLMs:** OpenAI (GPT-4o), Google (Gemini 2.5 Flash)
*   **Vector Store:** Pinecone
*   **Reranker:** Cohere
*   **Backend:** Python, FastAPI
*   **Cloud:** AWS (ECS, Fargate, ALB, Secrets Manager, S3)

## 💻 Local Development

To run the application locally:

```bash
# 1. Clone the repo
git clone https://github.com/ionutbogdan/bmw-techworks-rag-demo.git

# 2. Create .env file
# (Note: In prod, these are fetched from AWS Secrets Manager)
cp .env.example .env

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server
uvicorn app.main:app --reload
```

---
**Author:** Ionut Bogdan, Generative AI Engineer Candidate
