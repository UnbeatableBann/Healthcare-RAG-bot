# Healthcare AI Assistant

Production-Grade Agentic Hybrid RAG System for Healthcare Knowledge Management

---

# Overview

Healthcare AI Assistant is a healthcare-focused Retrieval-Augmented Generation (RAG) platform designed to answer user questions using only information available in a healthcare knowledge base.

The system combines:

* Agentic Routing
* Hybrid Retrieval (Dense + Sparse)
* Query Rewriting
* Multi Query Retrieval
* Advanced Chunking Strategies
* Reranking
* CRAG-Inspired Retrieval Validation
* Local LLM Support
* RAGAS Evaluation
* Dockerized Deployment

The primary objective is to generate grounded answers while minimizing hallucinations and providing source citations.

---

# Features

## Knowledge Base Ingestion

* Markdown document ingestion
* Metadata extraction
* Multiple chunking strategies
* Vector embedding generation
* Qdrant vector storage

---

## Advanced Retrieval

* Dense Retrieval
* Sparse Retrieval (BM25)
* Reciprocal Rank Fusion (RRF)
* Query Rewriting
* Multi Query Retrieval
* Deduplication

---

## Retrieval Validation

CRAG-inspired validation:

* Heuristic confidence scoring
* LLM-as-a-Judge validation
* Hallucination prevention

---

## Agentic Workflow

Single-agent architecture supporting:

* Knowledge-base questions
* Appointment-related tool requests
* Healthcare policy queries

---

## Evaluation

RAGAS-based evaluation:

* Context Precision
* Context Recall
* Faithfulness
* Answer Relevancy

---

## Observability

* Loguru Logging
* Prometheus Metrics
* Grafana Dashboards

---

# Architecture Overview

The system follows an Agentic Hybrid RAG architecture.

```text
User
 │
 ▼
FastAPI
 │
 ▼
Healthcare Agent
 │
 ├── Tool Route
 │
 └── RAG Route
        │
        ▼
 Query Processing
        │
        ├── Query Rewriting
        ├── Multi Query Generation
        └── Deduplication
                │
                ▼
         Hybrid Retrieval
                │
      ┌─────────┴─────────┐
      ▼                   ▼
 Dense Search       Sparse Search
 (Qdrant)              (BM25)
      │                   │
      └─────────┬─────────┘
                ▼
             RRF Fusion
                ▼
             Reranker
                ▼
               CRAG
                ▼
         LLM Generation
                ▼
            Response
```

A detailed architecture diagram is provided separately.

---

# Technology Stack

| Layer            | Technology   |
| ---------------- | ------------ |
| Backend          | FastAPI      |
| Language         | Python 3.12  |
| Validation       | Pydantic v2  |
| Agent            | LangGraph    |
| Vector Database  | Qdrant       |
| LLM              | Ollama       |
| Embeddings       | BGE          |
| Reranking        | BGE Reranker |
| Metrics          | Prometheus   |
| Monitoring       | Grafana      |
| Testing          | Pytest       |
| Evaluation       | RAGAS        |
| Packaging        | uv           |
| Containerization | Docker       |

---

# Project Structure

```text
healthcare-ai-assistant/

├── apps/
├── core/
├── agent/
├── rag/
├── llm/
├── embeddings/
├── rerankers/
├── vectorstores/
├── evaluation/
├── experiments/
├── data/
├── tests/
├── scripts/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

# Setup Instructions

## Prerequisites

Install:

* Python 3.12+
* Docker
* Docker Compose
* Ollama

---

## Clone Repository

```bash
git clone <repository-url>

cd healthcare-ai-assistant
```

---

## Create Environment

Using uv:

```bash
uv sync
```

Activate environment:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

---

## Configure Environment

Create:

```bash
.env
```

Example:

```env
APP_NAME=Healthcare AI Assistant

API_HOST=0.0.0.0
API_PORT=8000

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=healthcare_documents

LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b

EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

RERANKER_MODEL=BAAI/bge-reranker-base
```

---

## Pull LLM

```bash
ollama pull llama3.1:8b
```

---

# Running Locally

Start Qdrant:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Run API:

```bash
uv run uvicorn apps.api.main:app --reload
```

Application:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

---

# Docker Deployment

Build:

```bash
docker compose build
```

Start:

```bash
docker compose up -d
```

Services:

| Service    | Port |
| ---------- | ---- |
| FastAPI    | 8000 |
| Qdrant     | 6333 |
| Prometheus | 9090 |
| Grafana    | 3000 |

---

# API Endpoints

## Health Check

### Request

```bash
curl -X GET \
http://localhost:8000/api/v1/health
```

### Response

```json
{
  "status": "healthy"
}
```

---

## Ingest Documents

### Request

```bash
curl -X POST \
http://localhost:8000/api/v1/ingest
```

### Response

```json
{
  "status": "success",
  "documents": 10,
  "chunks": 845
}
```

---

## Ask Question

### Request

```bash
curl -X POST \
http://localhost:8000/api/v1/ask \
-H "Content-Type: application/json" \
-d '{
  "question":"Can patients request medication refills through telehealth?"
}'
```

### Response

```json
{
  "answer":"Patients may request medication refills through telehealth consultations if the medication qualifies for refill review and does not require an in-person assessment.",
  "sources":[
    {
      "document":"medication_refill_policy.md"
    },
    {
      "document":"telehealth_consultation_guidelines.md"
    }
  ],
  "confidence":0.91
}
```

---

# Sample Questions and Responses

## Example 1

Question:

```text
Can patients request medication refills through telehealth?
```

Expected:

```text
Yes. Eligible refill requests may be reviewed during telehealth consultations according to the Medication Refill Policy and Telehealth Consultation Guidelines.
```

---

## Example 2

Question:

```text
How can a patient reschedule an appointment?
```

Expected:

```text
Appointments may be rescheduled through the patient portal, by phone, or through approved scheduling channels.
```

---

## Example 3

Question:

```text
Are telehealth visits covered by insurance?
```

Expected:

```text
Coverage depends on the patient's insurance plan. Eligibility and coverage verification should be confirmed prior to the appointment.
```

---

## Example 4

Question:

```text
Can I take ibuprofen with warfarin?
```

Expected:

```text
I could not find this information in the provided documents.
```

---

# Dataset Details

The dataset is entirely synthetic.

No real patient information, PHI, or healthcare records are used.

Documents include:

```text
patient_discharge_instructions.md

appointment_scheduling_policy.md

insurance_eligibility_faq.md

hipaa_privacy_guidelines.md

medication_refill_policy.md

telehealth_consultation_guidelines.md

urgent_care_services.md

prescription_management_policy.md

patient_portal_usage_guide.md

billing_and_payments_policy.md
```

Total corpus size:

* ~10 documents
* ~10,000–15,000 words

The dataset intentionally contains overlapping concepts to improve retrieval evaluation.

---

# LLM Used

## Llama 3.1 8B

Provider:

```text
Ollama
```

Reason:

* Strong instruction following
* Good local inference quality
* Open-source
* Easy deployment

---

# Embedding Model Used

## BAAI/bge-small-en-v1.5

Reason:

* Strong retrieval performance
* Fast inference
* Low resource requirements
* Well-suited for semantic search

---

# Vector Database Used

## Qdrant

Reason:

* Production-ready
* Fast vector search
* Hybrid retrieval support
* Metadata filtering support
* Docker-friendly

---

# Prompting Strategy

The generation prompt enforces:

* Answer only from retrieved context
* No unsupported claims
* No guessing
* Refuse missing information
* Professional healthcare language

If sufficient evidence is unavailable:

```text
I could not find this information in the provided documents.
```

---

# Agent Workflow

Single-agent architecture.

```text
User Question
      ↓
Intent Detection
      ↓
Tool or RAG
```

Examples:

| Query                    | Route |
| ------------------------ | ----- |
| Book appointment         | Tool  |
| Appointment availability | Tool  |
| Telehealth policy        | RAG   |
| HIPAA question           | RAG   |

---

# Unit Tests

Coverage includes:

## Chunking

* Recursive chunking
* Semantic chunking
* Contextual chunking
* Hybrid chunking

## Query Processing

* Query rewriting
* Multi query generation
* Deduplication

## Retrieval

* Dense retrieval
* Sparse retrieval
* Hybrid retrieval

## Reranking

* Reranker scoring

## CRAG

* Heuristic evaluation
* LLM Judge

## Agent

* Routing behavior

---

# Evaluation Script

Run:

```bash
python scripts/evaluate.py
```

Metrics:

* Context Precision
* Context Recall
* Faithfulness
* Answer Relevancy

Results are stored in:

```text
evaluation/reports/
```

and

```text
experiments/results.json
```

---

# Current Limitations

* Synthetic dataset only
* English-only documents
* Single-agent architecture
* No authentication
* No user management
* No PHI detection
* No access control

---

# Future Improvements

* Authentication and RBAC
* PHI detection
* Advanced guardrails
* Multi-tenant support
* Document versioning
* Knowledge graph integration
* GraphRAG
* Human feedback workflows
* Kubernetes deployment
* CI/CD automation

---
