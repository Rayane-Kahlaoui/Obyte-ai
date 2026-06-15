# Orbyte — Sovereign Multi-Agent Legal RAG System

A clearance-aware Retrieval-Augmented Generation API for legal documents. Queries are filtered by user permission level before an LLM generates a grounded response. Every response is audited by a separate judge agent.

---

## What it does

1. **Identity Agent** — maps a username to a clearance level (Public / Internal / Confidential).
2. **Retrieval Agent** — queries a local ChromaDB vector store; filters out chunks the user is not cleared to see.
3. **Responder** — calls an LLM (OpenRouter by default) with only the authorized context. Temperature is 0.
4. **Judge Agent** — scores faithfulness and completeness of the response (1–5) and marks it approved or not.

---

## Requirements

| Requirement | Version |
|---|---|
| Docker | 24+ |
| Docker Compose | v2+ |
| OpenRouter API key | (or Ollama running locally, or HF token) |

The database (`database/`) and data (`data/`) directories are already included in the repo. If they are present, the container starts and is ready to serve queries immediately — no re-embedding required.

---

## Environment variables

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `LLM_API_TYPE` | `openrouter` | `openrouter` · `local_api` · `hf_inference` · `mock` |
| `OPENROUTER_API_KEY` | — | Required when `LLM_API_TYPE=openrouter` |
| `OPENROUTER_MODEL` | `openai/gpt-oss-20b:free` | Model slug used for the OpenRouter backend |
| `HF_TOKEN` | — | Required when `LLM_API_TYPE=hf_inference` |
| `LOCAL_LLM_API_URL` | `http://host.docker.internal:11434/v1` | Base URL of Ollama or llama.cpp server |
| `LOCAL_LLM_MODEL` | `llama3.2:1b` | Model name when using `local_api` |

> **Using Ollama locally?** Set `LLM_API_TYPE=local_api`. The compose file routes `host.docker.internal` to your machine automatically on Docker Desktop.

---

## Run with Docker

```bash
# 1. Build and start
docker compose up --build -d

# 2. Check logs
docker compose logs -f

# 3. Stop
docker compose down
```

The API is available at `http://localhost:8000`.

---

## Run without Docker (bare Python)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Interactive CLI (Testing)

An interactive chat session is included for hands-on testing. It lets you pick an LLM backend and a mock user before entering a prompt loop.

**Windows — double-click or run from a terminal:**

```bash
run_chat.bat
```

On startup you will be asked to:
1. **Select an LLM backend** — OpenRouter, local Ollama, Hugging Face, or mock.
2. **Select a user** — alice (Confidential), bob (Internal), or charlie (Public).

Inside the session you can switch users, backend, or retrieval depth on the fly:

| Command | Description |
|---|---|
| `/user <name>` | Switch active user (alice / bob / charlie) |
| `/mode <mode>` | Switch LLM backend (openrouter / local / hf / mock) |
| `/top-k <n>` | Change the number of retrieved chunks |
| `/exit` | End the session |

See [`test_scenarios.md`](test_scenarios.md) for a set of ready-made questions and expected results covering every clearance level.

---

## Re-embed the database (optional)

Only needed if you want to re-ingest from scratch or add documents.

```bash
# Inside the container
docker compose exec orbyte-api python rag/embed_documents.py

# Or locally
python rag/embed_documents.py --limit 100   # embed first 100 docs only
python rag/embed_documents.py               # embed all
```

---

## Endpoints

### `GET /api/v1/health`
Returns system status, ChromaDB collection info, and LLM configuration.

**Response**
```json
{
  "status": "healthy",
  "chroma_collection": "legal_documents",
  "chunk_count": 4200,
  "embedding_model": "all-MiniLM-L6-v2",
  "llm_mode": "openrouter",
  "llm_model": "openrouter/free",
  "version": "1.0.0"
}
```

---

### `POST /api/v1/query`
Main pipeline. Authenticates the user, retrieves authorized document chunks, generates a response, and audits it.

**Request body**
```json
{
  "username": "alice",
  "query": "What are the compliance training requirements?",
  "top_k": 3
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `username` | string | ✓ | Must be one of `alice`, `bob`, `charlie` (see Users section) |
| `query` | string | ✓ | Natural language question |
| `top_k` | int | — | Number of chunks to retrieve (1–10, default 3) |

**Response**
```json
{
  "query": "What are the compliance training requirements?",
  "username": "alice",
  "clearance": "Confidential",
  "response": "Annual compliance training is required for all employees...",
  "approved": true,
  "chunks_used": [
    {
      "chunk_id": "doc_0_chunk_2",
      "similarity": 0.87,
      "clearance": "Internal",
      "snippet": "All employees must complete..."
    }
  ],
  "chunks_filtered": 1,
  "evaluation": {
    "faithfulness_score": 5,
    "completeness_score": 4,
    "approved": true,
    "feedback": "Response approved for display."
  },
  "processing_time": 1.234
}
```

**Error codes**
| Code | Reason |
|---|---|
| `403` | User clearance too low for all matching documents |
| `500` | Internal server error |

---

### `GET /api/v1/auth/verify`
Checks whether a user is authorized to access a specific clearance level.

**Query parameters**

| Param | Type | Required | Description |
|---|---|---|---|
| `user` | string | ✓ | Username |
| `document_clearance` | string | ✓ | `Public` · `Internal` · `Confidential` |

**Example**
```
GET /api/v1/auth/verify?user=bob&document_clearance=Confidential
```

**Response**
```json
{
  "username": "bob",
  "user_clearance": "Internal",
  "resource_clearance": "Confidential",
  "authorized": false,
  "explanation": "User 'bob' has clearance 'Internal' which is below the required 'Confidential'."
}
```

---

### `POST /api/v1/audit`
Audits any query/context/response triple independently, without going through the full pipeline.

**Request body**
```json
{
  "query": "What training is required?",
  "retrieved_contexts": [
    "The compliance policy requires annual training for all staff."
  ],
  "response": "Annual training is required."
}
```

**Response**
```json
{
  "faithfulness_score": 5,
  "completeness_score": 5,
  "approved": true,
  "feedback": "Response approved for display."
}
```

---

### Interactive docs

| URL | Description |
|---|---|
| `http://localhost:8000/docs` | Swagger UI — try every endpoint in the browser |
| `http://localhost:8000/redoc` | ReDoc — clean reference documentation |

---

## Users (built-in)

| Username | Clearance | Can access |
|---|---|---|
| `alice` | Confidential | Public · Internal · Confidential |
| `bob` | Internal | Public · Internal |
| `charlie` | Public | Public only |

---

## Project structure

```
Orbyte/
├── api/
│   └── main.py              # FastAPI app and route definitions
├── rag/
│   ├── config.py            # All configuration and env var resolution
│   ├── orchestrator.py      # Pipeline coordinator
│   ├── llm_client.py        # LLM backends (OpenRouter, Ollama, HF, mock)
│   ├── embed_documents.py   # One-shot ingestion script
│   ├── semantic_chunker.py  # Semantic chunking via sentence-transformers
│   ├── query_db.py          # CLI entry-point and interactive chat loop
│   └── agents/
│       ├── identity_agent.py   # Clearance lookup and access control
│       ├── retrieval_agent.py  # Vector search + filtering
│       └── judge_agent.py      # Response quality auditor
├── data/                    # Parquet dataset files
├── database/                # Persisted ChromaDB (sqlite3 + vectors)
├── run_chat.bat             # One-click interactive CLI launcher (Windows)
├── test_scenarios.md        # Ready-made test questions and expected results
├── verify_system.py         # Unit-test suite for all agents and MCP tools
├── verify_api.py            # FastAPI endpoint integration tests
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```
