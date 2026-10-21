# Document Summarizer — Hierarchical Map-Reduce for Local LLMs

[![tests](https://github.com/silaylc/hierarchical-map-reduce-for-local-LLMs/actions/workflows/tests.yml/badge.svg)](https://github.com/silaylc/hierarchical-map-reduce-for-local-LLMs/actions/workflows/tests.yml)

A self-hosted document summarization service built for **local and on-prem LLM
deployments**. It reads every supported document in a folder, summarizes each
one independently using a hierarchical Map-Reduce pipeline, and exposes the
whole thing as an OpenAPI-compliant HTTP service that plugs into
[OpenWebUI](https://github.com/open-webui/open-webui) as a Tool Server with
**zero custom integration code**.

Developed and tested end-to-end on a 6 GB VRAM laptop GPU. The summarization
engine is decoupled from the deployment target, so the same code runs against a
small quantized model on a laptop GPU or a full-precision model on a multi-GPU
server without modification.

## Why this exists

Summarizing long documents with an LLM runs into two structural problems the
moment documents get longer than the model's context window:

1. **Content overflow** — the document itself doesn't fit in one prompt.
2. **Reduce overflow** — even after summarizing each chunk, combining *all*
   the chunk summaries into a final summary can itself exceed the context
   window if the document produced many chunks.

Most simple "summarize a folder" scripts only solve (1). This project solves
both — the reduce step estimates the token budget available on the configured
backend (a conservative character-based estimate, no tokenizer dependency)
and re-batches intermediate summaries accordingly, instead of using a fixed
group size. See [Architecture](#architecture) below.

## Key Features

- **Multi-format ingestion** — PDF, DOCX, TXT, and Markdown out of the box.
- **Hierarchical Map-Reduce summarization** — chunk → structured extraction →
  recursive reduction into a single coherent summary.
- **Structured extraction prompts** — each chunk is summarized against an
  explicit schema (main topic, key facts, figures/dates, decisions, risks)
  instead of a generic "summarize this" instruction, so that numbers, dates
  and risks are explicitly asked for at every chunk.
- **Paragraph-aware chunking** — text is split along paragraph boundaries
  first and packed greedily, instead of naive fixed-length character
  slicing that can cut a claim or sentence in half.
- **Context-budget-aware dynamic reduce batching** — the reduce step
  computes how many intermediate summaries actually fit in the backend's
  context window and batches accordingly, so the pipeline completes
  successfully even on small-context deployments instead of failing with a
  "context length exceeded" error.
- **Optional post-reduce normalization pass** — a lightweight pass that
  fixes stylistic/formatting drift introduced across reduce levels, without
  adding or removing any factual content.
- **Automatic retry** on transient backend failures (connection drops,
  temporary overload).
- **Synchronous and asynchronous APIs** — small folders can be summarized in
  a single request; large batches can be submitted as a background job and
  polled for completion, avoiding client-side HTTP timeouts.
- **Per-document fault isolation** — a corrupted or encrypted file never
  aborts the rest of the batch; it's reported inline as a failed item.
- **Flexible output** — summarize an entire folder or a specific subset of
  files, and save results either as one `.docx` per document or a single
  combined `.docx` report.
- **Drop-in OpenWebUI integration** — the service is a plain FastAPI app, so
  OpenWebUI discovers all of the above automatically from its OpenAPI schema.
  No Pipe, Function, or Tool code needs to be written or maintained.

## Architecture

```
                     ┌──────────────┐
                     │  OpenWebUI   │
                     └──────┬───────┘
                            │  OpenAPI Tool Server
                            ▼
                     ┌──────────────┐
                     │   FastAPI    │   <- this repo
                     │   service    │
                     └──────┬───────┘
                            │  OpenAI-compatible API
                            ▼
                     ┌──────────────┐
                     │  LLM backend │   vLLM / Ollama / any
                     │              │   OpenAI-compatible server
                     └──────────────┘

Per document:
  load → split (paragraph-aware) → map (structured extraction, per chunk)
       → hierarchical reduce (context-budget-aware batching)
       → optional normalize → save (.docx) / return
```

The summarization engine (`core/`) has no knowledge of FastAPI, Docker, or
OpenWebUI — it only knows how to talk to an OpenAI-compatible chat completion
endpoint. This means the API layer (`api/`) can be swapped, extended, or
wrapped in a queue system without touching the summarization logic itself.

## Tech Stack

- **Python 3.11**, **FastAPI** — HTTP service and OpenAPI schema generation
- **LangChain** (`langchain-openai`, `langchain-text-splitters`) — LLM client
  and text splitting primitives
- **pypdf**, **python-docx** — document ingestion and `.docx` report generation
- **Docker Compose** — orchestrates the LLM backend, this service, and
  OpenWebUI together
- Any **OpenAI-compatible inference server** as the LLM backend (developed
  and tested against [vLLM](https://github.com/vllm-project/vllm); Ollama's
  OpenAI-compatible endpoint also works with no code changes)

## Project Structure

```
.
├── core/
│   ├── loaders.py       # folder scanning + PDF/DOCX/TXT/MD reading
│   ├── summarizer.py    # async hierarchical Map-Reduce engine
│   └── doc_writer.py    # .docx report generation
├── api/
│   └── main.py          # FastAPI service (sync + async endpoints)
├── tests/               # pytest suite, runs without an LLM backend
├── documents/           # fictional sample documents for trying the pipeline
├── standalone_cli.py    # CLI entry point for local testing without an API server
├── Dockerfile
├── docker-compose.yml   # example: LLM backend + this service + OpenWebUI
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for the full stack) or an already-running
  OpenAI-compatible LLM endpoint (for local testing without Docker)

### Local setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Point the service at any OpenAI-compatible backend via environment variables:

```bash
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_MODEL=your-model-name
export LLM_MAX_MODEL_LEN=4096   # should match your backend's context window

uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
```

Or run the summarization engine directly, without an API server:

```bash
python standalone_cli.py --folder ./documents --out ./outputs \
    --base-url http://localhost:8000/v1 --model your-model-name
```

### Running the tests

The test suite replaces the LLM with a fake, so it needs no GPU or backend. It runs
automatically on every push via GitHub Actions (Python 3.11 and 3.12); the badge at
the top shows the latest result.

```bash
pip install -r requirements.txt pytest
pytest
```

It covers paragraph-aware chunking (no chunk exceeds `chunk_size`, paragraphs
are never cut in half), context-budget-aware reduce batching, termination of the
hierarchical reduce even with a tiny context window, retry behaviour, document
loading and `.docx` output, host/container path translation, and per-document
fault isolation.

### Full stack with Docker Compose

`docker-compose.yml` wires together an LLM backend, this API service, and
OpenWebUI. Adjust the model, volume mounts, and resource reservations to your
own setup, then:

```bash
docker compose up -d
```

## Configuration

All backend settings are read from environment variables so the same image
can point at different models/backends without rebuilding:

| Variable                | Description                                              | Default                        |
|--------------------------|-----------------------------------------------------------|---------------------------------|
| `LLM_BASE_URL`           | OpenAI-compatible endpoint of the LLM backend              | `http://localhost:8000/v1`     |
| `LLM_MODEL`              | Model name/identifier as expected by the backend           | *(backend-specific)*           |
| `LLM_MAX_MODEL_LEN`      | Context window size — must match the backend's actual limit for the dynamic reduce batching to size groups correctly | `2048` |
| `HOST_PATH_PREFIX` / `CONTAINER_PATH_PREFIX` | Optional path-translation pair for Docker deployments, so a natural host filesystem path in a prompt can be transparently mapped to the container's mount point | *(unset — no translation)* |

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/summarize-folder` | POST | Summarizes a folder synchronously; best for small batches |
| `/summarize-folder-async` | POST | Starts summarization in the background, returns a `job_id` immediately |
| `/summarize-folder-status/{job_id}` | GET | Polls the status/result of a background job |

Example request:

```bash
curl -X POST http://localhost:8001/summarize-folder \
  -H "Content-Type: application/json" \
  -d '{
        "folder": "/data/documents",
        "files": ["report.pdf"],
        "save_mode": "combined",
        "save_to_folder": true
      }'
```

Full request/response schemas are available at `/docs` (Swagger UI) once the
service is running.

## Connecting to OpenWebUI

1. Start the service (via Docker Compose or standalone).
2. In OpenWebUI: **Settings → Tools → Add** and enter the service URL
   (e.g. `http://host.docker.internal:8001` if OpenWebUI runs in Docker and
   the service runs on the host).
3. OpenWebUI reads the OpenAPI schema automatically — no further
   configuration is needed. Any model with tool-calling support can now
   invoke `summarize_folder` directly from a chat.

## Known Limitations

This is an actively evolving proof-of-concept, not a hardened production
service. Known gaps, in rough order of priority:

- **Job store is in-memory** — background job state (`/summarize-folder-async`)
  is lost on service restart; a persistent store (Redis, a database) would be
  needed for production use.
- **No authentication** — anything with network access to the service can
  read/write to any mounted path.
- **No OCR** — scanned/image-only PDFs are not supported; only PDFs with an
  extractable text layer.
- **Character/paragraph-based chunking only** — no semantic (embedding-based)
  chunking yet.
- **No end-to-end tests against a real model** — the test suite covers the
  pipeline logic with a fake LLM; summary quality is checked manually.
- **No persistent metrics/logging** — per-run metrics (chunk counts, timing)
  are returned in the API response but not stored anywhere for trend analysis.

## License

[MIT](LICENSE)
