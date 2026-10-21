"""
api/main.py
===========
Exposes the document summarization engine as an HTTP service.

This service can be added to OpenWebUI as an "OpenAPI Tool Server"
without writing any custom Python (Pipe/Tool) code, because FastAPI
already generates its own OpenAPI schema automatically at /openapi.json.

In OpenWebUI:
  Admin Panel -> Settings -> Tools -> "+" -> enter this service's URL
  (e.g. http://localhost:8000, or http://host.docker.internal:8000 from Docker)
The model can then automatically discover and call the "summarize_folder"
endpoint as a tool.

RUNNING (local PoC):
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.loaders import scan_folder, load_document
from core.summarizer import DocumentSummarizer, SummarizerConfig
from core.doc_writer import write_summary_docx, write_combined_docx

app = FastAPI(
    title="Document Summarization API",
    description="Summarizes every document in a folder individually using hierarchical Map-Reduce.",
    version="1.0.0",
)


def _translate_path(folder: str) -> str:
    """Translates the HOST path given by the user/model in the prompt
    (e.g. C:/Users/... on Windows) into the actual mount path inside
    the container.

    If the HOST_PATH_PREFIX and CONTAINER_PATH_PREFIX env vars are set
    (in docker-compose.yml) and the given folder starts with
    HOST_PATH_PREFIX, the prefix is replaced with CONTAINER_PATH_PREFIX.
    This way the user/model can always write the natural host path and
    never needs to know about the translation happening here.

    If the env vars are not set (e.g. the API runs directly on the host
    and Docker isn't involved at all), the folder is returned unchanged.
    """
    host_prefix = os.environ.get("HOST_PATH_PREFIX")
    container_prefix = os.environ.get("CONTAINER_PATH_PREFIX")
    if not host_prefix or not container_prefix:
        return folder

    normalized = folder.replace("\\", "/")
    host_prefix_norm = host_prefix.replace("\\", "/").rstrip("/")

    if normalized.lower().startswith(host_prefix_norm.lower()):
        rest = normalized[len(host_prefix_norm):]
        return container_prefix.rstrip("/") + rest

    return folder


class FolderRequest(BaseModel):
    folder: str = Field(..., description="Absolute path of the folder to scan on the server")
    recursive: bool = Field(True, description="Whether to also scan subfolders")
    files: Optional[list[str]] = Field(
        None,
        description=(
            "If given, only summarize these specific filenames within the folder "
            "(e.g. ['a.pdf', 'c.docx']) instead of every supported file. "
            "Filenames not found in the folder are reported separately, not errored."
        ),
    )
    # If the fields below are left empty, the service's default SummarizerConfig
    # is used (from the LLM_BASE_URL / LLM_MODEL env vars).
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    concurrency: Optional[int] = None
    base_url: Optional[str] = Field(None, description="Use a different LLM backend for this request")
    model: Optional[str] = Field(None, description="Use a different model for this request")
    structural_chunking: Optional[bool] = Field(
        None, description="Split text along paragraph boundaries instead of raw characters (default: on)"
    )
    enable_normalization: Optional[bool] = Field(
        None, description="Apply a post-reduce style/format normalization pass (default: on)"
    )
    save_to_folder: bool = Field(
        True, description="Also write summaries to disk as files, in addition to the JSON response"
    )
    save_mode: Literal["per_file", "combined"] = Field(
        "per_file",
        description=(
            "per_file: save each document's summary as a separate .docx file next to the original. "
            "combined: save all summaries from this run into a single .docx file "
            "at the folder root (overwritten each run, not appended)."
        ),
    )


class DocumentSummaryResponse(BaseModel):
    file: str
    summary: str
    chunk_count: int
    map_calls: int
    reduce_calls: int
    normalize_calls: int
    total_seconds: float


class FolderSummaryResponse(BaseModel):
    folder: str
    document_count: int
    documents: list[DocumentSummaryResponse]
    not_found: list[str] = []  # filenames requested via req.files but not found in the folder


def _build_config(req: FolderRequest) -> SummarizerConfig:
    config = SummarizerConfig()
    if req.chunk_size is not None:
        config.chunk_size = req.chunk_size
    if req.chunk_overlap is not None:
        config.chunk_overlap = req.chunk_overlap
    if req.concurrency is not None:
        config.concurrency = req.concurrency
    if req.base_url is not None:
        config.base_url = req.base_url
    if req.model is not None:
        config.model = req.model
    if req.structural_chunking is not None:
        config.structural_chunking = req.structural_chunking
    if req.enable_normalization is not None:
        config.enable_normalization = req.enable_normalization
    return config


@app.get("/health")
async def health():
    return {"status": "ok"}


async def _process_folder(req: FolderRequest) -> FolderSummaryResponse:
    """Core processing logic -- used by both the synchronous
    /summarize-folder endpoint and the async job worker (_run_job_in_background)."""
    folder = _translate_path(req.folder)
    paths = scan_folder(folder, recursive=req.recursive)  # FileNotFoundError is caught by the caller

    not_found: list[str] = []
    if req.files:
        # Filter down to only the requested filenames (exact name match, not path).
        wanted = set(req.files)
        found_names = {p.name for p in paths}
        not_found = sorted(wanted - found_names)
        paths = [p for p in paths if p.name in wanted]

    if not paths:
        return FolderSummaryResponse(
            folder=req.folder, document_count=0, documents=[], not_found=not_found
        )

    summarizer = DocumentSummarizer(_build_config(req))
    results: list[DocumentSummaryResponse] = []
    combined_parts: list[tuple[str, str]] = []  # (file_name, summary) -- for save_mode="combined"

    for path in paths:
        try:
            text = load_document(path)
        except Exception as e:
            # Encrypted/corrupt PDF, unsupported internal format, etc. -- mark
            # this document as FAILED and continue with the rest instead of
            # crashing the whole request.
            results.append(DocumentSummaryResponse(
                file=path.name, summary=f"[ERROR] Could not read document: {e}",
                chunk_count=0, map_calls=0, reduce_calls=0, normalize_calls=0, total_seconds=0.0,
            ))
            continue

        if not text.strip():
            results.append(DocumentSummaryResponse(
                file=path.name, summary="(Empty document, skipped.)",
                chunk_count=0, map_calls=0, reduce_calls=0, normalize_calls=0, total_seconds=0.0,
            ))
            continue

        try:
            summary, metrics = await summarizer.summarize_document(text, file_name=path.name)
        except Exception as e:
            # If the LLM still fails after 3 retries (dropped connection,
            # unrecoverable context error, etc.) -- again mark this document
            # as FAILED and continue with the rest.
            results.append(DocumentSummaryResponse(
                file=path.name, summary=f"[ERROR] Could not summarize: {e}",
                chunk_count=0, map_calls=0, reduce_calls=0, normalize_calls=0, total_seconds=0.0,
            ))
            continue

        if req.save_to_folder:
            if req.save_mode == "per_file":
                out_file = path.parent / f"{path.stem}_summary.docx"
                try:
                    write_summary_docx(out_file, f"{path.name} Summary", summary)
                except Exception as e:
                    print(f"[WARNING] Failed to write {out_file}: {e}")
            else:  # "combined"
                combined_parts.append((path.name, summary))

        results.append(DocumentSummaryResponse(
            file=path.name,
            summary=summary,
            chunk_count=metrics.chunk_count,
            map_calls=metrics.map_calls,
            reduce_calls=metrics.reduce_calls,
            normalize_calls=metrics.normalize_calls,
            total_seconds=round(metrics.total_seconds, 2),
        ))

    if req.save_to_folder and req.save_mode == "combined" and combined_parts:
        # NOTE: this file is OVERWRITTEN ON EVERY RUN (not appended) -- it
        # collects the summaries of every document from that run into a
        # single file.
        combined_file = Path(folder) / "ALL_SUMMARIES.docx"
        try:
            write_combined_docx(combined_file, req.folder, combined_parts)
        except Exception as e:
            print(f"[WARNING] Failed to write {combined_file}: {e}")

    return FolderSummaryResponse(
        folder=req.folder, document_count=len(results), documents=results, not_found=not_found
    )


@app.post("/summarize-folder", response_model=FolderSummaryResponse)
async def summarize_folder(req: FolderRequest):
    """Summarizes each document in the folder individually and returns the results
    in a single synchronous response. Best for small folders (a handful of short
    documents) -- for large batches, use /summarize-folder-async instead, since
    this endpoint can trigger client-side (browser/OpenWebUI) HTTP timeouts on
    long-running requests."""
    try:
        return await _process_folder(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# --------------------------------------------------------------------------
# ASYNC JOB ARCHITECTURE
# --------------------------------------------------------------------------
# Problem: /summarize-folder is synchronous -- the client (OpenWebUI/browser)
# waits until the response arrives. A folder with 20 large documents can take
# minutes, and the client side may not be able to wait that long, cutting the
# connection with an HTTP timeout (even though the server keeps working
# in the background).
#
# Solution: return a job_id as soon as the request comes in (processing
# continues in the background), and let the result be polled from a separate
# /summarize-folder-status/{job_id} endpoint. This way the client never waits
# and can check the status whenever it wants.
#
# NOTE (known limitation): the _jobs dict is kept IN MEMORY ONLY -- if the
# service restarts (e.g. docker compose restart), all job history is lost;
# it is not persistent. This is an acceptable simplification for a PoC;
# production would need a persistent job store such as Redis/a database.

_jobs: dict[str, dict] = {}


class JobStartResponse(BaseModel):
    job_id: str
    status: Literal["running"]
    message: str = "Job started. Poll /summarize-folder-status/{job_id} for the result."


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["running", "completed", "failed"]
    created_at: str
    finished_at: Optional[str] = None
    result: Optional[FolderSummaryResponse] = None
    error: Optional[str] = None


async def _run_job_in_background(job_id: str, req: FolderRequest) -> None:
    try:
        result = await _process_folder(req)
        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["result"] = result
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)
    finally:
        _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.post("/summarize-folder-async", response_model=JobStartResponse)
async def summarize_folder_async(req: FolderRequest):
    """Starts summarizing the folder in the background and returns immediately
    with a job_id. Use this for large folders / many documents to avoid client
    HTTP timeouts. Poll /summarize-folder-status/{job_id} to retrieve the
    result once processing completes."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "result": None,
        "error": None,
    }
    # Dispatched to the background via create_task -- this endpoint returns
    # right after this line, while processing continues in parallel.
    asyncio.create_task(_run_job_in_background(job_id, req))
    return JobStartResponse(job_id=job_id, status="running")


@app.get("/summarize-folder-status/{job_id}", response_model=JobStatusResponse)
async def summarize_folder_status(job_id: str):
    """Returns the current status of a background summarization job.
    status='running': still processing, poll again later.
    status='completed': check the 'result' field.
    status='failed': check the 'error' field."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job found with id: {job_id}")
    return JobStatusResponse(job_id=job_id, **job)
