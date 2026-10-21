"""API helpers: host/container path translation and per-document fault isolation."""

import asyncio
import os
import tempfile
from pathlib import Path

from conftest import FakeLLM

import api.main as api


def _with_env(env: dict, fn):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: v for k, v in env.items() if v is not None})
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
    try:
        return fn()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_path_is_unchanged_without_prefix_env():
    env = {"HOST_PATH_PREFIX": None, "CONTAINER_PATH_PREFIX": None}
    assert _with_env(env, lambda: api._translate_path("C:/Users/me/docs")) == "C:/Users/me/docs"


def test_host_path_is_translated_to_container_path():
    env = {"HOST_PATH_PREFIX": "C:\\Users\\me", "CONTAINER_PATH_PREFIX": "/data"}
    out = _with_env(env, lambda: api._translate_path("c:\\users\\me\\Reports\\Q3"))
    assert out == "/data/Reports/Q3"


def test_unrelated_path_is_left_alone():
    env = {"HOST_PATH_PREFIX": "C:/Users/me", "CONTAINER_PATH_PREFIX": "/data"}
    assert _with_env(env, lambda: api._translate_path("/srv/other")) == "/srv/other"


def test_one_broken_file_does_not_stop_the_batch():
    original = api.DocumentSummarizer._call_llm
    api.DocumentSummarizer._call_llm = FakeLLM().__call__
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "broken.pdf").write_bytes(b"not a real pdf")
            (root / "empty.txt").write_text("   ", encoding="utf-8")
            (root / "good.txt").write_text("A short report.\n\nWith two paragraphs.", encoding="utf-8")
            req = api.FolderRequest(folder=tmp, save_to_folder=False)
            result = asyncio.run(api._process_folder(req))
    finally:
        api.DocumentSummarizer._call_llm = original

    by_name = {d.file: d for d in result.documents}
    assert result.document_count == 3
    assert by_name["broken.pdf"].summary.startswith("[ERROR] Could not read document")
    assert by_name["empty.txt"].summary == "(Empty document, skipped.)"
    assert by_name["good.txt"].summary == "normalized summary"
    assert by_name["good.txt"].map_calls == 1


def test_requested_files_that_do_not_exist_are_reported():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "a.txt").write_text("text", encoding="utf-8")
        req = api.FolderRequest(folder=tmp, files=["missing.pdf"], save_to_folder=False)
        result = asyncio.run(api._process_folder(req))
    assert result.document_count == 0
    assert result.not_found == ["missing.pdf"]
