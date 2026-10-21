"""End-to-end document summarization with a fake LLM, and the retry logic."""

import asyncio
from types import SimpleNamespace

from conftest import make_summarizer, paragraphs


def test_empty_document_makes_no_llm_calls():
    s, fake = make_summarizer()
    summary, metrics = asyncio.run(s.summarize_document("   ", "empty.txt"))
    assert fake.calls == []
    assert metrics.chunk_count == 0
    assert "Empty document" in summary


def test_short_document_is_one_map_call_plus_normalization():
    s, fake = make_summarizer()
    summary, metrics = asyncio.run(s.summarize_document(paragraphs(2, 200), "short.txt"))
    assert fake.calls == ["map", "normalize"]
    assert metrics.map_calls == 1 and metrics.reduce_calls == 0
    assert summary == "normalized summary"


def test_long_document_maps_every_chunk_then_reduces():
    s, fake = make_summarizer(max_context_tokens=2048)
    _, metrics = asyncio.run(s.summarize_document(paragraphs(80, 900), "long.txt"))
    assert metrics.chunk_count > 1
    assert metrics.map_calls == metrics.chunk_count == fake.count("map")
    assert metrics.reduce_calls == fake.count("reduce") >= 1
    assert fake.calls[-1] == "normalize"


def test_normalization_can_be_disabled():
    s, fake = make_summarizer(enable_normalization=False)
    summary, metrics = asyncio.run(s.summarize_document(paragraphs(2, 200), "short.txt"))
    assert "normalize" not in fake.calls
    assert metrics.normalize_calls == 0
    assert summary == "m" * fake.map_len


class _FlakyLLM:
    """Fails a given number of times, then answers."""

    def __init__(self, failures: int):
        self.failures = failures
        self.attempts = 0

    async def ainvoke(self, prompt, max_tokens):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise ConnectionError("backend unavailable")
        return SimpleNamespace(content="ok")


def _no_wait_retry(s, failures):
    """Calls the real _call_llm with a flaky backend and without the 3 s waits."""
    from core import summarizer as mod

    s.llm = _FlakyLLM(failures)
    real_sleep = mod.asyncio.sleep

    async def no_sleep(_):
        return None

    mod.asyncio.sleep = no_sleep
    try:
        return asyncio.run(type(s)._call_llm(s, "prompt", max_tokens=10))
    finally:
        mod.asyncio.sleep = real_sleep


def test_transient_failures_are_retried():
    s, _ = make_summarizer()
    assert _no_wait_retry(s, failures=2) == "ok"
    assert s.llm.attempts == 3


def test_persistent_failure_is_raised_after_three_attempts():
    s, _ = make_summarizer()
    try:
        _no_wait_retry(s, failures=10)
    except ConnectionError:
        pass
    else:
        raise AssertionError("expected ConnectionError")
    assert s.llm.attempts == 3
