"""Shared helpers: a fake LLM so the pipeline can be tested without a backend."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.summarizer import DocumentSummarizer, SummarizerConfig  # noqa: E402


class FakeLLM:
    """Stands in for DocumentSummarizer._call_llm.

    Map prompts return a summary of a fixed length, reduce prompts return a
    shorter merged summary, and every call is recorded by stage.
    """

    def __init__(self, map_len: int = 600, reduce_len: int = 300):
        self.map_len = map_len
        self.reduce_len = reduce_len
        self.calls: list[str] = []

    async def __call__(self, prompt: str, max_tokens: int, retries: int = 3) -> str:
        if prompt.startswith("Extract the following information"):
            self.calls.append("map")
            return "m" * self.map_len
        if prompt.startswith("Combine the excerpt summaries"):
            self.calls.append("reduce")
            return "r" * self.reduce_len
        self.calls.append("normalize")
        return "normalized summary"

    def count(self, stage: str) -> int:
        return self.calls.count(stage)


def make_summarizer(**overrides) -> tuple[DocumentSummarizer, FakeLLM]:
    """A summarizer whose LLM calls go to a FakeLLM (no network access)."""
    config = SummarizerConfig(
        base_url="http://unused.invalid/v1", model="fake", **overrides
    )
    summarizer = DocumentSummarizer(config)
    fake = FakeLLM()
    summarizer._call_llm = fake
    return summarizer, fake


def paragraphs(n: int, length: int) -> str:
    """n distinct paragraphs of the given length, separated by blank lines."""
    return "\n\n".join(f"P{i:03d}" + "x" * (length - 4) for i in range(n))
