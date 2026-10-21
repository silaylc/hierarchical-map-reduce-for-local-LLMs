"""Context-budget-aware reduce batching and the hierarchical reduce loop."""

import asyncio

from conftest import make_summarizer
from core.summarizer import DocumentMetrics


def _budget(s):
    return max(
        s.config.max_context_tokens
        - s._reduce_prompt_overhead_tokens()
        - s.config.reduce_max_tokens,
        200,
    )


def test_batches_respect_token_budget_and_batch_size():
    for ctx in (2048, 4096, 8192):
        s, _ = make_summarizer(max_context_tokens=ctx, reduce_batch_size=5)
        summaries = [f"s{i}" + "y" * 900 for i in range(40)]
        batches = s._pack_reduce_batches(summaries)
        for batch in batches:
            assert len(batch) <= 5
            assert sum(s._estimate_tokens(x) for x in batch) <= _budget(s), ctx


def test_batching_keeps_every_summary_in_order():
    s, _ = make_summarizer(max_context_tokens=2048)
    summaries = [f"s{i:02d}" + "y" * 700 for i in range(25)]
    batches = s._pack_reduce_batches(summaries)
    assert [x for b in batches for x in b] == summaries


def test_small_context_gives_smaller_batches():
    small, _ = make_summarizer(max_context_tokens=2048, reduce_batch_size=10)
    large, _ = make_summarizer(max_context_tokens=16384, reduce_batch_size=10)
    summaries = ["y" * 900 for _ in range(30)]
    assert len(small._pack_reduce_batches(summaries)) > len(large._pack_reduce_batches(summaries))


def test_oversized_summary_is_truncated_to_budget():
    s, _ = make_summarizer(max_context_tokens=2048)
    huge = "z" * 50_000
    batches = s._pack_reduce_batches([huge])
    assert len(batches) == 1
    assert s._estimate_tokens(batches[0][0]) <= _budget(s)


def test_hierarchical_reduce_ends_with_one_summary():
    s, fake = make_summarizer(max_context_tokens=2048)
    metrics = DocumentMetrics(file_name="doc")
    result = asyncio.run(s.hierarchical_reduce(["m" * 600] * 60, metrics))
    assert result == "r" * fake.reduce_len
    assert metrics.reduce_calls == fake.count("reduce") > 1


def test_hierarchical_reduce_terminates_even_with_a_tiny_context():
    s, fake = make_summarizer(max_context_tokens=256, reduce_max_tokens=200)
    fake.reduce_len = 5000  # merged summaries never get shorter
    metrics = DocumentMetrics(file_name="doc")
    asyncio.run(s.hierarchical_reduce(["m" * 2000] * 16, metrics))
    assert metrics.reduce_calls == 15  # pairwise fallback: 8 + 4 + 2 + 1
