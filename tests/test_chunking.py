"""Paragraph-aware (structural) chunking."""

from conftest import make_summarizer, paragraphs


def test_empty_text_gives_no_chunks():
    s, _ = make_summarizer()
    assert s._split_text("") == []
    assert s._split_text("\n\n   \n\n") == []


def test_short_text_is_a_single_chunk():
    s, _ = make_summarizer()
    text = paragraphs(3, 100)
    assert s._split_text(text) == [text]


def test_no_chunk_exceeds_chunk_size():
    for overlap in (0, 250, 900):
        s, _ = make_summarizer(chunk_size=1800, chunk_overlap=overlap)
        for length in (300, 700, 1200, 1700):
            chunks = s._split_text(paragraphs(40, length))
            assert chunks
            assert max(len(c) for c in chunks) <= 1800, (overlap, length)


def test_paragraphs_are_not_cut_in_half():
    s, _ = make_summarizer(chunk_size=1800, chunk_overlap=0)
    text = paragraphs(30, 500)
    original = set(text.split("\n\n"))
    for chunk in s._split_text(text):
        for piece in chunk.split("\n\n"):
            assert piece in original


def test_every_paragraph_is_kept_in_order():
    s, _ = make_summarizer(chunk_size=1800, chunk_overlap=0)
    text = paragraphs(30, 500)
    rebuilt = "\n\n".join(s._split_text(text))
    assert rebuilt == text


def test_overlap_carries_previous_tail_when_it_fits():
    s, _ = make_summarizer(chunk_size=1800, chunk_overlap=100)
    chunks = s._split_text(paragraphs(10, 500))
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.startswith(prev[-100:])


def test_single_oversized_paragraph_falls_back_to_character_split():
    s, _ = make_summarizer(chunk_size=1000, chunk_overlap=0)
    words = " ".join(f"w{i}" for i in range(2000))  # one paragraph, ~10k chars
    chunks = s._split_text(words)
    assert len(chunks) > 1
    assert max(len(c) for c in chunks) <= 1000
