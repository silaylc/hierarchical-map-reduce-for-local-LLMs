"""
core/summarizer.py
===================
Async, hierarchical (recursive) Map-Reduce summarization engine.

Why hierarchical reduce?
If a document is split into 80-100 chunks, merging all intermediate
summaries in a SINGLE reduce call can exceed the reduce step's own
context window. Instead, summaries are first merged in small groups
(REDUCE_BATCH), and the result is fed through the same process again
(until a single summary remains) -- see hierarchical_reduce().

Why a structured Map prompt?
A plain "summarize this" instruction can skip important details
(numbers, dates, risks) at chunk boundaries. Explicitly telling the
model what kinds of information to extract during the Map stage
enriches the raw material that goes into the Reduce stage.

Why paragraph-aware (structural) chunking?
Naive character-based splitting can cut a paragraph/claim/section
in half -- leaving that piece without context during the Map stage.
_split_structural() first splits the text into paragraphs, then
greedily groups paragraphs without exceeding chunk_size; only a
single paragraph that alone exceeds chunk_size falls back to
character-based splitting (RecursiveCharacterTextSplitter). This
preserves paragraph/claim integrity in most cases.

Why a separate normalization stage?
Intermediate summaries produced at different levels of the hierarchical
reduce can be slightly inconsistent in tone/format (one bulleted, one
prose). After reduce, an extra LLM call (normalize_summary) that fixes
ONLY style and formatting WITHOUT ADDING/REMOVING CONTENT resolves this
inconsistency. Can be disabled with config.enable_normalization=False.

Why token-budget-aware (dynamic) reduce batching?
A fixed reduce_batch_size (e.g. 5) doesn't know the model's actual
context limit (--max-model-len in vLLM). With small-context models
(e.g. 2048), 5 intermediate summaries + the prompt template + the
requested output can easily exceed the limit, and vLLM returns an
explicit "context length exceeded" error -- since this error is
deterministic, the retry mechanism (_call_llm) can't recover from it.
_pack_reduce_batches() greedily groups summaries according to the
config.max_context_tokens budget: it starts a new batch whenever
adding the next summary would exceed the budget. This way
reduce_batch_size only acts as an UPPER BOUND, and the actual group
size shrinks according to the budget. If a single summary (rare, very
long) exceeds the budget on its own, it is truncated to fit the budget
so progress isn't blocked (information loss only happens in this
cheap, last-resort case).
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


MAP_PROMPT_TEMPLATE = """Extract the following information from the document
excerpt below, in English:

1. Main topic
2. Key facts / findings
3. Technical details
4. Important numbers and dates
5. Decisions / conclusions
6. Risks / warnings (if any)

Rules:
- Only use information found in the given text, do not invent new information.
- Preserve technical terms and proper nouns.
- Do not exceed 150-250 words.

DOCUMENT EXCERPT:
{chunk}
"""

REDUCE_PROMPT_TEMPLATE = """Combine the excerpt summaries below into a single,
coherent document summary.

Rules:
- Minimize information loss; merge repeated information.
- If there is conflicting information, do not invent a resolution -- state
  the conflict as it is.
- Preserve important technical details and numerical values.
- Clearly state the main purpose, method, findings, and conclusions.
- Write in English.

EXCERPT SUMMARIES:
{combined}
"""

NORMALIZE_PROMPT_TEMPLATE = """Make the document summary below consistent in
style and format ONLY, WITHOUT CHANGING ITS CONTENT.

Do NOT, under any circumstances:
- Add new information, numbers, dates, or claims.
- Remove existing information or details.
- Change the order or number of bullet points/paragraphs.
- Reinterpret in a way that changes the meaning.

DO:
- Rephrase the entire text in a consistent English tone (formal, objective).
- Keep sentence endings consistent (e.g. all in the same tense/mood --
  but don't repeat the exact same wording; you may use synonymous
  phrasing for natural variety).
- Keep the formatting consistent (either all bullet points or all plain
  paragraphs, don't mix).

ORIGINAL SUMMARY:
{summary}

NORMALIZED SUMMARY:
"""


@dataclass
class SummarizerConfig:
    # --- Backend (vLLM / LiteLLM / any OpenAI-compatible endpoint) ---
    # Defaults are read from an environment variable first, falling back
    # to a fixed value. Pass the service name via env in docker-compose
    # (e.g. LLM_BASE_URL=http://llm:8001/v1) -- no code changes needed.
    base_url: str = field(
        default_factory=lambda: os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
    )
    model: str = field(
        default_factory=lambda: os.environ.get("LLM_MODEL", "/model")
    )
    api_key: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY", "not-needed"))
    temperature: float = 0.1

    # --- Chunking ---
    chunk_size: int = 1800          # characters (~400-600 tokens, varies by tokenizer)
    chunk_overlap: int = 250
    structural_chunking: bool = True  # split while preserving paragraph boundaries (see module docstring)

    # --- Map-Reduce behavior ---
    map_max_tokens: int = 400
    reduce_max_tokens: int = 500
    reduce_batch_size: int = 5      # UPPER BOUND group size in hierarchical reduce (see max_context_tokens)
    concurrency: int = 1            # GTX 3050 -> 1. H200 -> 8-32.

    # --- Context budget (for dynamic reduce batching) ---
    # Set this to the SAME value as --max-model-len in vLLM (default 2048).
    # If this value is wrong, reduce batches end up unnecessarily small/large.
    max_context_tokens: int = field(
        default_factory=lambda: int(os.environ.get("LLM_MAX_MODEL_LEN", "2048"))
    )
    # We don't use a real tokenizer for token counting (extra dependency/
    # slowdown), and instead APPROXIMATE with a rough character/token ratio.
    # 3.0 = conservative (OVERESTIMATES token count), so we stay on the safe
    # side even if the real tokenizer counts somewhat differently.
    chars_per_token: float = 3.0
    # Extra buffer for special tokens added by the chat template, system
    # prompt, etc.
    context_safety_margin: int = 150

    # --- Normalization (post-reduce style/format consistency) ---
    enable_normalization: bool = True
    normalize_max_tokens: int = 600  # keep a bit larger than the final summary


@dataclass
class DocumentMetrics:
    file_name: str
    char_count: int = 0
    chunk_count: int = 0
    map_calls: int = 0
    reduce_calls: int = 0
    map_seconds: float = 0.0
    reduce_seconds: float = 0.0
    normalize_seconds: float = 0.0
    normalize_calls: int = 0
    total_seconds: float = 0.0


class DocumentSummarizer:
    def __init__(self, config: Optional[SummarizerConfig] = None):
        self.config = config or SummarizerConfig()
        self.llm = ChatOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            model=self.config.model,
            temperature=self.config.temperature,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        # Semaphore to avoid sending more than one chunk to the GPU at a
        # time on a GTX 3050. On larger cards like the H200, parallelism
        # can be increased via config.concurrency.
        self._semaphore = asyncio.Semaphore(self.config.concurrency)

    async def _call_llm(self, prompt: str, max_tokens: int, retries: int = 3) -> str:
        """LLM call with retry handling and semaphore protection.
        3 attempts with a 3s wait in between, for transient connection
        drops (e.g. vLLM briefly not responding under load)."""
        for attempt in range(retries):
            try:
                async with self._semaphore:
                    response = await self.llm.ainvoke(
                        prompt, max_tokens=max_tokens
                    )
                return response.content
            except Exception as e:
                if attempt == retries - 1:
                    raise e
                print(f"[WARNING] LLM connection dropped (attempt {attempt+1}/{retries}). "
                      f"Retrying in 3 seconds... Error: {e}")
                await asyncio.sleep(3)

    def _split_text(self, text: str) -> list[str]:
        """Structural (paragraph-aware) or plain character-based splitting.
        Uses _split_structural if config.structural_chunking=True (default);
        otherwise falls back directly to RecursiveCharacterTextSplitter.
        """
        if self.config.structural_chunking:
            return self._split_structural(text)
        return self.splitter.split_text(text)

    def _split_structural(self, text: str) -> list[str]:
        """First splits the text into paragraphs (blocks separated by a
        blank line), then greedily merges these paragraphs without
        exceeding chunk_size -- preserving paragraph/claim integrity in
        most cases. Only a paragraph that alone exceeds chunk_size falls
        back to character-based splitting (self.splitter) for that
        paragraph only.
        """
        raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not raw_paragraphs:
            return []

        # Pre-split any individual paragraphs that are too long into
        # character-based chunks.
        paragraphs: list[str] = []
        for p in raw_paragraphs:
            if len(p) > self.config.chunk_size:
                paragraphs.extend(self.splitter.split_text(p))
            else:
                paragraphs.append(p)

        chunks: list[str] = []
        current = ""
        for p in paragraphs:
            candidate = f"{current}\n\n{p}" if current else p
            if len(candidate) <= self.config.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Overlap: carry over the tail end of the previous chunk into
                # the new one so context isn't lost entirely.
                # The tail is only carried over if the new chunk still fits in
                # chunk_size; otherwise the chunk starts with the paragraph alone.
                tail = current[-self.config.chunk_overlap:] if self.config.chunk_overlap > 0 else ""
                with_tail = f"{tail}\n\n{p}" if tail else p
                current = with_tail if len(with_tail) <= self.config.chunk_size else p
        if current:
            chunks.append(current)

        return chunks

    def _estimate_tokens(self, text: str) -> int:
        """Rough, CONSERVATIVE token estimate (not a real tokenizer).
        Since chars_per_token is kept low, this generally OVERESTIMATES --
        the goal is to stay on the safe side, not precise measurement."""
        return max(1, int(len(text) / self.config.chars_per_token))

    def _reduce_prompt_overhead_tokens(self) -> int:
        """The fixed (non-summary) part of REDUCE_PROMPT_TEMPLATE + safety margin."""
        template_only = REDUCE_PROMPT_TEMPLATE.format(combined="")
        return self._estimate_tokens(template_only) + self.config.context_safety_margin

    def _pack_reduce_batches(self, summaries: list[str]) -> list[list[str]]:
        """Greedily groups summaries so a single reduce call doesn't exceed
        the context budget. reduce_batch_size is also enforced as an UPPER
        BOUND (to keep the number of parallel calls reasonable when there
        are too many small summaries).
        Budget = max_context_tokens - prompt_overhead - reduce_max_tokens (for output).
        """
        overhead = self._reduce_prompt_overhead_tokens()
        budget = self.config.max_context_tokens - overhead - self.config.reduce_max_tokens
        budget = max(budget, 200)  # guarantee progress even with extremely tight configs

        batches: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0

        for s in summaries:
            s_tokens = self._estimate_tokens(s)

            # If a single summary (rare) exceeds the budget on its own,
            # truncate it to fit the budget so progress isn't blocked. This
            # is the ONLY place information loss can occur -- a last resort,
            # not part of the main flow.
            if s_tokens > budget:
                max_chars = int(budget * self.config.chars_per_token)
                s = s[:max_chars]
                s_tokens = budget

            would_exceed = current_tokens + s_tokens > budget
            would_exceed_count = len(current) >= self.config.reduce_batch_size
            if current and (would_exceed or would_exceed_count):
                batches.append(current)
                current = []
                current_tokens = 0

            current.append(s)
            current_tokens += s_tokens

        if current:
            batches.append(current)

        return batches

    async def summarize_chunk(self, chunk: str) -> str:
        prompt = MAP_PROMPT_TEMPLATE.format(chunk=chunk)
        return await self._call_llm(prompt, self.config.map_max_tokens)

    async def reduce_batch(self, summaries: list[str]) -> str:
        combined = "\n\n".join(f"SUMMARY {i + 1}:\n{s}" for i, s in enumerate(summaries))
        prompt = REDUCE_PROMPT_TEMPLATE.format(combined=combined)
        return await self._call_llm(prompt, self.config.reduce_max_tokens)

    async def hierarchical_reduce(self, summaries: list[str], metrics: DocumentMetrics) -> str:
        current = summaries
        while len(current) > 1:
            batches = self._pack_reduce_batches(current)

            # Safety net: if packing produced no reduction at all
            # (theoretically, if every summary alone strains the budget),
            # force splitting into pairs to guarantee progress -- avoid an
            # infinite loop.
            if len(batches) >= len(current):
                batches = [current[i:i + 2] for i in range(0, len(current), 2)]

            t0 = time.monotonic()
            # Reduce calls at the same level are independent of each other;
            # the semaphore already limits how many requests hit the GPU at
            # once, so gather() is safe here.
            next_level = await asyncio.gather(*(self.reduce_batch(b) for b in batches))
            metrics.reduce_seconds += time.monotonic() - t0
            metrics.reduce_calls += len(batches)
            current = list(next_level)
        return current[0]

    async def normalize_summary(self, summary: str) -> str:
        """Makes the final summary produced after reduce consistent in
        style and format only, WITHOUT ADDING/REMOVING CONTENT. See the
        module docstring -- intermediate summaries from different levels
        of the hierarchical reduce can be slightly inconsistent in
        tone/format.
        """
        prompt = NORMALIZE_PROMPT_TEMPLATE.format(summary=summary)
        return await self._call_llm(prompt, self.config.normalize_max_tokens)

    async def summarize_document(self, text: str, file_name: str = "") -> tuple[str, DocumentMetrics]:
        t_start = time.monotonic()
        metrics = DocumentMetrics(file_name=file_name, char_count=len(text))

        chunks = self._split_text(text)
        metrics.chunk_count = len(chunks)

        if not chunks:
            metrics.total_seconds = time.monotonic() - t_start
            return "(Empty document or no text could be extracted.)", metrics

        if len(chunks) == 1:
            summary = await self.summarize_chunk(chunks[0])
            metrics.map_calls = 1
        else:
            t0 = time.monotonic()
            chunk_summaries = await asyncio.gather(*(self.summarize_chunk(c) for c in chunks))
            metrics.map_seconds = time.monotonic() - t0
            metrics.map_calls = len(chunks)

            summary = await self.hierarchical_reduce(list(chunk_summaries), metrics)

        if self.config.enable_normalization:
            t0 = time.monotonic()
            summary = await self.normalize_summary(summary)
            metrics.normalize_seconds = time.monotonic() - t0
            metrics.normalize_calls = 1

        metrics.total_seconds = time.monotonic() - t_start
        return summary, metrics
