"""
standalone_cli.py
==================
DEVELOPMENT STAGE 1: OpenWebUI/FastAPI is not involved at all yet.
Verify the summarization engine itself (chunking + map + hierarchical
reduce) here first, then turn it into a service with api/main.py.

USAGE:
  python standalone_cli.py --folder ./documents --out ./outputs \
      --base-url http://localhost:8000/v1 \
      --model casperhansen/llama-3.2-3b-instruct-awq
"""

import argparse
import asyncio
import json
from pathlib import Path

from core.loaders import scan_folder, load_document
from core.summarizer import DocumentSummarizer, SummarizerConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True)
    p.add_argument("--out", default="./outputs")
    p.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM OpenAI-compatible endpoint")
    p.add_argument("--model", default="casperhansen/llama-3.2-3b-instruct-awq")
    p.add_argument("--chunk-size", type=int, default=1800)
    p.add_argument("--chunk-overlap", type=int, default=250)
    p.add_argument("--reduce-batch-size", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=1, help="keep at 1 for a GTX 3050")
    return p.parse_args()


async def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    config = SummarizerConfig(
        base_url=args.base_url,
        model=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        reduce_batch_size=args.reduce_batch_size,
        concurrency=args.concurrency,
    )
    summarizer = DocumentSummarizer(config)

    documents = scan_folder(args.folder)
    print(f"Found {len(documents)} document(s).")

    all_metrics = []
    for path in documents:
        print("\n" + "=" * 60)
        print(f"Processing: {path.name}")

        text = load_document(path)
        if not text.strip():
            print("Empty document, skipped.")
            continue

        summary, metrics = await summarizer.summarize_document(text, file_name=path.name)

        out_file = out_dir / f"{path.stem}_summary.md"
        out_file.write_text(f"# {path.name}\n\n{summary}\n", encoding="utf-8")

        print(f"Chunks: {metrics.chunk_count} | Map: {metrics.map_calls} calls "
              f"({metrics.map_seconds:.1f}s) | Reduce: {metrics.reduce_calls} calls "
              f"({metrics.reduce_seconds:.1f}s) | Normalize: {metrics.normalize_calls} calls "
              f"({metrics.normalize_seconds:.1f}s) | Total: {metrics.total_seconds:.1f}s")
        print(f"Saved: {out_file}")

        all_metrics.append(vars(metrics))

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    asyncio.run(main())
