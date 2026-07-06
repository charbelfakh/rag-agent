#!/usr/bin/env python3
"""Embedder bake-off: compare embedding models on a corpus sample.

For each candidate model this script:
  1. samples chunks from the live collection (always including every chunk
     whose source appears in the golden set, plus random distractors),
  2. embeds the sample into a throwaway side collection ``bakeoff_<model>``,
  3. runs the golden questions (no HyDE — raw embedder quality) and reports
     recall@5 / recall@10 / MRR plus query-embed latency.

Requires live Qdrant + Ollama. The production collection is only read.

Usage:
    python eval/run_embedder_bakeoff.py --models nomic-embed-text,bge-m3,mxbai-embed-large
    python eval/run_embedder_bakeoff.py --models bge-m3 --sample-size 3000 --keep-collections
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.metrics import recall_at_k, reciprocal_rank
from eval.run_retrieval_eval import load_dataset


def slugify_model(name: str) -> str:
    """'mxbai-embed-large:latest' → 'mxbai_embed_large_latest'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def select_sample(
    chunks: list[dict],
    expected_sources: set[str],
    sample_size: int,
    seed: int = 42,
) -> list[dict]:
    """Every expected-source chunk + deterministic random distractor fill."""
    required = [c for c in chunks if c.get("source") in expected_sources]
    others = [c for c in chunks if c.get("source") not in expected_sources]
    rng = random.Random(seed)
    rng.shuffle(others)
    fill = max(sample_size - len(required), 0)
    return required + others[:fill]


def summarize_run(per_item: list[dict], embed_ms: list[int]) -> dict:
    count = len(per_item)
    if not count:
        return {"count": 0}
    return {
        "count": count,
        "recall_at_5": sum(i["recall_at_5"] for i in per_item) / count,
        "recall_at_10": sum(i["recall_at_10"] for i in per_item) / count,
        "mrr": sum(i["mrr"] for i in per_item) / count,
        "mean_query_embed_ms": sum(embed_ms) / len(embed_ms) if embed_ms else None,
    }


def scroll_corpus_chunks(client, collection: str, page_size: int = 500) -> list[dict]:
    """Read (text, source) for all chunks in the source collection."""
    chunks: list[dict] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection,
            limit=page_size,
            offset=offset,
            with_payload=["text", "text_full", "source", "content_type"],
            with_vectors=False,
        )
        if not records:
            break
        for record in records:
            payload = record.payload or {}
            text = payload.get("text_full") or payload.get("text") or ""
            source = payload.get("source") or ""
            if text.strip() and source:
                chunks.append({"text": text, "source": source})
        if offset is None:
            break
    return chunks


def _make_embedder(model: str):
    from providers.ollama_embed import OllamaEmbedder

    embedder = OllamaEmbedder()
    embedder.model = model
    return embedder


def run_model_bakeoff(
    *,
    client,
    model: str,
    sample: list[dict],
    dataset: list[dict],
    collection_prefix: str,
    embed_batch_size: int,
    keep_collection: bool,
) -> dict:
    from qdrant_client.models import Distance, PointStruct, VectorParams

    embedder = _make_embedder(model)
    collection = f"{collection_prefix}_{slugify_model(model)}"

    # Embed + upsert the sample.
    t_index = time.perf_counter()
    created = False
    indexed = 0
    for start in range(0, len(sample), embed_batch_size):
        batch = sample[start : start + embed_batch_size]
        vectors = embedder.embed([c["text"] for c in batch])
        pairs = [
            (c, v) for c, v in zip(batch, vectors) if v is not None and len(v) > 0
        ]
        if not pairs:
            continue
        if not created:
            names = {c.name for c in client.get_collections().collections}
            if collection in names:
                client.delete_collection(collection)
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=len(pairs[0][1]), distance=Distance.COSINE
                ),
            )
            created = True
        points = [
            PointStruct(
                id=indexed + i,
                vector=vector,
                payload={"source": chunk["source"]},
            )
            for i, (chunk, vector) in enumerate(pairs)
        ]
        client.upsert(collection_name=collection, points=points)
        indexed += len(points)
    index_s = time.perf_counter() - t_index
    if not created:
        raise RuntimeError(f"No embeddings produced by model {model!r}")

    # Query the golden set.
    per_item: list[dict] = []
    embed_ms: list[int] = []
    for item in dataset:
        question = item["question"]
        expected = item.get("expected_sources", [])
        t_embed = time.perf_counter()
        vector = embedder.embed([question])[0]
        embed_ms.append(int((time.perf_counter() - t_embed) * 1000))
        results = client.query_points(
            collection_name=collection, query=vector, limit=10
        )
        sources = [
            (hit.payload or {}).get("source", "") for hit in results.points
        ]
        per_item.append(
            {
                "id": item.get("id", question[:40]),
                "recall_at_5": recall_at_k(sources, expected, 5),
                "recall_at_10": recall_at_k(sources, expected, 10),
                "mrr": reciprocal_rank(sources, expected),
            }
        )

    if not keep_collection:
        client.delete_collection(collection)

    summary = summarize_run(per_item, embed_ms)
    summary.update(
        {
            "model": model,
            "collection": collection,
            "chunks_indexed": indexed,
            "index_seconds": round(index_s, 1),
        }
    )
    return {"summary": summary, "items": per_item}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare embedding models on a corpus sample")
    parser.add_argument("--models", required=True, help="Comma-separated Ollama embed model names")
    parser.add_argument("--dataset", type=Path, default=ROOT / "eval" / "dataset.jsonl")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--collection-prefix", default="bakeoff")
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--keep-collections", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Write full JSON report")
    args = parser.parse_args()

    import os

    from qdrant_client import QdrantClient

    client = QdrantClient(url=os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333"))
    source_collection = os.getenv("QDRANT_COLLECTION", "rag_docs")

    dataset = load_dataset(args.dataset)
    expected_sources = {
        src for item in dataset for src in item.get("expected_sources", [])
    }

    print(f"Scrolling corpus chunks from '{source_collection}' …", file=sys.stderr)
    corpus = scroll_corpus_chunks(client, source_collection)
    sample = select_sample(corpus, expected_sources, args.sample_size, args.seed)
    print(
        f"Sample: {len(sample)} chunks "
        f"({sum(1 for c in sample if c['source'] in expected_sources)} from expected sources)",
        file=sys.stderr,
    )

    reports = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"=== {model} ===", file=sys.stderr)
        report = run_model_bakeoff(
            client=client,
            model=model,
            sample=sample,
            dataset=dataset,
            collection_prefix=args.collection_prefix,
            embed_batch_size=args.embed_batch_size,
            keep_collection=args.keep_collections,
        )
        print(json.dumps(report["summary"], indent=2))
        reports.append(report)

    if args.output:
        args.output.write_text(
            json.dumps([r["summary"] for r in reports], indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote report to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
