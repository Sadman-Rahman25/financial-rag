"""
src/index.py
Embed all chunks with BGE-large-en-v1.5 and index them in Qdrant local file mode.

Usage:
    python -m src.index --test-embed
    python -m src.index --setup
    python -m src.index --build
    python -m src.index --count
    python -m src.index --test-query "query text" [--ticker AAPL] [--fiscal-year 2024]
                                                  [--chunk-type prose] [--limit 5]
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

torch.set_num_threads(4)

CHUNKS_PATH = Path("data/chunks.jsonl")
QDRANT_PATH = Path("./qdrant_data")
COLLECTION_NAME = "financial_rag"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
VECTOR_SIZE = 1024


def get_embedder() -> SentenceTransformer:
    print(f"Loading {MODEL_NAME} ...", flush=True)
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Loaded. Max sequence length: {model.max_seq_length} tokens", flush=True)
    return model


def _test_embedder() -> None:
    model = get_embedder()
    test_sentence = "Apple reported strong iPhone revenue in fiscal year 2024."
    vec = model.encode(test_sentence, normalize_embeddings=True)
    print(f"Test sentence: {test_sentence!r}")
    print(f"Vector shape:  {vec.shape}")
    print(f"Vector dtype:  {vec.dtype}")
    print(f"Norm (should be ~1.0): {np.linalg.norm(vec):.4f}")
    assert vec.shape == (VECTOR_SIZE,), f"Expected ({VECTOR_SIZE},), got {vec.shape}"
    print("OK: embedder works, produces 1024-dim normalized vectors.")


def get_qdrant_client() -> QdrantClient:
    QDRANT_PATH.mkdir(exist_ok=True)
    return QdrantClient(path=str(QDRANT_PATH))


def setup_collection() -> None:
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        n = client.count(collection_name=COLLECTION_NAME, exact=True).count
        print(f"Collection {COLLECTION_NAME!r} already exists with {n} points.")
        print(f"To recreate from scratch, delete {QDRANT_PATH}/ and re-run.")
        return

    print(f"Creating collection {COLLECTION_NAME!r} (size={VECTOR_SIZE}, distance=Cosine)...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    indexes = [
        ("ticker", PayloadSchemaType.KEYWORD),
        ("fiscal_year", PayloadSchemaType.INTEGER),
        ("section", PayloadSchemaType.KEYWORD),
        ("chunk_type", PayloadSchemaType.KEYWORD),
    ]
    for field, schema in indexes:
        print(f"  Creating payload index on {field}...")
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=schema,
        )
    n = client.count(collection_name=COLLECTION_NAME, exact=True).count
    print(f"OK: collection created. Points: {n} (expected 0). Payload indexes: {len(indexes)}.")


def build_index() -> None:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"{CHUNKS_PATH} not found. Run Day 3 chunking first.")

    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        raise RuntimeError(f"Collection {COLLECTION_NAME!r} doesn't exist. Run --setup first.")

    n_existing = client.count(collection_name=COLLECTION_NAME, exact=True).count
    if n_existing > 0:
        print(f"Note: collection already has {n_existing} points. Upsert will replace by ID.", flush=True)

    print(f"Loading chunks from {CHUNKS_PATH}...", flush=True)
    chunks = []
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"  Loaded {len(chunks)} chunks.", flush=True)

    model = get_embedder()

    EMBED_BATCH = 1
    UPSERT_BATCH = 32

    n_batches = (len(chunks) + UPSERT_BATCH - 1) // UPSERT_BATCH
    print(f"Embedding+upserting {len(chunks)} chunks in {n_batches} groups of {UPSERT_BATCH} "
          f"(internal encode batch={EMBED_BATCH})...", flush=True)
    t0 = time.time()
    n_done = 0

    for batch_idx, batch_start in enumerate(range(0, len(chunks), UPSERT_BATCH)):
        batch = chunks[batch_start:batch_start + UPSERT_BATCH]
        texts = [c["text"] for c in batch]

        print(f"  [Batch {batch_idx+1:3d}/{n_batches}] encoding {len(texts)} chunks...", flush=True)
        try:
            vectors = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=EMBED_BATCH,
            )
        except Exception as e:
            print(f"  ENCODE FAILED at batch {batch_idx+1}: {type(e).__name__}: {e}", flush=True)
            raise

        points = []
        for i, (chunk, vec) in enumerate(zip(batch, vectors)):
            point_id = batch_start + i
            payload = dict(chunk["metadata"])
            payload["text"] = chunk["text"]
            points.append(PointStruct(id=point_id, vector=vec.tolist(), payload=payload))

        try:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
        except Exception as e:
            print(f"  UPSERT FAILED at batch {batch_idx+1}: {type(e).__name__}: {e}", flush=True)
            raise

        n_done += len(batch)
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0
        print(f"  [Batch {batch_idx+1:3d}/{n_batches}] upserted | {n_done}/{len(chunks)} | {rate:.1f}/s",
              flush=True)

    elapsed = time.time() - t0
    final_count = client.count(collection_name=COLLECTION_NAME, exact=True).count
    print(f"OK: indexed {n_done} chunks in {elapsed:.1f}s. Collection has {final_count} points.",
          flush=True)


def count_points() -> None:
    """Print number of points in the collection."""
    client = get_qdrant_client()
    n = client.count(collection_name=COLLECTION_NAME, exact=True).count
    print(f"Collection {COLLECTION_NAME!r}: {n} points")


def test_query(
    query: str,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    chunk_type: str | None = None,
    limit: int = 5,
) -> None:
    """Run a semantic search with optional metadata filters and print top results."""
    client = get_qdrant_client()
    model = get_embedder()

    print(f"\nQuery: {query!r}")
    filters = []
    if ticker:
        filters.append(FieldCondition(key="ticker", match=MatchValue(value=ticker)))
        print(f"  Filter: ticker = {ticker}")
    if fiscal_year:
        filters.append(FieldCondition(key="fiscal_year", match=MatchValue(value=fiscal_year)))
        print(f"  Filter: fiscal_year = {fiscal_year}")
    if chunk_type:
        filters.append(FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type)))
        print(f"  Filter: chunk_type = {chunk_type}")
    query_filter = Filter(must=filters) if filters else None

    vec = model.encode(query, normalize_embeddings=True)
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec.tolist(),
        limit=limit,
        query_filter=query_filter,
    )
    results = response.points

    print(f"\nTop {len(results)} results:")
    print("-" * 100)
    for i, point in enumerate(results, 1):
        meta = point.payload
        text = meta.get("text", "")
        preview = text[:240].replace("\n", " ").strip()
        section = meta.get("section", "")[:90]
        print(f"[{i}] score={point.score:.4f} | {meta.get('ticker')} FY{meta.get('fiscal_year')} | "
              f"{meta.get('chunk_type')} | {meta.get('n_tokens')} tok")
        print(f"    Section:    {section}")
        if meta.get("subsection"):
            print(f"    Subsection: {meta['subsection']}")
        print(f"    Text:       {preview}...")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-embed", action="store_true", help="Stage 2: verify embedder")
    parser.add_argument("--setup", action="store_true", help="Stage 3: create empty Qdrant collection")
    parser.add_argument("--build", action="store_true", help="Stage 4: embed and upsert all chunks")
    parser.add_argument("--count", action="store_true", help="Stage 5: print point count")
    parser.add_argument("--test-query", type=str, default=None, help="Stage 5: semantic search")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--fiscal-year", type=int, default=None)
    parser.add_argument("--chunk-type", type=str, default=None)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    if args.test_embed:
        _test_embedder()
    elif args.setup:
        setup_collection()
    elif args.build:
        build_index()
    elif args.count:
        count_points()
    elif args.test_query is not None:
        test_query(
            query=args.test_query,
            ticker=args.ticker,
            fiscal_year=args.fiscal_year,
            chunk_type=args.chunk_type,
            limit=args.limit,
        )
    else:
        parser.print_help()