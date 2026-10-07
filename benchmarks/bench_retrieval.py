"""
NeuroShift -- Binary Retrieval Benchmark
=========================================
How much retrieval quality survives binary quantization, and what does it buy?

Part 1 (quality): a BEIR dataset with human relevance judgments (default
SciFact: 5,183 scientific abstracts, 300 test queries). Compares exact float32
search against binary codes, plus BM25 and the old lexical HDCRetrievalEngine
as references.

Part 2 (scale): search latency and memory over N synthetic codes.

Run:  python -m benchmarks.bench_retrieval [--dataset scifact] [--model NAME] [--scale-n 1000000]
Needs: pip install "neuroshift[embed]" datasets   (rank_bm25 optional)
"""

import argparse
import math
import os
import time
from typing import Callable, Dict, List

import numpy as np

from neuroshift.retrieval import BinaryIndex, SentenceTransformerEmbedder
from neuroshift.retrieval.hamming import hamming_distances, top_k


CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_beir(name: str = "scifact"):
    from datasets import load_dataset

    corpus = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    queries = load_dataset(f"BeIR/{name}", "queries", split="queries")
    qrels = load_dataset(f"BeIR/{name}-qrels", split="test")

    doc_ids = [str(d) for d in corpus["_id"]]
    doc_texts = [f"{t} {x}".strip() for t, x in zip(corpus["title"], corpus["text"])]

    relevant: Dict[str, Dict[str, int]] = {}
    for row in qrels:
        if row["score"] > 0:
            relevant.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])

    query_text = {str(q): t for q, t in zip(queries["_id"], queries["text"])}
    query_ids = [q for q in relevant if q in query_text]
    return doc_ids, doc_texts, query_ids, [query_text[q] for q in query_ids], relevant


def cached_embeddings(name: str, texts: List[str], fn: Callable) -> np.ndarray:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}.npy")
    if os.path.exists(path):
        cached = np.load(path)
        if len(cached) == len(texts):
            return cached
    vectors = fn(texts)
    np.save(path, vectors)
    return vectors


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def ndcg_at_k(ranked: List[str], rels: Dict[str, int], k: int = 10) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked: List[str], rels: Dict[str, int], k: int) -> float:
    return len(set(ranked[:k]) & set(rels)) / len(rels)


def evaluate(name, rankings, query_ids, relevant, exact_top10, bytes_per_doc, ms_per_query):
    ndcg = np.mean([ndcg_at_k(r, relevant[q]) for r, q in zip(rankings, query_ids)])
    rec = np.mean([recall_at_k(r, relevant[q], 100) for r, q in zip(rankings, query_ids)])
    overlap = (np.mean([len(set(r[:10]) & set(e)) / 10 for r, e in zip(rankings, exact_top10)])
               if exact_top10 is not None else float("nan"))
    return dict(name=name, ndcg=ndcg, recall100=rec, overlap=overlap,
                bytes_per_doc=bytes_per_doc, ms=ms_per_query)


def print_rows(rows, baseline_ndcg):
    header = f"{'Method':<40}{'nDCG@10':>9}{'% of f32':>10}{'R@100':>8}{'Top10 overlap':>15}{'Bytes/doc':>11}{'ms/query':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        overlap = "-" if math.isnan(r["overlap"]) else f"{r['overlap']:.3f}"
        print(f"{r['name']:<40}{r['ndcg']:>9.4f}{100 * r['ndcg'] / baseline_ndcg:>9.1f}%"
              f"{r['recall100']:>8.3f}{overlap:>15}{r['bytes_per_doc']:>11}{r['ms']:>10.2f}")


# ---------------------------------------------------------------------------
# Part 1: quality on SciFact
# ---------------------------------------------------------------------------
def bench_quality(dataset: str, model_name: str, query_prefix: str, include_hdc: bool):
    print(f"[1/2] Retrieval quality on BEIR {dataset}  (model: {model_name})")
    doc_ids, doc_texts, query_ids, query_texts, relevant = load_beir(dataset)
    print(f"      {len(doc_texts)} documents, {len(query_texts)} queries")

    embedder = SentenceTransformerEmbedder(model_name, query_prefix=query_prefix)
    slug = model_name.replace("/", "__")
    docs = cached_embeddings(f"{dataset}-docs-{slug}", doc_texts, embedder.embed)
    queries = cached_embeddings(f"{dataset}-queries-{slug}", query_texts, embedder.embed_queries)
    dim = docs.shape[1]
    k = 100

    def to_ids(index_rows):
        return [[doc_ids[i] for i in row] for row in index_rows]

    rows = []

    # Exact float32 baseline
    d = docs / np.linalg.norm(docs, axis=1, keepdims=True)
    q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    t0 = time.perf_counter()
    exact_idx = [top_k(row, k) for row in q @ d.T]
    ms = (time.perf_counter() - t0) / len(q) * 1000
    exact = to_ids(exact_idx)
    exact_top10 = [r[:10] for r in exact]
    rows.append(evaluate("float32 exact (baseline)", exact, query_ids, relevant, None, dim * 4, ms))

    configs = [
        ("sign", None, False), ("sign", None, True),
        ("simhash", 2 * dim, False), ("simhash", 2 * dim, True),
        ("simhash", 4 * dim, False), ("simhash", 4 * dim, True),
    ]
    for method, n_bits, rescore in configs:
        index = BinaryIndex(dim, n_bits=n_bits, method=method)
        index.add(docs)
        t0 = time.perf_counter()
        ids, _ = index.search(queries, k=k, rescore=rescore)
        ms = (time.perf_counter() - t0) / len(q) * 1000
        name = f"binary {method} {index.n_bits}b" + (" + rescore" if rescore else "")
        rows.append(evaluate(name, to_ids(ids), query_ids, relevant, exact_top10,
                             index.quantizer.n_bytes, ms))

    # Centering ablation on the default configuration
    index = BinaryIndex(dim, center=True)
    index.add(docs)
    t0 = time.perf_counter()
    ids, _ = index.search(queries, k=k)
    ms = (time.perf_counter() - t0) / len(q) * 1000
    rows.append(evaluate(f"binary sign {dim}b + rescore + center", to_ids(ids), query_ids,
                         relevant, exact_top10, index.quantizer.n_bytes, ms))

    # Lexical references
    try:
        from rank_bm25 import BM25Okapi

        tokenize = lambda s: s.lower().split()
        bm25 = BM25Okapi([tokenize(t) for t in doc_texts])
        t0 = time.perf_counter()
        bm = [top_k(bm25.get_scores(tokenize(t)), k) for t in query_texts]
        ms = (time.perf_counter() - t0) / len(q) * 1000
        rows.append(evaluate("BM25 (lexical, rank_bm25)", to_ids(bm), query_ids, relevant,
                             exact_top10, 0, ms))
    except ImportError:
        print("      (rank_bm25 not installed: skipping BM25)")

    if include_hdc:
        from neuroshift.hybrid.hdc_rag import HDCRetrievalEngine

        print("      encoding corpus with the old HDCRetrievalEngine ...")
        engine = HDCRetrievalEngine(dimensions=10000, device="cpu")
        engine.index_batch(doc_texts)
        t0 = time.perf_counter()
        hdc = [[r["doc_id"] for r in engine.search(t, top_k=k)] for t in query_texts]
        ms = (time.perf_counter() - t0) / len(q) * 1000
        rows.append(evaluate("old HDCRetrievalEngine (10k-dim)", to_ids(hdc), query_ids, relevant,
                             exact_top10, 10000 * 4, ms))

    print()
    print_rows(rows, rows[0]["ndcg"])
    print()
    print("  % of f32:      nDCG@10 relative to exact float32 search with the same model")
    print("  Top10 overlap: fraction of exact float32 top-10 also returned in the top-10")
    print("  ms/query:      single-threaded numpy, includes Python overhead per query")
    print()


# ---------------------------------------------------------------------------
# Part 2: scale
# ---------------------------------------------------------------------------
def bench_scale(n: int, dim: int = 384, n_queries: int = 20):
    print(f"[2/2] Scale: {n:,} documents, {dim}-dim embeddings")
    rng = np.random.default_rng(0)
    q_codes = rng.integers(0, 256, (n_queries, dim // 8), dtype=np.uint8)
    codes = rng.integers(0, 256, (n, dim // 8), dtype=np.uint8)

    t0 = time.perf_counter()
    for row in hamming_distances(q_codes, codes):
        top_k(row, 100, largest=False)
    binary_ms = (time.perf_counter() - t0) / n_queries * 1000

    float_bytes = n * dim * 4
    float_ms = float("nan")
    try:
        vectors = rng.standard_normal((n, dim), dtype=np.float32)
        q = rng.standard_normal((n_queries, dim), dtype=np.float32)
        t0 = time.perf_counter()
        for row in q:
            top_k(vectors @ row, 100)
        float_ms = (time.perf_counter() - t0) / n_queries * 1000
        del vectors
    except MemoryError:
        pass

    print(f"  {'':<26}{'Memory':>12}{'ms/query (brute force)':>26}")
    print(f"  {'float32':<26}{float_bytes / 2**20:>10.1f}MB{float_ms:>26.1f}")
    print(f"  {'binary sign (XOR+popcnt)':<26}{codes.nbytes / 2**20:>10.1f}MB{binary_ms:>26.1f}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name, e.g. scifact, nfcorpus")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--query-prefix", default="",
                        help="instruction prefix some models expect on queries (e.g. BGE, E5)")
    parser.add_argument("--scale-n", type=int, default=1_000_000)
    parser.add_argument("--skip-hdc", action="store_true", help="skip the slow old-HDC baseline")
    parser.add_argument("--skip-quality", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  NEUROSHIFT BINARY RETRIEVAL BENCHMARK")
    print("=" * 60)
    if not args.skip_quality:
        bench_quality(args.dataset, args.model, args.query_prefix, include_hdc=not args.skip_hdc)
    if args.scale_n > 0:
        bench_scale(args.scale_n)


if __name__ == "__main__":
    main()
