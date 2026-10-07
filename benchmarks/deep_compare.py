"""
NeuroShift -- Deep Comparison
==============================
Compares neuroshift.retrieval against standard embedding-compression methods
and vector-search libraries, on identical cached embeddings
(see benchmarks/embed_beir.py).

  quality: nDCG@10 / Recall@100 on BEIR datasets for every (model, dataset),
           with per-query scores kept for paired bootstrap confidence intervals.
  scale:   latency, memory and build time over 1M vectors.

Run:
  python -m benchmarks.deep_compare quality --models minilm bge-small --datasets scifact nfcorpus
  python -m benchmarks.deep_compare scale --n 1000000 --dim 384
Results are written to benchmarks/results/*.json and printed as tables.
"""

import argparse
import json
import math
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from benchmarks.embed_beir import MODELS, cache_path, load_beir
from neuroshift.retrieval import BinaryIndex
from neuroshift.retrieval.hamming import top_k


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
K = 100


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def ndcg_at_10(ranked: List[str], rels: Dict[str, int]) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rels.values(), reverse=True)[:10]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at_100(ranked: List[str], rels: Dict[str, int]) -> float:
    return len(set(ranked[:100]) & set(rels)) / len(rels)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    """Mean of (a - b) and its 95% bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    diff = a - b
    idx = rng.integers(0, len(diff), (n, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def renorm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


# ---------------------------------------------------------------------------
# Methods: each returns (indices (m, k), bytes_per_doc, build_s, query_ms_per_query)
# ---------------------------------------------------------------------------
def faiss_method(make: Callable, train: bool = False, params=None):
    def run(D, Q, k):
        import faiss

        faiss.omp_set_num_threads(1)
        t0 = time.perf_counter()
        index = make(D.shape[1])
        if train:
            index.train(D)
        index.add(D)
        build = time.perf_counter() - t0
        t0 = time.perf_counter()
        _, I = index.search(Q, k, params=params) if params is not None else index.search(Q, k)
        ms = (time.perf_counter() - t0) / len(Q) * 1000
        return I, index.sa_code_size() if hasattr(index, "sa_code_size") else None, build, ms
    return run


def faiss_binary_sign(D, Q, k):
    import faiss

    faiss.omp_set_num_threads(1)
    t0 = time.perf_counter()
    index = faiss.IndexBinaryFlat(D.shape[1])
    index.add(np.packbits(D > 0, axis=1))
    build = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, I = index.search(np.packbits(Q > 0, axis=1), k)
    return I, D.shape[1] // 8, build, (time.perf_counter() - t0) / len(Q) * 1000


def ours(method="sign", n_bits_mult=None, rescore=True, trunc=None, center=False):
    def run(D, Q, k):
        if trunc:
            D, Q = renorm(D[:, :trunc]), renorm(Q[:, :trunc])
        n_bits = D.shape[1] * n_bits_mult if n_bits_mult else None
        t0 = time.perf_counter()
        index = BinaryIndex(D.shape[1], n_bits=n_bits, method=method, center=center)
        index.add(D)
        build = time.perf_counter() - t0
        t0 = time.perf_counter()
        I, _ = index.search(Q, k=k, rescore=rescore)
        return I, index.quantizer.n_bytes, build, (time.perf_counter() - t0) / len(Q) * 1000
    return run


def truncated_float(trunc):
    def run(D, Q, k):
        import faiss

        D, Q = renorm(D[:, :trunc]), renorm(Q[:, :trunc])
        return faiss_method(lambda d: faiss.IndexFlatIP(d))(D, Q, k)
    return run


def quality_methods(dim: int) -> Dict[str, Callable]:
    import faiss

    ip = faiss.METRIC_INNER_PRODUCT
    m = {
        "float32 (exact)": faiss_method(lambda d: faiss.IndexFlatIP(d)),
        "int8 scalar (SQ8)": faiss_method(lambda d: faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit, ip), train=True),
        "float32 MRL-256": truncated_float(256),
        "float32 MRL-128": truncated_float(128),
        f"PQ {dim // 8}B (FAISS, trained in-domain)": faiss_method(lambda d: faiss.IndexPQ(d, d // 8, 8, ip), train=True),
        f"PQ {dim // 2}B (FAISS, trained in-domain)": faiss_method(lambda d: faiss.IndexPQ(d, d // 2, 8, ip), train=True),
        "RaBitQ 1-bit (FAISS)": faiss_method(lambda d: faiss.IndexRaBitQ(d, ip), train=True),
        "binary sign, Hamming only (FAISS)": faiss_binary_sign,
        "ours: sign": ours(rescore=False),
        "ours: sign + rescore": ours(),
        "ours: sign + rescore + center": ours(center=True),
        "ours: simhash 4x bits + rescore": ours(method="simhash", n_bits_mult=4),
        "ours: MRL-256 + sign + rescore": ours(trunc=256),
        "ours: MRL-128 + sign + rescore": ours(trunc=128),
    }
    return m


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
def rank_ids(I: np.ndarray, doc_ids: List[str], query_ids: List[str]) -> List[List[str]]:
    """Map indices to doc ids, dropping a doc whose id equals the query id (BEIR convention)."""
    out = []
    for row, qid in zip(I, query_ids):
        out.append([doc_ids[i] for i in row if i >= 0 and doc_ids[i] != qid][:K])
    return out


def bm25_rankings(doc_titles, doc_texts, query_texts, k):
    from rank_bm25 import BM25Okapi

    tok = lambda s: s.lower().split()
    bm25 = BM25Okapi([tok(f"{t} {x}") for t, x in zip(doc_titles, doc_texts)])
    t0 = time.perf_counter()
    I = np.array([top_k(bm25.get_scores(tok(q)), k) for q in query_texts])
    return I, (time.perf_counter() - t0) / len(query_texts) * 1000


def run_quality(models: List[str], datasets: List[str]):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for ds in datasets:
        doc_ids, titles, texts, query_ids, query_texts, rel = load_beir(ds)
        out_path = os.path.join(RESULTS_DIR, f"quality-{ds}.json")
        results = json.load(open(out_path)) if os.path.exists(out_path) else {}

        if "bm25" not in results:
            I, ms = bm25_rankings(titles, texts, query_texts, K + 1)
            ranked = rank_ids(I, doc_ids, query_ids)
            results["bm25"] = {"BM25 (lexical)": {
                "ndcg": [ndcg_at_10(r, rel[q]) for r, q in zip(ranked, query_ids)],
                "recall": [recall_at_100(r, rel[q]) for r, q in zip(ranked, query_ids)],
                "bytes": None, "build_s": None, "ms": ms}}

        for model in models:
            if not all(os.path.exists(cache_path(ds, kind, model)) for kind in ("docs", "queries")):
                print(f"  skipping {model}/{ds}: no cached embeddings")
                continue
            D = renorm(np.load(cache_path(ds, "docs", model)).astype(np.float32))
            Q = renorm(np.load(cache_path(ds, "queries", model)).astype(np.float32))
            res = results.setdefault(model, {})
            for name, fn in quality_methods(D.shape[1]).items():
                if name in res:
                    continue
                if "MRL-" in name and int(name.split("MRL-")[1].split()[0]) >= D.shape[1]:
                    continue
                I, nbytes, build, ms = fn(D, Q, K + 1)
                ranked = rank_ids(np.asarray(I), doc_ids, query_ids)
                res[name] = {
                    "ndcg": [ndcg_at_10(r, rel[q]) for r, q in zip(ranked, query_ids)],
                    "recall": [recall_at_100(r, rel[q]) for r, q in zip(ranked, query_ids)],
                    "bytes": nbytes, "build_s": build, "ms": ms}
                print(f"  {ds:9s} {model:17s} {name:40s} nDCG@10 {np.mean(res[name]['ndcg']):.4f}", flush=True)
            json.dump(results, open(out_path, "w"))
        print_quality(ds, results)


def print_quality(ds: str, results: Dict):
    print(f"\n=== {ds} ===")
    for model, methods in results.items():
        if model == "bm25" or "float32 (exact)" not in methods:
            continue
        base = np.array(methods["float32 (exact)"]["ndcg"])
        print(f"\n  {model}  (float32 nDCG@10 = {base.mean():.4f})")
        print(f"  {'method':42s}{'bytes':>7}{'nDCG@10':>9}{'% f32':>8}{'Δ vs f32 [95% CI]':>26}{'R@100':>8}{'ms/q':>8}")
        rows = list(methods.items()) + list(results.get("bm25", {}).items())
        for name, r in rows:
            nd = np.array(r["ndcg"])
            d, lo, hi = paired_bootstrap(nd, base)
            b = "-" if r["bytes"] is None else str(r["bytes"])
            print(f"  {name:42s}{b:>7}{nd.mean():>9.4f}{100 * nd.mean() / base.mean():>7.1f}%"
                  f"{d:>+10.4f} [{lo:+.4f},{hi:+.4f}]{np.mean(r['recall']):>8.3f}{r['ms']:>8.2f}")


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------
def synthetic(n: int, dim: int, n_queries: int, seed: int = 0):
    """Clustered unit vectors (Gaussian mixture), closer to real embeddings than uniform noise."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((1000, dim)).astype(np.float32)
    def draw(m):
        out = np.empty((m, dim), dtype=np.float32)
        for s in range(0, m, 100_000):
            e = min(s + 100_000, m)
            out[s:e] = centers[rng.integers(0, 1000, e - s)] + 0.7 * rng.standard_normal((e - s, dim), dtype=np.float32)
        return renorm(out)
    return draw(n), draw(n_queries)


def run_scale(n: int, dim: int, n_queries: int, threads: int):
    import faiss
    from usearch.index import Index as UIndex

    print(f"=== scale: n={n:,} dim={dim} queries={n_queries} threads={threads} (clustered synthetic) ===")
    D, Q = synthetic(n, dim, n_queries)
    faiss.omp_set_num_threads(threads)
    exact = faiss.IndexFlatIP(dim)
    exact.add(D)
    _, truth = exact.search(Q, 10)

    def recall10(I):
        return float(np.mean([len(set(a[:10]) & set(b)) / 10 for a, b in zip(I, truth)]))

    rows = []

    def record(name, build, I, ms, mem_bytes, note=""):
        rows.append(dict(name=name, build_s=build, ms=ms, mem_mb=mem_bytes / 2 ** 20, recall10=recall10(I), note=note))
        r = rows[-1]
        print(f"  {name:44s} build {build:7.1f}s  {ms:8.2f} ms/q  {r['mem_mb']:8.1f} MB  recall@10 {r['recall10']:.3f} {note}", flush=True)

    def time_search(fn):
        t0 = time.perf_counter()
        I = fn()
        return I, (time.perf_counter() - t0) / n_queries * 1000

    I, ms = time_search(lambda: exact.search(Q, 10)[1])
    record("FAISS float32 flat (exact)", 0.0, I, ms, D.nbytes)

    for name, make, train in [
        ("FAISS int8 SQ8 flat", lambda: faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT), True),
        ("FAISS RaBitQ 1-bit flat", lambda: faiss.IndexRaBitQ(dim, faiss.METRIC_INNER_PRODUCT), True),
        ("FAISS PQ d/8 bytes flat", lambda: faiss.IndexPQ(dim, dim // 8, 8, faiss.METRIC_INNER_PRODUCT), True),
    ]:
        t0 = time.perf_counter()
        idx = make()
        if train:
            idx.train(D[:100_000])
        idx.add(D)
        build = time.perf_counter() - t0
        I, ms = time_search(lambda: idx.search(Q, 10)[1])
        record(name, build, I, ms, idx.sa_code_size() * n)
        del idx

    t0 = time.perf_counter()
    hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 64
    hnsw.add(D)
    build = time.perf_counter() - t0
    for ef in (32, 128):
        hnsw.hnsw.efSearch = ef
        I, ms = time_search(lambda: hnsw.search(Q, 10)[1])
        record(f"FAISS HNSW float32 (M=32, ef={ef})", build, I, ms, D.nbytes + n * 32 * 2 * 4, "graph approx.")
    del hnsw

    codes = np.packbits(D > 0, axis=1)
    qcodes = np.packbits(Q > 0, axis=1)
    t0 = time.perf_counter()
    bflat = faiss.IndexBinaryFlat(dim)
    bflat.add(codes)
    build = time.perf_counter() - t0
    I, ms = time_search(lambda: bflat.search(qcodes, 10)[1])
    record("FAISS binary flat (Hamming only)", build, I, ms, codes.nbytes)
    del bflat

    t0 = time.perf_counter()
    bh = faiss.IndexBinaryHNSW(dim, 32)
    bh.add(codes)
    build = time.perf_counter() - t0
    bh.hnsw.efSearch = 128
    I, ms = time_search(lambda: bh.search(qcodes, 10)[1])
    record("FAISS binary HNSW (Hamming only, ef=128)", build, I, ms, codes.nbytes + n * 32 * 2 * 4, "graph approx.")
    del bh

    for kind in ("f32", "b1"):
        t0 = time.perf_counter()
        u = UIndex(ndim=dim, metric="cos" if kind == "f32" else "hamming", dtype=kind, connectivity=16)
        u.add(np.arange(n), D if kind == "f32" else codes, threads=threads)
        build = time.perf_counter() - t0
        t0 = time.perf_counter()
        m = u.search(Q if kind == "f32" else qcodes, 10, threads=threads)
        ms = (time.perf_counter() - t0) / n_queries * 1000
        record(f"usearch HNSW {kind}", build, np.asarray(m.keys), ms, u.memory_usage, "graph approx.")
        del u

    t0 = time.perf_counter()
    ours_idx = BinaryIndex(dim)
    ours_idx.add(D)
    build = time.perf_counter() - t0
    for rescore in (False, True):
        I, ms = time_search(lambda: ours_idx.search(Q, k=10, rescore=rescore)[0])
        record(f"ours: binary flat{' + rescore' if rescore else ''} (numpy, 1 thread)", build, I, ms, ours_idx.memory_bytes())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump(rows, open(os.path.join(RESULTS_DIR, f"scale-{n}-{dim}-t{threads}.json"), "w"), indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quality")
    q.add_argument("--models", nargs="+", default=list(MODELS))
    q.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus", "arguana"])
    s = sub.add_parser("scale")
    s.add_argument("--n", type=int, default=1_000_000)
    s.add_argument("--dim", type=int, default=384)
    s.add_argument("--queries", type=int, default=200)
    s.add_argument("--threads", type=int, default=1)
    args = p.parse_args()
    if args.cmd == "quality":
        run_quality(args.models, args.datasets)
    else:
        run_scale(args.n, args.dim, args.queries, args.threads)


if __name__ == "__main__":
    main()
