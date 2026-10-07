"""
Embed BEIR datasets with several models and cache the float32 vectors.

The deep comparison (benchmarks/deep_compare.py) reads these caches, so every
compression method and search system is evaluated on identical embeddings.

Run:  python -m benchmarks.embed_beir --models minilm bge-small --datasets scifact nfcorpus
"""

import argparse
import os
import time
from typing import Dict, List

import numpy as np


CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")

# name -> (hf id, query prefix, document template)
MODELS: Dict[str, tuple] = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "", "{title} {text}"),
    "bge-small": ("BAAI/bge-small-en-v1.5",
                  "Represent this sentence for searching relevant passages: ", "{title} {text}"),
    "embeddinggemma-2": ("google/embeddinggemma-2", "task: search result | query: ",
                         "title: {title_or_none} | text: {text}"),
    "mxbai-large": ("mixedbread-ai/mxbai-embed-large-v1",
                    "Represent this sentence for searching relevant passages: ", "{title} {text}"),
}


def load_beir(name: str):
    """Returns doc ids, doc titles, doc texts, query ids, query texts, qrels {qid: {docid: rel}}."""
    from datasets import load_dataset

    corpus = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    queries = load_dataset(f"BeIR/{name}", "queries", split="queries")
    qrels = load_dataset(f"BeIR/{name}-qrels", split="test")

    relevant: Dict[str, Dict[str, int]] = {}
    for row in qrels:
        if row["score"] > 0:
            relevant.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    query_text = {str(q): t for q, t in zip(queries["_id"], queries["text"])}
    query_ids = [q for q in relevant if q in query_text]
    return ([str(d) for d in corpus["_id"]], list(corpus["title"]), list(corpus["text"]),
            query_ids, [query_text[q] for q in query_ids], relevant)


def cache_path(dataset: str, kind: str, model: str) -> str:
    slug = MODELS[model][0].replace("/", "__")
    return os.path.join(CACHE_DIR, f"{dataset}-{kind}-{slug}.npy")


def doc_strings(model: str, titles: List[str], texts: List[str]) -> List[str]:
    template = MODELS[model][2]
    return [template.format(title=t or "", text=x, title_or_none=t or "none").strip()
            for t, x in zip(titles, texts)]


def embed(model_name: str, dataset: str, batch_size: int):
    from sentence_transformers import SentenceTransformer

    hf_id, query_prefix, _ = MODELS[model_name]
    paths = {k: cache_path(dataset, k, model_name) for k in ("docs", "queries")}
    if all(os.path.exists(p) for p in paths.values()):
        print(f"  {model_name} / {dataset}: cached")
        return
    _, titles, texts, _, query_texts, _ = load_beir(dataset)
    model = SentenceTransformer(hf_id, device="cpu")
    t0 = time.perf_counter()
    q = model.encode([query_prefix + t for t in query_texts], batch_size=batch_size,
                     normalize_embeddings=True, convert_to_numpy=True)
    d = model.encode(doc_strings(model_name, titles, texts), batch_size=batch_size,
                     normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(paths["queries"], q.astype(np.float32))
    np.save(paths["docs"], d.astype(np.float32))
    print(f"  {model_name} / {dataset}: {len(d)} docs, {len(q)} queries, dim {d.shape[1]}, "
          f"{time.perf_counter() - t0:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus", "arguana"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    for m in args.models:
        for ds in args.datasets:
            embed(m, ds, args.batch_size)


if __name__ == "__main__":
    main()
