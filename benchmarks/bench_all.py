import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
NeuroShift -- Full Benchmark Suite
====================================
Measures encoding, classification, training, evolution, and retrieval
throughput across the five core components.

Run:  python -m benchmarks.bench_all   (from project root)
"""

import time
import math
import torch

from neuroshift.hdc.engine import HyperdimensionalEngine, OneShotClassifier, AnalogyEngine
from neuroshift.morphic.network import MorphicNet
from neuroshift.swarm.ecosystem import NeuralSwarm
from neuroshift.hybrid.hdc_rag import HDCRetrievalEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_results = []  # (section, name, value, unit)


def record(section, name, value, unit="ms"):
    _results.append((section, name, f"{value:.2f}", unit))


def print_table():
    """Print all recorded results as a clean ASCII table."""
    sec_w = max(len(r[0]) for r in _results) + 2
    name_w = max(len(r[1]) for r in _results) + 2
    val_w = max(len(r[2]) for r in _results) + 2
    unit_w = max(len(r[3]) for r in _results) + 2
    total_w = sec_w + name_w + val_w + unit_w + 5  # separators

    line = "+" + "-" * (sec_w) + "+" + "-" * (name_w) + "+" + "-" * (val_w) + "+" + "-" * (unit_w) + "+"
    print()
    print("=" * total_w)
    print("  NEUROSHIFT BENCHMARK RESULTS")
    print("=" * total_w)
    print(f"  Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
    print(f"  PyTorch: {torch.__version__}")
    print()
    print(line)
    print(f"|{'Section':^{sec_w}}|{'Benchmark':^{name_w}}|{'Value':^{val_w}}|{'Unit':^{unit_w}}|")
    print(line)

    prev_sec = None
    for sec, name, val, unit in _results:
        if prev_sec is not None and sec != prev_sec:
            print(line)
        print(f"| {sec:<{sec_w-1}}| {name:<{name_w-1}}| {val:>{val_w-1}}| {unit:<{unit_w-1}}|")
        prev_sec = sec
    print(line)
    print()


# ---------------------------------------------------------------------------
# 1. HDC Encoding Speed
# ---------------------------------------------------------------------------
def bench_hdc_encoding():
    print("[1/5] Benchmarking HDC encoding speed ...")
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning enables computers to learn from data",
        "Quantum computing will change cryptography forever",
        "The stock market reacted to the new policy announcement",
        "Deep neural networks require large datasets for training",
    ]
    n_texts = len(sample_texts)
    n_repeats = 50  # encode each text this many times

    for dim in (1000, 5000, 10000):
        engine = HyperdimensionalEngine(dimensions=dim, device=DEVICE)

        # -- text encoding --
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            for txt in sample_texts:
                engine.encode_text(txt)
        elapsed = time.perf_counter() - t0
        total_ops = n_texts * n_repeats
        record("HDC Encode", f"text  d={dim}", elapsed / total_ops * 1000, "ms/op")

        # -- numeric encoding --
        values = [i / 99.0 for i in range(100)]
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            for v in values:
                engine.encode_numeric(v)
        elapsed = time.perf_counter() - t0
        total_ops = len(values) * n_repeats
        record("HDC Encode", f"num   d={dim}", elapsed / total_ops * 1e6, "us/op")


# ---------------------------------------------------------------------------
# 2. HDC Classification Speed
# ---------------------------------------------------------------------------
def bench_hdc_classification():
    print("[2/5] Benchmarking HDC classification speed ...")
    dim = 10000
    engine = HyperdimensionalEngine(dimensions=dim, device=DEVICE)

    for n_classes in (5, 20, 100):
        clf = OneShotClassifier(engine)

        # Train: 3 examples per class
        train_texts = []
        for c in range(n_classes):
            for ex in range(3):
                txt = f"class {c} example {ex} with some words and content here number {c * 100 + ex}"
                hv = engine.encode_text(txt)
                clf.learn(f"c{c}", hv)
                train_texts.append(txt)

        # Predict
        n_pred = 200
        test_hvs = [engine.encode_text(f"test query for class {i % n_classes} with extra context") for i in range(n_pred)]

        t0 = time.perf_counter()
        for hv in test_hvs:
            clf.predict(hv)
        elapsed = time.perf_counter() - t0

        record("HDC Classify", f"{n_classes} classes, {n_pred} preds", elapsed * 1000, "ms")
        record("HDC Classify", f"{n_classes} cls latency", elapsed / n_pred * 1e6, "us/pred")


# ---------------------------------------------------------------------------
# 3. MorphicNet Training Speed
# ---------------------------------------------------------------------------
def bench_morphicnet():
    print("[3/5] Benchmarking MorphicNet training speed ...")
    torch.manual_seed(42)
    device = torch.device(DEVICE)

    input_dim = 8
    output_dim = 4
    n_samples = 512
    X = torch.randn(n_samples, input_dim, device=device)
    Y = torch.randint(0, output_dim, (n_samples,), device=device)

    for init_hidden in (8, 32):
        model = MorphicNet(input_dim=input_dim, output_dim=output_dim,
                           initial_hidden=init_hidden, device=DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        n_epochs = 200
        t0 = time.perf_counter()
        evolutions = 0
        for epoch in range(n_epochs):
            loss_val, evolved = model.train_step(X, Y, optimizer, epoch)
            if evolved:
                evolutions += 1
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        elapsed = time.perf_counter() - t0

        record("MorphicNet", f"h={init_hidden}, {n_epochs} steps", elapsed * 1000, "ms")
        record("MorphicNet", f"h={init_hidden}, per-step", elapsed / n_epochs * 1000, "ms/step")
        record("MorphicNet", f"h={init_hidden}, evolutions", evolutions, "count")


# ---------------------------------------------------------------------------
# 4. Neural Swarm Evolution Speed
# ---------------------------------------------------------------------------
def bench_swarm():
    print("[4/5] Benchmarking Neural Swarm evolution speed ...")
    torch.manual_seed(42)

    n_dims = 5

    def rastrigin_fitness(agent, context):
        obs = torch.zeros(agent.input_dim, device=agent.device)
        solution = agent.act(obs)
        A = 10
        n = solution.shape[0]
        val = A * n + (solution ** 2 - A * torch.cos(2 * math.pi * solution)).sum()
        return -val.item()

    for pop_size in (20, 50, 100):
        swarm = NeuralSwarm(
            population_size=pop_size,
            input_dim=n_dims,
            output_dim=n_dims,
            fitness_fn=rastrigin_fitness,
            hidden=32,
            device=DEVICE,
        )

        n_gens = 50
        t0 = time.perf_counter()
        for _ in range(n_gens):
            swarm.evolve_generation()
        elapsed = time.perf_counter() - t0

        record("Swarm", f"pop={pop_size}, {n_gens} gens", elapsed * 1000, "ms")
        record("Swarm", f"pop={pop_size}, per-gen", elapsed / n_gens * 1000, "ms/gen")
        record("Swarm", f"pop={pop_size}, best fit", swarm.best_ever_fitness, "fitness")


# ---------------------------------------------------------------------------
# 5. HDC-RAG Indexing & Search Throughput
# ---------------------------------------------------------------------------
def bench_hdc_rag():
    print("[5/5] Benchmarking HDC-RAG indexing & search ...")

    base_docs = [
        "Python is a high-level programming language known for readability",
        "Django is a Python web framework for rapid development",
        "Flask is a lightweight Python micro web framework",
        "JavaScript powers interactive web pages in the browser",
        "React is a JavaScript library for user interfaces",
        "Node.js runs JavaScript on the server side",
        "PostgreSQL is a powerful relational database system",
        "MongoDB is a document-oriented NoSQL database",
        "Redis is an in-memory data store for caching",
        "Docker containerizes applications for deployment",
    ]

    for n_docs in (10, 100, 500):
        rag = HDCRetrievalEngine(dimensions=10000, device=DEVICE)

        # Generate documents by cycling the base set
        docs = [base_docs[i % len(base_docs)] + f" variant {i}" for i in range(n_docs)]

        # Index
        t0 = time.perf_counter()
        for doc in docs:
            rag.index(doc)
        index_time = time.perf_counter() - t0
        record("HDC-RAG", f"index {n_docs} docs", index_time * 1000, "ms")

        # Search throughput
        n_queries = 200
        queries = [
            "How to build a web application",
            "Best database for my project",
            "Machine learning frameworks",
            "Server side JavaScript",
            "Container orchestration tools",
        ]

        t0 = time.perf_counter()
        for i in range(n_queries):
            rag.search(queries[i % len(queries)], top_k=5)
        search_time = time.perf_counter() - t0

        qps = n_queries / search_time
        record("HDC-RAG", f"search {n_docs} docs", search_time / n_queries * 1000, "ms/query")
        record("HDC-RAG", f"search {n_docs} docs QPS", qps, "q/s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print()
    print("=" * 60)
    print("  NEUROSHIFT BENCHMARK SUITE")
    print("=" * 60)
    dev = DEVICE
    if dev == "cuda":
        dev += f" ({torch.cuda.get_device_name(0)})"
    print(f"  Device:  {dev}")
    print(f"  PyTorch: {torch.__version__}")
    print()

    bench_hdc_encoding()
    bench_hdc_classification()
    bench_morphicnet()
    bench_swarm()
    bench_hdc_rag()

    print_table()


if __name__ == "__main__":
    main()
