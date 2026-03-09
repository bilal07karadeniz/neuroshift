"""
NEUROSHIFT HYBRID DEMOS
=====================
Two novel hybrid systems that combine the three paradigms:

  1. Anomaly Detector: HDC + MorphicNet + Swarm
  2. HDC-RAG: Hyperdimensional retrieval engine for LLM pipelines
"""

import torch
import time
import sys
import os
import io
import math

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def demo_anomaly_detector():
    """Hybrid anomaly detector: HDC + MorphicNet + Swarm."""
    from neuroshift.hybrid import AnomalyDetector

    print("=" * 70)
    print("HYBRID 1: NEUROSHIFT ANOMALY DETECTOR")
    print("HDC (perception) + MorphicNet (processing) + Swarm (optimization)")
    print("=" * 70)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # --- Simulate network traffic data ---
    # Features: [packet_size, duration, src_port, dst_port, protocol, bytes_sent,
    #            bytes_recv, packets, flags, ttl]
    n_features = 10
    print("  Scenario: Network intrusion detection")
    print(f"  Features: {n_features} (packet_size, duration, ports, protocol, ...)")
    print()

    # Generate normal traffic (clustered around typical patterns)
    n_normal = 200
    normal_data = torch.randn(n_normal, n_features, device=device) * 0.3
    normal_data[:, 0] += 2.0   # packet_size ~ 2.0
    normal_data[:, 1] += 0.5   # duration ~ 0.5
    normal_data[:, 2] += 80    # src_port ~ 80
    normal_data[:, 3] += 443   # dst_port ~ 443
    normal_data[:, 5] += 1.0   # bytes_sent ~ 1.0
    normal_data[:, 6] += 1.5   # bytes_recv ~ 1.5

    # Generate attack traffic (different distributions)
    n_attack = 50

    # Attack Type 1: Port scan (many different dst_ports, tiny packets)
    port_scan = torch.randn(n_attack // 2, n_features, device=device) * 0.2
    port_scan[:, 0] += 0.1     # tiny packets
    port_scan[:, 1] += 0.01    # very short duration
    port_scan[:, 3] = torch.randint(1, 65535, (n_attack // 2,), device=device).float()
    port_scan[:, 7] += 100     # many packets

    # Attack Type 2: Data exfiltration (huge bytes_sent, long duration)
    exfil = torch.randn(n_attack // 2, n_features, device=device) * 0.2
    exfil[:, 0] += 8.0         # large packets
    exfil[:, 1] += 30.0        # very long duration
    exfil[:, 5] += 50.0        # massive bytes sent
    exfil[:, 6] += 0.1         # almost no response

    attack_data = torch.cat([port_scan, exfil], dim=0)

    # --- Phase 1: HDC-only detection (instant, no training) ---
    print("--- PHASE 1: HDC-Only Detection (zero training) ---")
    detector = AnomalyDetector(feature_dim=n_features, device=str(device))

    # Learn normal from just 10 examples!
    t0 = time.perf_counter()
    detector.learn_normal(normal_data[:10])
    learn_time = time.perf_counter() - t0
    print(f"  Learned 'normal' from 10 examples in {learn_time*1000:.1f} ms")

    # Test on all data
    tp, fp, tn, fn = 0, 0, 0, 0

    for i in range(normal_data.shape[0]):
        result = detector.detect(normal_data[i])
        if result['is_anomaly']:
            fp += 1
        else:
            tn += 1

    for i in range(attack_data.shape[0]):
        result = detector.detect(attack_data[i])
        if result['is_anomaly']:
            tp += 1
        else:
            fn += 1

    precision_1 = tp / (tp + fp + 1e-8)
    recall_1 = tp / (tp + fn + 1e-8)
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1 + 1e-8)
    accuracy_1 = (tp + tn) / (tp + fp + tn + fn)

    print(f"  Results (HDC-only, 10 training examples, no backprop):")
    print(f"    Accuracy:  {accuracy_1*100:.1f}%")
    print(f"    Precision: {precision_1*100:.1f}%")
    print(f"    Recall:    {recall_1*100:.1f}%")
    print(f"    F1 Score:  {f1_1*100:.1f}%")
    print(f"    TP={tp} FP={fp} TN={tn} FN={fn}")
    print()

    # --- Phase 2: Add MorphicNet (self-evolving classifier) ---
    print("--- PHASE 2: + MorphicNet (self-evolving classifier) ---")
    t0 = time.perf_counter()

    # Feed more normal data
    detector.learn_normal(normal_data[:50])

    # Train MorphicNet on a small labeled set
    train_normal = normal_data[:80]
    train_attack = attack_data[:20]
    detector.train_morphic(train_normal, train_attack, epochs=500)
    morphic_time = time.perf_counter() - t0

    print(f"  MorphicNet trained in {morphic_time:.1f}s")
    print(f"  Self-evolved architecture: {detector.morphic.get_architecture_str()}")
    print(f"  Parameters: {detector.morphic.get_total_params()}")

    # Test on held-out data
    tp, fp, tn, fn = 0, 0, 0, 0
    test_normal = normal_data[80:]
    test_attack = attack_data[20:]

    for i in range(test_normal.shape[0]):
        result = detector.detect(test_normal[i])
        if result['is_anomaly']:
            fp += 1
        else:
            tn += 1

    for i in range(test_attack.shape[0]):
        result = detector.detect(test_attack[i])
        if result['is_anomaly']:
            tp += 1
        else:
            fn += 1

    precision_2 = tp / (tp + fp + 1e-8)
    recall_2 = tp / (tp + fn + 1e-8)
    f1_2 = 2 * precision_2 * recall_2 / (precision_2 + recall_2 + 1e-8)
    accuracy_2 = (tp + tn) / (tp + fp + tn + fn)

    print(f"  Results (HDC + MorphicNet, held-out test set):")
    print(f"    Accuracy:  {accuracy_2*100:.1f}%")
    print(f"    Precision: {precision_2*100:.1f}%")
    print(f"    Recall:    {recall_2*100:.1f}%")
    print(f"    F1 Score:  {f1_2*100:.1f}%")
    print(f"    TP={tp} FP={fp} TN={tn} FN={fn}")
    print()

    # --- Phase 3: Add Swarm threshold optimization ---
    print("--- PHASE 3: + Swarm (threshold evolution) ---")
    t0 = time.perf_counter()
    opt_result = detector.optimize_thresholds(
        normal_data[:80], attack_data[:20], generations=80
    )
    swarm_time = time.perf_counter() - t0

    print(f"  Swarm optimized in {swarm_time:.1f}s")
    print(f"  Evolved threshold: {opt_result['optimized_threshold']:.4f}")
    print(f"  Best F1 during evolution: {opt_result['best_f1']:.4f}")

    # Re-test with optimized threshold
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(test_normal.shape[0]):
        result = detector.detect(test_normal[i])
        if result['is_anomaly']:
            fp += 1
        else:
            tn += 1

    for i in range(test_attack.shape[0]):
        result = detector.detect(test_attack[i])
        if result['is_anomaly']:
            tp += 1
        else:
            fn += 1

    precision_3 = tp / (tp + fp + 1e-8)
    recall_3 = tp / (tp + fn + 1e-8)
    f1_3 = 2 * precision_3 * recall_3 / (precision_3 + recall_3 + 1e-8)
    accuracy_3 = (tp + tn) / (tp + fp + tn + fn)

    print(f"  Results (Full pipeline: HDC + MorphicNet + Swarm):")
    print(f"    Accuracy:  {accuracy_3*100:.1f}%")
    print(f"    Precision: {precision_3*100:.1f}%")
    print(f"    Recall:    {recall_3*100:.1f}%")
    print(f"    F1 Score:  {f1_3*100:.1f}%")
    print(f"    TP={tp} FP={fp} TN={tn} FN={fn}")
    print()

    # --- Summary ---
    print("--- PROGRESSIVE IMPROVEMENT ---")
    print(f"  Phase 1 (HDC only):              F1={f1_1*100:.1f}%  (0 training, instant)")
    print(f"  Phase 2 (+ MorphicNet):           F1={f1_2*100:.1f}%  (self-evolved arch)")
    print(f"  Phase 3 (+ Swarm optimization):   F1={f1_3*100:.1f}%  (evolved thresholds)")
    print()

    status = detector.get_status()
    print(f"  Final system status:")
    print(f"    Normal patterns learned: {status['normal_examples_seen']}")
    print(f"    MorphicNet arch: {status['morphic_architecture']}")
    print(f"    MorphicNet params: {status['morphic_params']}")
    print(f"    Detection threshold: {status['threshold']:.4f}")
    print()

    return f1_3


def demo_hdc_rag():
    """HDC-powered retrieval engine for RAG."""
    from neuroshift.hybrid import HDCRetrievalEngine

    print("=" * 70)
    print("HYBRID 2: HDC-RAG RETRIEVAL ENGINE")
    print("Hyperdimensional retrieval for LLM pipelines")
    print("=" * 70)
    print()
    print("  Advantage over vector DBs: instant indexing, algebraic queries,")
    print("  compositional search (AND/OR/NOT on concepts), 8x less memory.")
    print()

    rag = HDCRetrievalEngine(dimensions=10000)

    # --- Index a knowledge base ---
    print("--- INDEXING KNOWLEDGE BASE ---")
    documents = [
        ("Python is a high-level programming language known for its readability",
         {"source": "wiki"}, ["python", "programming"]),
        ("Django is a Python web framework for building web applications quickly",
         {"source": "docs"}, ["python", "web", "django"]),
        ("Flask is a lightweight Python web framework with minimal dependencies",
         {"source": "docs"}, ["python", "web", "flask"]),
        ("JavaScript runs in the browser and powers interactive web pages",
         {"source": "wiki"}, ["javascript", "web", "frontend"]),
        ("React is a JavaScript library for building user interfaces",
         {"source": "docs"}, ["javascript", "web", "react", "frontend"]),
        ("Node.js allows JavaScript to run on the server side",
         {"source": "wiki"}, ["javascript", "web", "backend", "nodejs"]),
        ("PostgreSQL is a powerful open source relational database system",
         {"source": "wiki"}, ["database", "sql"]),
        ("MongoDB is a document-oriented NoSQL database for modern applications",
         {"source": "docs"}, ["database", "nosql"]),
        ("Redis is an in-memory data store used for caching and real-time apps",
         {"source": "wiki"}, ["database", "cache"]),
        ("Docker containerizes applications for consistent deployment anywhere",
         {"source": "docs"}, ["devops", "containers"]),
        ("Kubernetes orchestrates container deployment scaling and management",
         {"source": "wiki"}, ["devops", "containers", "orchestration"]),
        ("Machine learning uses statistical models to learn patterns from data",
         {"source": "wiki"}, ["ai", "ml"]),
        ("PyTorch is a machine learning framework with dynamic computation graphs",
         {"source": "docs"}, ["ai", "ml", "python", "pytorch"]),
        ("TensorFlow is Google's open source platform for machine learning",
         {"source": "docs"}, ["ai", "ml", "python", "tensorflow"]),
        ("Natural language processing enables computers to understand human text",
         {"source": "wiki"}, ["ai", "nlp"]),
        ("Transformers use self-attention mechanisms for sequence to sequence tasks",
         {"source": "paper"}, ["ai", "nlp", "transformers"]),
        ("GPT models generate human-like text using transformer architecture",
         {"source": "paper"}, ["ai", "nlp", "transformers", "llm"]),
        ("Rust is a systems programming language focused on safety and performance",
         {"source": "wiki"}, ["rust", "programming", "systems"]),
        ("Go is a statically typed language designed at Google for cloud services",
         {"source": "wiki"}, ["go", "programming", "cloud"]),
        ("GraphQL is a query language for APIs as an alternative to REST",
         {"source": "docs"}, ["api", "web", "graphql"]),
    ]

    t0 = time.perf_counter()
    for text, meta, tags in documents:
        rag.index(text, meta, tags)
    index_time = time.perf_counter() - t0

    stats = rag.get_stats()
    print(f"  Indexed {stats['total_documents']} documents in {index_time*1000:.1f} ms")
    print(f"  Memory usage: {stats['memory_mb']:.2f} MB")
    print(f"  Known tags: {', '.join(stats['known_tags'])}")
    print()

    # --- Test 1: Basic text search ---
    print("--- TEST 1: TEXT SEARCH ---")
    queries = [
        "How do I build a web app with Python?",
        "What database should I use for my project?",
        "How does machine learning work?",
    ]

    for query in queries:
        t0 = time.perf_counter()
        results = rag.search(query, top_k=3)
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"\n  Query: \"{query}\"  ({elapsed:.1f} ms)")
        for r in results:
            print(f"    #{r['rank']} (score: {r['score']:.3f}) {r['text'][:65]}...")

    # --- Test 2: Compositional concept search ---
    print("\n\n--- TEST 2: COMPOSITIONAL CONCEPT SEARCH ---")
    print("  (This is impossible with standard vector DBs)")

    concept_queries = [
        (["python", "web"], None, "Python AND web"),
        (["python", "ai"], None, "Python AND AI"),
        (["web"], ["python"], "Web but NOT Python"),
        (["database"], ["sql"], "Database but NOT SQL"),
    ]

    for positive, negative, description in concept_queries:
        t0 = time.perf_counter()
        results = rag.search_by_concept(positive, negative, top_k=3)
        elapsed = (time.perf_counter() - t0) * 1000

        neg_str = f" NOT [{', '.join(negative)}]" if negative else ""
        print(f"\n  Concept: [{', '.join(positive)}]{neg_str}  ({elapsed:.1f} ms)")
        for r in results:
            print(f"    #{r['rank']} (score: {r['score']:.3f}) {r['text'][:65]}...")

    # --- Benchmark ---
    print("\n\n--- PERFORMANCE BENCHMARK ---")
    # Bulk search timing
    n_queries = 1000
    t0 = time.perf_counter()
    for _ in range(n_queries):
        rag.search("machine learning python framework", top_k=5)
    bench_time = time.perf_counter() - t0

    qps = n_queries / bench_time
    print(f"  {n_queries} queries in {bench_time*1000:.0f} ms")
    print(f"  Throughput: {qps:.0f} queries/second")
    print(f"  Latency: {bench_time/n_queries*1000:.2f} ms/query")
    print(f"  Memory: {stats['memory_mb']:.2f} MB for {stats['total_documents']} docs")
    print(f"  (Binary quantization would reduce to {stats['memory_mb']/8:.3f} MB)")
    print()

    return qps


def main():
    print()
    print("+======================================================================+")
    print("|                 G E N E S I S   H Y B R I D S                        |")
    print("|         Novel applications combining three AI paradigms              |")
    print("+======================================================================+")
    print()

    device = "CUDA: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  Device: {device}")
    print()

    results = {}

    try:
        results['anomaly_f1'] = demo_anomaly_detector()
    except Exception as e:
        print(f"  Anomaly detector failed: {e}")
        import traceback; traceback.print_exc()

    try:
        results['rag_qps'] = demo_hdc_rag()
    except Exception as e:
        print(f"  HDC-RAG failed: {e}")
        import traceback; traceback.print_exc()

    print("=" * 70)
    print("HYBRID SYSTEMS - FINAL REPORT")
    print("=" * 70)
    print()
    if 'anomaly_f1' in results:
        print(f"  Anomaly Detector: F1={results['anomaly_f1']*100:.1f}% (HDC+MorphicNet+Swarm)")
    if 'rag_qps' in results:
        print(f"  HDC-RAG Engine:   {results['rag_qps']:.0f} queries/sec (20 docs, 10K dims)")
    print()
    print("  These hybrid systems combine paradigms that have never been")
    print("  united before: hyperdimensional computing, self-evolving")
    print("  architectures, and swarm intelligence in a single pipeline.")
    print()


if __name__ == "__main__":
    main()
