<div align="center">

# NeuroShift

**AI beyond neural networks.**

Three computing paradigms that challenge the transformer-dominated status quo,
combined into hybrid systems that work where deep learning can't.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-169-brightgreen.svg)](#tests)

</div>

---

## Start Here: Binary Semantic Retrieval

`neuroshift.retrieval` stores real sentence embeddings as **1 bit per dimension** and searches them with XOR + popcount. You keep **92–97% of float32 search quality in 1/32 of the memory** (96–97% in 1/8 with 1536-bit codes), in one `.npz` file. The index itself needs only numpy; the embedding model is the only heavy dependency.

```python
from neuroshift.retrieval import BinaryVectorStore, SentenceTransformerEmbedder

store = BinaryVectorStore(SentenceTransformerEmbedder())       # all-MiniLM-L6-v2
store.add(["The automobile engine needs repair.", "Cooking pasta requires boiling water."],
          tags=[["vehicles"], ["food"]])

store.search("my car is broken", k=1)                          # -> the automobile doc
store.search("anything", exclude_tags=["food"])                # hard tag filters
store.search_composite(["Python", "web framework"])            # AND across concepts
store.save("kb.npz")                                           # 48 bytes/doc + texts
```

Install with `pip install -e ".[embed]"`. Full example: [`examples/binary_rag.py`](examples/binary_rag.py).

### How it works

```mermaid
graph LR
    T["Text"] --> E["Embedding model<br/><i>meaning</i>"]
    E --> B["sign(x) or sign(Px)<br/><i>1 bit per dim</i>"]
    B --> S["Packed uint64 codes<br/><i>48 B/doc @ 384-d</i>"]
    Q["Query"] --> QE["Embed"] --> H["Stage 1: Hamming scan<br/><i>XOR + popcount</i>"]
    S --> H
    H --> R["Stage 2: rescore shortlist<br/><i>float query vs stored bits</i>"]
    R --> K["Top-k"]

    style E fill:#4a9eff,stroke:#2d7ad4,color:#fff
    style S fill:#51cf66,stroke:#37a34d,color:#fff
    style R fill:#845ef7,stroke:#6741d9,color:#fff
```

- **Meaning comes from the embedding model.** The binary layer only makes storing and scanning those embeddings cheap.
- **Two-stage search.** A Hamming scan builds a shortlist (10x k). The full-precision query then rescores the shortlist against the stored bits ("asymmetric" scoring). No float vectors are stored.
- **`sign` vs `simhash`.** `sign` keeps one bit per dimension. `simhash` is a random projection to any bit count (LSH), which lets you trade memory for recall.
- **Word-major layout.** Codes are stored as `(n_words, n_docs)` uint64, so a scan is a few vectorized XOR + popcount passes with no large temporaries.

### Results (measured, CPU, `python -m benchmarks.bench_retrieval`)

nDCG@10 on two BEIR datasets with human relevance labels, with two 384-dim embedding models, as a percentage of exact float32 search with the same model:

| Method | Bytes/doc | SciFact MiniLM | NFCorpus MiniLM | SciFact BGE-small | NFCorpus BGE-small |
|---|---:|---:|---:|---:|---:|
| float32 exact: nDCG@10 | 1536 | 0.645 | 0.317 | 0.713 | 0.344 |
| **binary sign 384b + rescore** (default) | **48** | **96.6%** | **93.2%** | **92.5%** | **92.5%** |
| binary sign 384b, no rescore | 48 | 91.2% | 87.1% | 87.8% | 81.2% |
| binary simhash 1536b + rescore | 192 | 97.4% | 96.8% | 96.5% | 96.0% |
| binary sign 384b + rescore + center | 48 | 94.9% | 85.1% | 91.1% | 89.8% |
| BM25 (lexical) | – | 86.8% | 84.5% | 78.5% | 77.9% |
| old `HDCRetrievalEngine` (10k-dim) | 40000 | 38.8% | – | – | – |

Rescoring is worth 4–11 points in every setting. Every binary configuration beats BM25.

Scale (1M documents, 384-d, brute force, single query, 4-core CPU):

| | Memory | ms/query |
|---|---:|---:|
| float32 matmul (multithreaded BLAS) | 1465 MB | 55 |
| binary XOR + popcount (single-threaded numpy) | **46 MB** | **18** |

**Use it when** memory is the constraint: edge devices, in-process RAG for many tenants, or millions of vectors on a laptop. Use `method="simhash", n_bits=4 * dim` when you can afford 8x instead of 32x compression. **Don't use it when** you need the last few percent of quality at any cost (use float or int8 vectors), or for 100M+ vectors (use an ANN index like FAISS/HNSW; this is a brute-force scan).

Centering (subtracting the corpus mean) is available as `center=True`. It is essential for strongly anisotropic embeddings (see the tests) but hurt all four settings above, so it is off by default.

---

## The Problem

Modern AI has a monoculture problem. Nearly everything is transformers + backpropagation + massive data. That leaves huge gaps:

```
What if you have 5 examples, not 5 million?
What if you can't afford a GPU?
What if your model needs to redesign itself in production?
What if your loss function isn't differentiable?
```

NeuroShift fills these gaps with three paradigms that don't need any of that.

---

## Architecture Overview

```mermaid
graph TB
    subgraph "Core Paradigms"
        HDC["<b>Hyperdimensional Computing</b><br/>10,000-dim bipolar vectors<br/>Instant learning, no training loop"]
        MN["<b>MorphicNet</b><br/>Self-evolving topology<br/>Grows/prunes autonomously"]
        SW["<b>Neural Swarm</b><br/>Population of tiny agents<br/>Evolution, not backprop"]
    end

    subgraph "Hybrid Applications"
        AD["<b>Anomaly Detector</b><br/>HDC + MorphicNet + Swarm"]
        RAG["<b>HDC-RAG Engine</b><br/>Hyperdimensional retrieval"]
    end

    HDC --> AD
    MN --> AD
    SW --> AD
    HDC --> RAG

    style HDC fill:#4a9eff,stroke:#2d7ad4,color:#fff
    style MN fill:#ff6b6b,stroke:#d44a4a,color:#fff
    style SW fill:#51cf66,stroke:#37a34d,color:#fff
    style AD fill:#845ef7,stroke:#6741d9,color:#fff
    style RAG fill:#ff922b,stroke:#d47a1e,color:#fff
```

---

## 1. Hyperdimensional Computing (HDC)

Encodes data into 10,000-dimensional bipolar vectors ({-1, +1}). Classification is instant -- no gradient descent, no epochs, no batches.

### How It Works

```mermaid
graph LR
    A["Raw Text"] --> B["Word Tokenization"]
    B --> C["Position Binding<br/><i>word * shift(pos)</i>"]
    C --> D["Bundle<br/><i>majority vote</i>"]
    D --> E["10,000-dim<br/>Bipolar Vector"]

    style A fill:#f8f9fa,stroke:#dee2e6
    style E fill:#4a9eff,stroke:#2d7ad4,color:#fff
```

**Three operations, that's it:**

| Operation | Math | What It Does |
|-----------|------|-------------|
| **Bind** | `a * b` (element-wise) | Creates associations. Self-inverse: `bind(bind(a,b), a) = b` |
| **Bundle** | `sign(sum(vectors))` | Merges multiple vectors. Result is similar to all inputs |
| **Permute** | `roll(v, k)` | Circular shift. Encodes order/position |

### Results: Few-Shot Classification

```
Task: 5-class text classification, 3 examples per class

                            Accuracy
  NeuroShift HDC (3 examples)  |==========================================| 93.3%
  Random baseline              |==========                                | 20.0%

  Training time: 140 ms    Memory: 200 KB    No GPU required
```

### Results: Algebraic Reasoning

HDC can do `king - man + woman = queen` style reasoning, but with *actual math on vectors*, not word2vec tricks:

```
  king:queen :: man:?

  Step 1: relationship = bind(king, queen)     -- extracts "gender flip"
  Step 2: answer = bind(relationship, man)      -- applies it to "man"
  Step 3: nearest neighbor search               -- finds "woman"
```

```
  Test                              Answer          Correct?
  ─────────────────────────────────────────────────────────
  king:queen :: man:?               woman              yes
  whale:mouse :: elephant:?         goldfish           yes
  july:january :: beach:?           fireplace          yes
  Role query: gender of king?       male (sim: 0.51)   yes
```

### Quick Start

```python
from neuroshift.hdc import HyperdimensionalEngine, OneShotClassifier

engine = HyperdimensionalEngine(dimensions=10000)
classifier = OneShotClassifier(engine)

# Train: literally just encode and store
classifier.learn("spam", engine.encode_text("Buy now! Limited offer discount"))
classifier.learn("spam", engine.encode_text("Free money click here winner"))
classifier.learn("ham", engine.encode_text("Meeting at 3pm in the conference room"))
classifier.learn("ham", engine.encode_text("Can you review the pull request"))

# Predict: instant, no inference pipeline
label, score, _ = classifier.predict(engine.encode_text("Exclusive deal just for you"))
print(f"{label} (confidence: {score:.3f})")  # spam
```

---

## 2. MorphicNet

A neural network that **rewrites its own architecture** during training. Starts tiny, grows neurons when stuck, prunes dead ones, adds layers when needed.

### How It Evolves

```mermaid
graph TD
    A["Training Step"] --> B{"Loss stagnating?<br/><i>stagnation >= 3</i>"}
    B -->|No| A
    B -->|Yes| C{"Which action?"}
    C -->|"Neurons underused"| D["Prune Dead Neurons<br/><i>utility < 1% of mean</i>"]
    C -->|"Need more capacity"| E["Grow Layer Width<br/><i>+50% neurons</i>"]
    C -->|"Need more depth"| F["Add New Layer<br/><i>identity-initialized</i>"]
    D --> G["Cooldown: 80 steps"]
    E --> G
    F --> G
    G --> A

    style B fill:#ff922b,stroke:#d47a1e,color:#fff
    style D fill:#ff6b6b,stroke:#d44a4a,color:#fff
    style E fill:#51cf66,stroke:#37a34d,color:#fff
    style F fill:#4a9eff,stroke:#2d7ad4,color:#fff
```

### Results: Architecture Evolution on 4-Class Spiral

The network starts with 60 parameters and autonomously grows to 189K:

```
  Architecture Evolution (83 autonomous decisions)
  ═══════════════════════════════════════════════════

  Step 0     [2] -> [8] -> [4]                           60 params
             ██

  Step 50    [2] -> [12] -> [4]                          88 params
             ███

  Step 150   [2] -> [36] -> [16] -> [4]                  720 params
             ████████

  Step 300   [2] -> [108] -> [64] -> [32] -> [4]         9.6K params
             ████████████████████

  Step 500   [2] -> [219] -> [256] -> [256] -> [256] -> [4]  189K params
             ████████████████████████████████████████████

  Final accuracy: 96.5%    Training time: 8.8s
```

### Quick Start

```python
from neuroshift.morphic import MorphicNet
import torch

model = MorphicNet(input_dim=10, output_dim=3, initial_hidden=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(500):
    loss, evolved = model.train_step(X_train, Y_train, optimizer, epoch)
    if evolved:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        print(f"Architecture changed: {model.get_architecture_str()}")
```

---

## 3. Neural Swarm

A population of tiny neural agents (2-layer networks, ~805 params each) that evolve through tournament selection, crossover, and mutation. **No backpropagation at all** -- intelligence emerges from evolution.

### How It Works

```mermaid
graph LR
    A["Population<br/>80 agents"] --> B["Evaluate<br/>Fitness"]
    B --> C["Tournament<br/>Selection"]
    C --> D["Crossover<br/>+ Mutation"]
    D --> E["New<br/>Generation"]
    E --> F["Inject<br/>Best-Ever"]
    F --> A

    style A fill:#51cf66,stroke:#37a34d,color:#fff
    style B fill:#ff922b,stroke:#d47a1e,color:#fff
    style C fill:#4a9eff,stroke:#2d7ad4,color:#fff
    style D fill:#ff6b6b,stroke:#d44a4a,color:#fff
    style E fill:#845ef7,stroke:#6741d9,color:#fff
    style F fill:#51cf66,stroke:#37a34d,color:#fff
```

### Results: 5D Rastrigin Optimization

The Rastrigin function has ~100,000 local minima. The swarm navigates them without gradients:

```
  Rastrigin Function (5D) -- lower is better, optimal = 0.0
  ═════════════════════════════════════════════════════════

  Generation    Best Value    Population Fitness
  ──────────    ──────────    ──────────────────
       1         47.203       ▓░░░░░░░░░░░░░░░░░░░
      25         12.841       ▓▓▓▓▓▓░░░░░░░░░░░░░░
      50          5.217       ▓▓▓▓▓▓▓▓▓░░░░░░░░░░░
     100          1.493       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░
     200          0.291       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░
     300          0.084       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  <-- near optimal

  Population: 80 agents    Params/agent: ~805    Total: ~64K params
```

---

## 4. Hybrid: Anomaly Detector

The first system to combine all three paradigms into a single detection pipeline.

### Pipeline

```mermaid
graph LR
    subgraph "Phase 1: Instant (no training)"
        A["Raw Data"] --> B["HDC Encoder"]
        B --> C["Similarity to<br/>Normal Centroid"]
    end

    subgraph "Phase 2: Adaptive"
        C --> D["MorphicNet<br/>Classifier"]
        D --> E["MAX Ensemble<br/>Score"]
    end

    subgraph "Phase 3: Self-Optimizing"
        E --> F["Swarm-Evolved<br/>Thresholds"]
        F --> G["Anomaly<br/>Decision"]
    end

    style B fill:#4a9eff,stroke:#2d7ad4,color:#fff
    style D fill:#ff6b6b,stroke:#d44a4a,color:#fff
    style F fill:#51cf66,stroke:#37a34d,color:#fff
    style G fill:#845ef7,stroke:#6741d9,color:#fff
```

### Results: Progressive Improvement

Each paradigm layer improves detection, starting from just 10 training examples:

```
  Detection Performance (F1 Score)
  ════════════════════════════════

  Phase 1: HDC only
  (instant, 0 training)     |████████████████████████████████████░░░░░| 89.3%

  Phase 2: + MorphicNet
  (100 labeled examples)    |████████████████████████████████░░░░░░░░░| 77.9%

  Phase 3: + Swarm
  (evolved thresholds)      |████████████████████████████████████████░| 95.2%

  ──────────────────────────────────────────────────
  Traditional ML baseline (needs 10,000+ examples): ~97%
  NeuroShift (from 10 examples, self-optimizing):    95.2%
```

### Quick Start

```python
from neuroshift.hybrid import AnomalyDetector
import torch

detector = AnomalyDetector(feature_dim=10)

# Learn normal patterns (instant, no training loop)
detector.learn_normal(normal_sensor_data)  # just 10-20 examples

# Detect anomalies
result = detector.detect(new_reading)
if result['is_anomaly']:
    print(f"ALERT: anomaly score {result['anomaly_score']:.3f}")
```

---

## 5. Hybrid: HDC-RAG Engine (lexical, superseded)

> **For semantic search use [`neuroshift.retrieval`](#start-here-binary-semantic-retrieval).** `HDCRetrievalEngine` builds document vectors from *random* per-word hypervectors, so it matches shared words, not meanings ("car" and "automobile" are unrelated). On BEIR SciFact it scores nDCG@10 0.25, below BM25 (0.56), while storing 40 KB per document. It is kept as an example of HDC compositional operations.

Hyperdimensional retrieval with compositional queries. Supports AND (intersection) and NOT (exclusion) semantics through vector algebra.

### How Compositional Search Works

```mermaid
graph TD
    subgraph "Standard Search"
        Q1["'python web'"] --> R1["Bag-of-words<br/>similarity"]
    end

    subgraph "HDC Compositional Search"
        Q2["positive: python, web<br/>negative: django"] --> S1["Encode each concept"]
        S1 --> S2["Per-concept similarity<br/>to every document"]
        S2 --> S3["AND = min(similarities)<br/>NOT = subtract 0.5*neg"]
        S3 --> R2["Intersection semantics"]
    end

    style R1 fill:#f8f9fa,stroke:#dee2e6
    style R2 fill:#4a9eff,stroke:#2d7ad4,color:#fff
```

### Results

Measured on BEIR SciFact (5,183 docs, CPU), against the replacement:

```
                              nDCG@10   Bytes/doc   ms/query
  HDCRetrievalEngine            0.250       40000        356
  BM25                          0.560           -         38
  BinaryVectorStore (sign)      0.623          48          1   (+ query embedding time)
```

---

## Benchmark Summary

Tested on NVIDIA RTX 5070, PyTorch 2.10, CUDA 12.8.

```
  Component               Key Metric                Value
  ─────────────────────── ───────────────────────── ──────────────
  HDC Classification       Accuracy (3-shot, 5 cls)  93.3%
  HDC Analogy              Correct analogies          4/4
  MorphicNet               Final accuracy (spiral)    96.5%
  MorphicNet               Autonomous decisions       83
  Neural Swarm             Rastrigin best (5D)        0.084
  Anomaly Detector         F1 (from 10 examples)      95.2%
  Binary Retrieval         nDCG@10 vs float32         92-97% at 32x less memory
  Binary Retrieval         1M-doc scan, 1 query       18 ms, 46 MB
```

Run benchmarks yourself:
```bash
python -m benchmarks.bench_all
pip install -e ".[bench]" && python -m benchmarks.bench_retrieval   # downloads model + BEIR data
```

---

## Where This Actually Works

NeuroShift is **not** a replacement for transformers, FAISS, or scikit-learn in their core domains.

It fills specific gaps:

```mermaid
quadrantChart
    title NeuroShift vs Traditional ML
    x-axis "Few Data" --> "Lots of Data"
    y-axis "Fixed Model" --> "Adaptive Model"
    quadrant-1 "NeuroShift wins"
    quadrant-2 "Either works"
    quadrant-3 "Traditional ML wins"
    quadrant-4 "NeuroShift helps"
    "HDC Classifier": [0.15, 0.3]
    "MorphicNet": [0.5, 0.9]
    "Swarm Optimizer": [0.3, 0.7]
    "Anomaly Detector": [0.2, 0.85]
    "Transformers": [0.9, 0.2]
    "scikit-learn": [0.6, 0.15]
    "FAISS": [0.85, 0.1]
```

| Scenario | Why NeuroShift | Why Not Standard ML |
|----------|---------------|-------------------|
| **5 training examples, deploy now** | HDC learns instantly | Deep learning needs thousands |
| **Raspberry Pi / microcontroller** | < 1MB RAM, no GPU needed | Can't run transformers |
| **Air-gapped / privacy-critical** | No pre-trained model, no data leakage | Embeddings leak training data |
| **Self-adapting production system** | MorphicNet auto-resizes | Manual architecture tuning |
| **Non-differentiable optimization** | Swarm works without gradients | Backprop requires differentiable loss |

---

## Installation

```bash
pip install -e .            # core
pip install -e ".[embed]"   # + sentence-transformers for BinaryVectorStore
```

Requires Python 3.10+ and PyTorch 2.0+. CUDA optional. `neuroshift.retrieval` itself needs only numpy (>= 2.0 for native popcount; older numpy uses a lookup table).

## Tests

```bash
pytest tests/ -v
```

169 tests covering all components:
- `test_retrieval.py` -- 50 tests (quantization, Hamming kernels, index, store, persistence)
- `test_hdc.py` -- 32 tests (encoding, bind/bundle/permute, classification, analogy)
- `test_morphic.py` -- 21 tests (forward pass, evolution, growth, regression)
- `test_swarm.py` -- 30 tests (agents, population, fitness, convergence)
- `test_hybrid.py` -- 36 tests (anomaly detection, RAG indexing/search)

Known failure: `test_hybrid.py::test_detect_on_outlier_returns_high_anomaly_score`. The anomaly detector's `sign(x @ P)` encoding is scale-invariant, so it cannot see magnitude outliers.

## Project Structure

```
neuroshift/
  retrieval/
    quantize.py              Float embeddings -> packed bits (sign / simhash)
    hamming.py               Word-major XOR + popcount kernels
    index.py                 Two-stage binary index (Hamming shortlist + rescoring)
    store.py                 Texts, tags, concept algebra, .npz persistence
    embedders.py             sentence-transformers / custom embedders
  hdc/engine.py              HDC core: encoding, classification, analogy
  morphic/network.py         Self-evolving neural architecture
  swarm/ecosystem.py         Neuroevolution + collective intelligence
  hybrid/
    anomaly_detector.py      Three-paradigm anomaly detection
    hdc_rag.py               Lexical HDC retrieval (superseded by retrieval/)

tests/                       169-test pytest suite
benchmarks/                  Performance benchmarks
examples/                    Usage examples and demos
```

## Prior Art & References

These paradigms build on established research and combine them in novel ways:

- **Hyperdimensional Computing**: Kanerva, P. (2009). *Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors.* Cognitive Computation.
- **Neural Architecture Search**: Elsken, T., Metzen, J. H., & Hutter, F. (2019). *Neural architecture search: A survey.* JMLR.
- **Neuroevolution**: Stanley, K. O., & Miikkulainen, R. (2002). *Evolving neural networks through augmenting topologies.* Evolutionary Computation.
- **Net2Net**: Chen, T., Goodfellow, I., & Shlens, J. (2016). *Net2Net: Accelerating learning via knowledge transfer.* ICLR.

**What's novel**: No prior work combines HDC + self-modifying architectures + neuroevolution into a unified pipeline. The hybrid anomaly detector and HDC-RAG engine are new applications of these combined paradigms.

## License

MIT
