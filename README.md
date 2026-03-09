# NeuroShift

**AI beyond neural networks.** Three computing paradigms that challenge the transformer-dominated status quo.

NeuroShift implements three underexplored AI paradigms and combines them into hybrid applications that work where traditional deep learning can't: few-shot scenarios, edge devices, air-gapped systems, and self-adapting deployments.

## What's Inside

### Core Paradigms

| Paradigm | What It Does | Key Property |
|----------|-------------|-------------|
| **Hyperdimensional Computing (HDC)** | Encodes data into 10,000-dim bipolar vectors | Learns from 3-5 examples. No training loop. |
| **MorphicNet** | Neural network that rewrites its own topology | Grows neurons, prunes dead ones, adds layers autonomously |
| **Neural Swarm** | Population of tiny agents evolving collectively | No backpropagation. Intelligence emerges from evolution. |

### Hybrid Applications

| Application | Pipeline | Use Case |
|-------------|----------|----------|
| **Anomaly Detector** | HDC + MorphicNet + Swarm | Network intrusion, IoT sensors, fraud detection |
| **HDC-RAG Engine** | Hyperdimensional retrieval | Fast document search with compositional queries |

## Installation

```bash
pip install -e .
```

Requires PyTorch 2.0+ with CUDA (optional but recommended).

## Quick Start

### Few-Shot Classification (HDC)

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

### Algebraic Reasoning (HDC)

```python
from neuroshift.hdc import HyperdimensionalEngine, AnalogyEngine

engine = HyperdimensionalEngine(dimensions=10000)
analogy = AnalogyEngine(engine)

# Define concepts with structured properties
for prop in ["gender", "role"]:
    analogy.define(prop)
for val in ["male", "female", "royalty", "common"]:
    analogy.define(val)

analogy.define_composite("king",  {"gender": "male",   "role": "royalty"})
analogy.define_composite("queen", {"gender": "female", "role": "royalty"})
analogy.define_composite("man",   {"gender": "male",   "role": "common"})
analogy.define_composite("woman", {"gender": "female", "role": "common"})

# Solve: king:queen :: man:?
results = analogy.solve_analogy("king", "queen", "man")
print(results[0])  # ('woman', 0.12)
```

### Self-Evolving Network (MorphicNet)

```python
from neuroshift.morphic import MorphicNet
import torch

model = MorphicNet(input_dim=10, output_dim=3, initial_hidden=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Train - architecture evolves automatically
for epoch in range(500):
    loss, evolved = model.train_step(X_train, Y_train, optimizer, epoch)
    if evolved:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        print(f"Architecture changed: {model.get_architecture_str()}")
```

### Anomaly Detection (Hybrid)

```python
from neuroshift.hybrid import AnomalyDetector
import torch

detector = AnomalyDetector(feature_dim=10)

# Learn normal patterns (instant, no training)
detector.learn_normal(normal_sensor_data)  # just 10-20 examples

# Detect anomalies
result = detector.detect(new_reading)
if result['is_anomaly']:
    print(f"ALERT: anomaly score {result['anomaly_score']:.3f}")
```

## Benchmark Results

Tested on NVIDIA RTX 5070, PyTorch 2.10, CUDA 12.8.

### HDC Classification

| Training Examples | Classes | Accuracy | Training Time | Memory |
|-------------------|---------|----------|---------------|--------|
| 3 per class | 5 | 93.3% | 140 ms | 200 KB |

### HDC Analogy Reasoning

| Test | Result |
|------|--------|
| king:queen :: man:? | woman (correct) |
| whale:mouse :: elephant:? | goldfish (correct) |
| july:january :: beach:? | fireplace (correct) |
| Role query: gender of king? | male (sim: 0.51) |

### MorphicNet Self-Evolution

| Metric | Value |
|--------|-------|
| Starting architecture | 2 -> 8 -> 4 (60 params) |
| Final architecture | 2 -> 219 -> 256 -> 256 -> 256 -> 4 (189K params) |
| Final accuracy | 96.5% (4-class spiral) |
| Architecture changes | 83 autonomous decisions |
| Training time | 8.8s |

### Neural Swarm Optimization

| Metric | Value |
|--------|-------|
| Problem | 5D Rastrigin (~100K local minima) |
| Best value found | 0.084 (optimal: 0.0) |
| Population | 80 agents, ~805 params each |
| Generations | 300 |

### Hybrid Anomaly Detector

| Phase | F1 Score | Training Data |
|-------|----------|--------------|
| HDC only (instant) | 89.3% | 10 examples, no backprop |
| + MorphicNet | 77.9% | 100 labeled examples |
| + Swarm optimization | 95.2% | evolved thresholds |

### HDC-RAG Retrieval

| Metric | Value |
|--------|-------|
| Throughput | 1,700+ queries/sec |
| Latency | 0.55 ms/query |
| Index time | 32 ms for 20 docs |
| Memory | 0.76 MB (0.095 MB with binary quantization) |

## Where This Actually Works (Honest Assessment)

NeuroShift is **not** a replacement for transformers, FAISS, or scikit-learn in their core domains.

It fills specific gaps where traditional approaches can't:

| Scenario | Why NeuroShift | Why Not Standard ML |
|----------|---------------|-------------------|
| **5 training examples, deploy now** | HDC learns instantly | Deep learning needs thousands |
| **Raspberry Pi / microcontroller** | < 1MB RAM, no GPU needed | Can't run transformers |
| **Air-gapped / privacy-critical** | No pre-trained model, no data leakage | Embeddings leak training data |
| **Self-adapting production system** | MorphicNet auto-resizes | Manual architecture tuning |
| **Non-differentiable optimization** | Swarm works without gradients | Backprop requires differentiable loss |

## Project Structure

```
neuroshift/
  hdc/engine.py              HDC core: encoding, classification, analogy
  morphic/network.py         Self-evolving neural architecture
  swarm/ecosystem.py         Neuroevolution + collective intelligence
  hybrid/
    anomaly_detector.py      Three-paradigm anomaly detection
    hdc_rag.py               Hyperdimensional retrieval engine

tests/                       pytest test suite
benchmarks/                  Performance benchmarks
examples/                    Usage examples and demos
```

## Prior Art & References

These paradigms aren't invented from scratch. NeuroShift builds on established research and combines them in novel ways:

- **Hyperdimensional Computing**: Kanerva, P. (2009). "Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors." Cognitive Computation.
- **Neural Architecture Search**: Elsken, T., Metzen, J. H., & Hutter, F. (2019). "Neural architecture search: A survey." JMLR.
- **Neuroevolution**: Stanley, K. O., & Miikkulainen, R. (2002). "Evolving neural networks through augmenting topologies." Evolutionary Computation.
- **Net2Net**: Chen, T., Goodfellow, I., & Shlens, J. (2016). "Net2Net: Accelerating learning via knowledge transfer." ICLR.

**What's novel here**: No prior work combines HDC + self-modifying architectures + neuroevolution into a unified pipeline. The hybrid anomaly detector and HDC-RAG engine are new applications of these combined paradigms.

## License

MIT
