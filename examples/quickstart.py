import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
NeuroShift Quickstart -- all five components in ~100 lines of logic.

Run:  python -m examples.quickstart   (from project root)
"""

import torch
import math

from neuroshift.hdc.engine import HyperdimensionalEngine, OneShotClassifier, AnalogyEngine
from neuroshift.morphic.network import MorphicNet
from neuroshift.swarm.ecosystem import NeuralSwarm
from neuroshift.hybrid.hdc_rag import HDCRetrievalEngine

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. HDC One-Shot Classification ──────────────────────────────────────────
print("=== 1. HDC Classification (3 classes, 2 examples each) ===")
engine = HyperdimensionalEngine(dimensions=10000, device=device)
clf = OneShotClassifier(engine)

training = {
    "sport":  ["The team scored a goal in the match", "The runner won the race"],
    "tech":   ["The new chip delivers faster computing", "The software update improved speed"],
    "food":   ["The chef cooked a delicious pasta", "She baked fresh bread with herbs"],
}
for label, texts in training.items():
    for t in texts:
        clf.learn(label, engine.encode_text(t))

tests = [
    ("The player hit a home run in the game", "sport"),
    ("The app launched with better performance", "tech"),
    ("The recipe uses garlic and olive oil", "food"),
]
for text, expected in tests:
    pred, conf, _ = clf.predict(engine.encode_text(text))
    ok = "OK" if pred == expected else "XX"
    print(f"  [{ok}] '{text[:50]}' -> {pred} (conf {conf:.3f})")

# ── 2. HDC Analogy ──────────────────────────────────────────────────────────
print("\n=== 2. HDC Analogy (king:queen :: man:?) ===")
ana = AnalogyEngine(engine)
for prop in ("gender", "role"):
    ana.define(prop)
for val in ("male", "female", "royalty", "common"):
    ana.define(val)
ana.define_composite("king",  {"gender": "male",   "role": "royalty"})
ana.define_composite("queen", {"gender": "female", "role": "royalty"})
ana.define_composite("man",   {"gender": "male",   "role": "common"})
ana.define_composite("woman", {"gender": "female", "role": "common"})

results = ana.solve_analogy("king", "queen", "man")
print(f"  king:queen :: man:? -> {results[0][0]} (sim {results[0][1]:.3f})")

# ── 3. MorphicNet on XOR-like problem ───────────────────────────────────────
print("\n=== 3. MorphicNet (XOR-like classification) ===")
torch.manual_seed(42)
dev = torch.device(device)
# Create a 2D dataset where class = quadrant (4 classes, non-linear boundaries)
N = 400
X = torch.randn(N, 2, device=dev)
Y = ((X[:, 0] > 0).long() * 2 + (X[:, 1] > 0).long())  # 4 quadrants

model = MorphicNet(input_dim=2, output_dim=4, initial_hidden=4, device=device)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(300):
    loss, evolved = model.train_step(X, Y, opt, epoch)
    if evolved:
        opt = torch.optim.Adam(model.parameters(), lr=0.01)

model.eval()
with torch.no_grad():
    acc = (model(X).argmax(1) == Y).float().mean().item() * 100
print(f"  Architecture: {model.get_architecture_str()}")
print(f"  Parameters:   {model.get_total_params()}")
print(f"  Accuracy:     {acc:.1f}%")

# ── 4. Neural Swarm optimization ────────────────────────────────────────────
print("\n=== 4. Swarm (minimize Rastrigin, 3D) ===")

def rastrigin_fitness(agent, ctx):
    obs = torch.zeros(agent.input_dim, device=agent.device)
    x = agent.act(obs)
    val = 10 * 3 + (x ** 2 - 10 * torch.cos(2 * math.pi * x)).sum()
    return -val.item()

swarm = NeuralSwarm(population_size=40, input_dim=3, output_dim=3,
                    fitness_fn=rastrigin_fitness, hidden=16, device=device)
result = swarm.run(generations=100, report_every=101)  # silent
best = result["best_agent"]
sol = best.act(torch.zeros(3, device=best.device)).cpu()
val = 10 * 3 + (sol ** 2 - 10 * torch.cos(2 * math.pi * sol)).sum().item()
print(f"  Best solution: [{', '.join(f'{v:.4f}' for v in sol)}]")
print(f"  Rastrigin val: {val:.4f}  (optimal: 0.0)")

# ── 5. HDC-RAG retrieval ────────────────────────────────────────────────────
print("\n=== 5. HDC-RAG (5 documents, semantic search) ===")
rag = HDCRetrievalEngine(dimensions=10000, device=device)
docs = [
    "Python is a high-level language known for readability",
    "Django is a Python web framework for rapid development",
    "React is a JavaScript library for building UIs",
    "PostgreSQL is a powerful relational database",
    "Docker containerizes apps for consistent deployment",
]
for d in docs:
    rag.index(d)

query = "How do I build a web app with Python?"
hits = rag.search(query, top_k=3)
print(f"  Query: '{query}'")
for h in hits:
    print(f"    #{h['rank']} (score {h['score']:.3f}) {h['text'][:60]}")

print("\nDone.")
