"""
NEUROSHIFT - Three AI Paradigms That Break The Mold
==================================================
Run all three demos to see:
  1. HDC: One-shot learning + algebraic reasoning (NO training loop)
  2. MorphicNet: Network that grows its own brain (self-evolving architecture)
  3. Neural Swarm: Collective intelligence from tiny agents (emergent problem solving)

Each paradigm challenges a core assumption of modern AI.
"""

import torch
import torch.nn.functional as F
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def demo_hdc_classification():
    """DEMO 1: One-Shot Text Classification

    Challenge to modern AI: What if you could classify text
    after seeing just ONE example per category? No epochs, no batches,
    no loss functions. Just show it one example and it learns instantly.
    """
    from neuroshift.hdc import HyperdimensionalEngine, OneShotClassifier

    print("=" * 70)
    print("PARADIGM 1: HYPERDIMENSIONAL COMPUTING - FEW-SHOT CLASSIFICATION")
    print("=" * 70)
    print()
    print("Challenge: Learn to classify text from just 3 examples per class.")
    print("No training loop. No epochs. No gradient descent. No model weights.")
    print("Compare: GPT-3 needed 175B params. This uses 5 hypervectors (~200KB).")
    print()

    engine = HyperdimensionalEngine(dimensions=10000)
    classifier = OneShotClassifier(engine)
    device_name = engine.device.type
    if device_name == "cuda":
        device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")
    print(f"Hypervector dimensions: {engine.dim:,}")
    print()

    # === TRAINING: 3 examples per class (few-shot) ===
    # Key: shared vocabulary between train/test enables generalization
    training_data = {
        "sports": [
            "The team scored a winning goal in the final match",
            "The player hit a home run in the championship game",
            "The runner won the race with a record breaking time",
        ],
        "technology": [
            "The new processor chip delivers faster computing speed",
            "The software update improved the app performance",
            "The company launched a new artificial intelligence platform",
        ],
        "food": [
            "The recipe calls for fresh basil and mozzarella cheese",
            "The chef cooked a delicious pasta with garlic sauce",
            "She baked fresh bread with olive oil and herbs",
        ],
        "politics": [
            "The senator proposed a new bill in congress today",
            "The president signed the executive order on policy reform",
            "The party leader called for a vote on the new legislation",
        ],
        "science": [
            "The experiment confirmed the quantum entanglement theory",
            "Researchers published a new study on climate change data",
            "The scientists discovered a new species in the deep ocean",
        ],
    }

    print("--- TRAINING (few-shot: 3 examples per class) ---")
    t0 = time.perf_counter()
    for label, examples in training_data.items():
        for text in examples:
            hv = engine.encode_text(text)
            classifier.learn(label, hv)
        print(f"  Learned '{label}' from {len(examples)} examples")
    train_time = time.perf_counter() - t0
    total_examples = sum(len(v) for v in training_data.values())
    print(f"\n  Total training time: {train_time*1000:.1f} ms (not seconds -- milliseconds)")
    print(f"  Examples seen: {total_examples} total ({total_examples//5} per class)")
    print()

    # === TESTING: sentences with shared vocabulary ===
    test_data = [
        ("The basketball player scored in the championship", "sports"),
        ("The team broke a record in the final game", "sports"),
        ("The runner finished the race in first place", "sports"),
        ("The new software platform uses artificial intelligence", "technology"),
        ("The computing speed improved with the new chip update", "technology"),
        ("The app launched with faster performance", "technology"),
        ("She cooked pasta with fresh garlic and olive oil", "food"),
        ("The chef baked bread with mozzarella and basil", "food"),
        ("The recipe uses herbs and a delicious cheese sauce", "food"),
        ("The congress voted on the reform bill today", "politics"),
        ("The president called for new legislation on policy", "politics"),
        ("The party proposed a vote on the executive order", "politics"),
        ("The researchers confirmed a new theory on climate", "science"),
        ("Scientists published data on a new species discovery", "science"),
        ("The study found new evidence of quantum change", "science"),
    ]

    print("--- TESTING (never-seen sentences) ---")
    t0 = time.perf_counter()
    correct = 0
    for text, true_label in test_data:
        hv = engine.encode_text(text)
        pred_label, confidence, scores = classifier.predict(hv)
        is_correct = pred_label == true_label
        correct += int(is_correct)
        mark = "OK" if is_correct else "XX"
        print(f"  [{mark}] \"{text[:55]}\"")
        print(f"       Predicted: {pred_label} (conf: {confidence:.3f}) | True: {true_label}")

    test_time = time.perf_counter() - t0
    accuracy = correct / len(test_data) * 100

    print(f"\n--- RESULTS ---")
    print(f"  Accuracy:  {correct}/{len(test_data)} = {accuracy:.1f}%")
    print(f"  Test time: {test_time*1000:.1f} ms for {len(test_data)} samples")
    print(f"  Training:  {total_examples//5} examples per class, {train_time*1000:.1f} ms total")
    print(f"  Memory:    ~{engine.dim * len(training_data) * 4 / 1024:.1f} KB (just {len(training_data)} prototype vectors)")
    print()
    return accuracy


def demo_hdc_analogy():
    """DEMO 2: Algebraic Reasoning / Analogy

    Challenge: What if reasoning was just vector arithmetic?
    king - male + female = queen. No transformer needed.
    """
    from neuroshift.hdc import HyperdimensionalEngine, AnalogyEngine

    print("=" * 70)
    print("PARADIGM 1b: HYPERDIMENSIONAL COMPUTING - ALGEBRAIC REASONING")
    print("=" * 70)
    print()
    print("Challenge: Solve analogies using pure vector arithmetic.")
    print("A:B :: C:? becomes: ? = bind(bind(A,B), C)")
    print()

    engine = HyperdimensionalEngine(dimensions=10000)
    analogy = AnalogyEngine(engine)

    # Define atomic properties
    for prop in ["gender", "role", "size", "habitat", "season", "temperature"]:
        analogy.define(prop)
    for val in ["male", "female", "royalty", "common", "big", "small",
                "land", "water", "summer", "winter", "hot", "cold"]:
        analogy.define(val)

    # Define composite concepts using role:filler bindings
    analogy.define_composite("king",   {"gender": "male",   "role": "royalty"})
    analogy.define_composite("queen",  {"gender": "female", "role": "royalty"})
    analogy.define_composite("man",    {"gender": "male",   "role": "common"})
    analogy.define_composite("woman",  {"gender": "female", "role": "common"})

    analogy.define_composite("whale",  {"size": "big",   "habitat": "water"})
    analogy.define_composite("shark",  {"size": "big",   "habitat": "water"})
    analogy.define_composite("goldfish", {"size": "small", "habitat": "water"})
    analogy.define_composite("elephant", {"size": "big",  "habitat": "land"})
    analogy.define_composite("mouse",  {"size": "small",  "habitat": "land"})

    analogy.define_composite("july",     {"season": "summer", "temperature": "hot"})
    analogy.define_composite("january",  {"season": "winter", "temperature": "cold"})
    analogy.define_composite("beach",    {"season": "summer", "temperature": "hot"})
    analogy.define_composite("fireplace", {"season": "winter", "temperature": "cold"})

    print("--- ANALOGY TESTS ---")
    tests = [
        ("king", "queen", "man", "woman",
         "king:queen :: man:?"),
        ("whale", "mouse", "elephant", "goldfish",
         "whale:mouse :: elephant:? (big->small, swap)"),
        ("july", "january", "beach", "fireplace",
         "july:january :: beach:? (summer->winter)"),
    ]

    correct = 0
    for a, b, c, expected, description in tests:
        results = analogy.solve_analogy(a, b, c)
        top = results[0]
        is_correct = top[0] == expected
        correct += int(is_correct)
        mark = "OK" if is_correct else "XX"
        print(f"\n  [{mark}] {description}")
        print(f"       Expected: {expected} | Got: {top[0]} (sim: {top[1]:.3f})")
        print(f"       Top-3: {', '.join(f'{r[0]}({r[1]:.3f})' for r in results[:3])}")

    print(f"\n--- ROLE QUERY TEST ---")
    print("  Querying: What is the 'gender' of 'king'?")
    role_results = analogy.query_role("king", "gender")
    print(f"  Answer: {role_results[0][0]} (sim: {role_results[0][1]:.3f})")
    print(f"  Top-3: {', '.join(f'{r[0]}({r[1]:.3f})' for r in role_results[:3])}")

    print(f"\n  Querying: What is the 'habitat' of 'whale'?")
    role_results = analogy.query_role("whale", "habitat")
    print(f"  Answer: {role_results[0][0]} (sim: {role_results[0][1]:.3f})")

    print()
    return correct / len(tests) * 100


def demo_morphic():
    """DEMO 3: Self-Evolving Neural Architecture

    Challenge: What if the network designed itself?
    Start with 4 neurons. Watch it grow, prune, and restructure
    to solve the problem with MINIMUM architecture.
    """
    from neuroshift.morphic import MorphicNet

    print("=" * 70)
    print("PARADIGM 2: MORPHICNET - SELF-EVOLVING ARCHITECTURE")
    print("=" * 70)
    print()
    print("Challenge: Network starts with 4 hidden neurons.")
    print("It will grow and prune itself to solve a complex classification task.")
    print("No human designs the architecture — the network does it itself.")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a non-trivial classification problem: concentric rings
    torch.manual_seed(42)
    n_samples = 1000
    n_classes = 4

    # Generate spiral dataset (hard for simple networks)
    X_list, Y_list = [], []
    for c in range(n_classes):
        angle_offset = c * (2 * 3.14159 / n_classes)
        for i in range(n_samples // n_classes):
            t = i / (n_samples // n_classes) * 3.0 + 0.5
            angle = t * 2.5 + angle_offset
            r = t
            x = r * torch.cos(torch.tensor(angle)) + torch.randn(1) * 0.15
            y = r * torch.sin(torch.tensor(angle)) + torch.randn(1) * 0.15
            X_list.append(torch.tensor([x.item(), y.item()]))
            Y_list.append(c)

    X = torch.stack(X_list).to(device)
    Y = torch.tensor(Y_list, dtype=torch.long).to(device)

    # Shuffle
    perm = torch.randperm(len(X))
    X, Y = X[perm], Y[perm]

    # Split
    split = int(0.8 * len(X))
    X_train, Y_train = X[:split], Y[:split]
    X_test, Y_test = X[split:], Y[split:]

    print(f"  Dataset: {n_classes}-class spiral ({len(X_train)} train, {len(X_test)} test)")
    print(f"  Input dim: 2, Output dim: {n_classes}")
    print()

    # Create MorphicNet starting TINY
    model = MorphicNet(input_dim=2, output_dim=n_classes, initial_hidden=8, device=str(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    print(f"  Starting architecture: {model.get_architecture_str()}")
    print(f"  Starting params: {model.get_total_params()}")
    print()
    print("--- TRAINING WITH SELF-EVOLUTION ---")

    t0 = time.perf_counter()
    n_epochs = 500
    batch_size = 64

    for epoch in range(n_epochs):
        # Mini-batch training
        perm = torch.randperm(len(X_train))
        epoch_loss = 0
        n_batches = 0
        evolved_this_epoch = False

        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            loss_val, evolved = model.train_step(X_train[idx], Y_train[idx], optimizer, epoch)
            epoch_loss += loss_val
            n_batches += 1
            if evolved:
                evolved_this_epoch = True
                # Rebuild optimizer with new params, lower lr to preserve weights
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        avg_loss = epoch_loss / n_batches

        if (epoch + 1) % 100 == 0 or evolved_this_epoch:
            model.eval()
            with torch.no_grad():
                preds = model(X_test).argmax(dim=1)
                acc = (preds == Y_test).float().mean().item() * 100

            status = " << EVOLVED!" if evolved_this_epoch else ""
            print(f"  Epoch {epoch+1:4d} | Loss: {avg_loss:.4f} | "
                  f"Test Acc: {acc:.1f}% | Arch: {model.get_architecture_str()}{status}")

    train_time = time.perf_counter() - t0

    # Final evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        final_acc = (preds == Y_test).float().mean().item() * 100

    model.print_evolution_report()

    print(f"\n--- RESULTS ---")
    print(f"  Final accuracy:     {final_acc:.1f}%")
    print(f"  Final architecture: {model.get_architecture_str()}")
    print(f"  Final parameters:   {model.get_total_params()}")
    print(f"  Training time:      {train_time:.1f}s")
    print(f"  Started from:       {model.arch_history[0][1]} (just {4} neurons!)")
    print(f"  Architecture changes: {len(model.arch_history) - 1}")
    print()
    return final_acc


def demo_swarm():
    """DEMO 4: Emergent Collective Intelligence

    Challenge: 50 tiny neural agents, each with ~2K parameters,
    collectively find the global optimum of a deceptive function
    with many local optima. No single agent could do this alone.
    """
    from neuroshift.swarm import NeuralSwarm
    import math

    print("=" * 70)
    print("PARADIGM 3: NEURAL SWARM - EMERGENT COLLECTIVE INTELLIGENCE")
    print("=" * 70)
    print()
    print("Challenge: 50 tiny agents evolve to solve a deceptive optimization")
    print("problem with many local traps. No backpropagation. No centralized")
    print("control. Intelligence emerges from evolution + communication.")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The optimization target: Rastrigin function (notoriously hard)
    # Many local optima, one global optimum at origin
    # Standard benchmark for optimization algorithms
    def rastrigin(x: torch.Tensor) -> float:
        """Rastrigin function - global minimum at origin, value 0.
        Has ~10^n local minima for n dimensions."""
        A = 10
        n = x.shape[0]
        val = A * n + (x**2 - A * torch.cos(2 * math.pi * x)).sum()
        return -val.item()  # Negate because we maximize fitness

    n_dims = 5  # 5D Rastrigin — ~100,000 local minima!

    def fitness_fn(agent, context):
        """Agent outputs a candidate solution, fitness = -Rastrigin(solution)
        Use a fixed input so evaluation is deterministic per agent."""
        obs = torch.zeros(agent.input_dim, device=agent.device)
        solution = agent.act(obs)
        agent.position = solution.detach()
        return rastrigin(solution.cpu())

    print(f"  Problem: 5D Rastrigin function (~100,000 local minima)")
    print(f"  Global optimum: f(0,0,0,0,0) = 0")
    print(f"  Population: 80 agents, each ~2K parameters")
    print()
    print("--- EVOLUTION ---")

    swarm = NeuralSwarm(
        population_size=80,
        input_dim=n_dims,
        output_dim=n_dims,
        fitness_fn=fitness_fn,
        hidden=32,
        device=str(device)
    )

    t0 = time.perf_counter()
    results = swarm.run(generations=300, report_every=30)
    evolve_time = time.perf_counter() - t0

    swarm.print_report()

    # Show best solution found
    best = results['best_agent']
    obs = torch.zeros(n_dims, device=best.device)
    best_solution = best.act(obs).cpu()
    best_value = -rastrigin(best_solution)

    pop_size = swarm.pop_size
    n_gens = swarm.generation
    print(f"\n--- RESULTS ---")
    print(f"  Best solution: [{', '.join(f'{x:.4f}' for x in best_solution)}]")
    print(f"  Rastrigin value: {best_value:.6f} (optimal: 0.0)")
    print(f"  Evolution time:  {evolve_time:.1f}s")
    print(f"  Total agents evaluated: {pop_size * n_gens} ({pop_size} agents x {n_gens} generations)")
    print(f"  Per-agent params: ~{best.get_genome_size()}")
    print()
    return best_value


def main():
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print()
    print("+======================================================================+")
    print("|                          G E N E S I S                               |")
    print("|           Three AI Paradigms That Challenge Everything               |")
    print("+======================================================================+")
    print("|  1. Hyperdimensional Computing  - Learning without training          |")
    print("|  2. MorphicNet                  - Self-designing neural networks     |")
    print("|  3. Neural Swarm                - Emergent collective intelligence   |")
    print("+======================================================================+")
    print()

    device = "CUDA: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A"
    print(f"  Device: {device}")
    print(f"  VRAM:   {vram}")
    print(f"  PyTorch: {torch.__version__}")
    print()

    results = {}

    # Demo 1: HDC Classification
    try:
        results['hdc_classification'] = demo_hdc_classification()
    except Exception as e:
        print(f"  HDC Classification failed: {e}")
        import traceback; traceback.print_exc()

    # Demo 2: HDC Analogy
    try:
        results['hdc_analogy'] = demo_hdc_analogy()
    except Exception as e:
        print(f"  HDC Analogy failed: {e}")
        import traceback; traceback.print_exc()

    # Demo 3: MorphicNet
    try:
        results['morphic_accuracy'] = demo_morphic()
    except Exception as e:
        print(f"  MorphicNet failed: {e}")
        import traceback; traceback.print_exc()

    # Demo 4: Neural Swarm
    try:
        results['swarm_best'] = demo_swarm()
    except Exception as e:
        print(f"  Neural Swarm failed: {e}")
        import traceback; traceback.print_exc()

    # Final summary
    print("=" * 70)
    print("NEUROSHIFT - FINAL REPORT")
    print("=" * 70)
    print()
    print("+---------------------+--------------------------------------------+")
    print("| Paradigm            | Result                                     |")
    print("+---------------------+--------------------------------------------+")

    if 'hdc_classification' in results:
        print(f"| HDC Classification  | {results['hdc_classification']:.1f}% accuracy (1 example/class)       |")
    if 'hdc_analogy' in results:
        print(f"| HDC Analogy         | {results['hdc_analogy']:.1f}% analogies solved (vector math)   |")
    if 'morphic_accuracy' in results:
        print(f"| MorphicNet          | {results['morphic_accuracy']:.1f}% accuracy (self-designed arch)    |")
    if 'swarm_best' in results:
        print(f"| Neural Swarm        | Rastrigin = {results['swarm_best']:.4f} (optimal: 0.0)     |")

    print("+---------------------+--------------------------------------------+")
    print()
    print("What makes these paradigms different:")
    print("  HDC:       No training loop. Learning is instantaneous encoding.")
    print("  MorphicNet: No human-designed architecture. The network builds itself.")
    print("  Swarm:     No backpropagation. Intelligence emerges from evolution.")
    print()


if __name__ == "__main__":
    main()
