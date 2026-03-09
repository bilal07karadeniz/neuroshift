"""
NEURAL SWARM - EMERGENT COLLECTIVE INTELLIGENCE
=================================================
A population of small neural agents that evolve, communicate, and
collectively solve problems that no individual agent could solve alone.

Key differences from standard neural networks:
  - No single model - intelligence emerges from MANY tiny agents
  - No backpropagation - evolution drives learning
  - No centralized control - agents self-organize
  - Specialization emerges naturally (some agents explore, others exploit)

Inspired by ant colonies, bird flocking, and biological evolution.

Use cases:
  - Distributed optimization (find global optima in complex landscapes)
  - Creative problem solving (diverse population = diverse solutions)
  - Robust AI (no single point of failure)
  - Multi-objective optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Callable, Optional, Dict
import math
import random


class NeuralAgent:
    """A tiny neural network agent with its own brain, fitness, and energy."""

    def __init__(self, input_dim: int, output_dim: int, hidden: int = 32,
                 device: torch.device = torch.device('cpu')):
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = hidden

        # Small brain - 2 layer network
        self.brain = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, output_dim),
        ).to(device)

        self.fitness = 0.0
        self.energy = 1.0
        self.age = 0
        self.species_id = 0
        self.n_offspring = 0
        self.best_fitness = float('-inf')
        self.position = torch.zeros(output_dim, device=device)

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        """Decide action based on observation."""
        with torch.no_grad():
            return self.brain(observation.to(self.device))

    def get_genome_size(self) -> int:
        return sum(p.numel() for p in self.brain.parameters())

    def copy(self) -> 'NeuralAgent':
        """Deep copy this agent."""
        clone = NeuralAgent(self.input_dim, self.output_dim,
                            hidden=self.hidden, device=self.device)
        clone.brain.load_state_dict(self.brain.state_dict())
        clone.species_id = self.species_id
        return clone


class NeuralSwarm:
    """Ecosystem of evolving neural agents solving problems collectively.

    The swarm uses a combination of:
      - Tournament selection (survival of the fittest)
      - Gaussian mutation (exploration of nearby solutions)
      - Crossover (combining good solutions)
      - Elitism (preserving the best)
      - Communication (agents share information about good regions)
    """

    def __init__(self, population_size: int, input_dim: int, output_dim: int,
                 fitness_fn: Callable, hidden: int = 32, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.pop_size = population_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fitness_fn = fitness_fn

        # Create population
        self.agents: List[NeuralAgent] = []
        for i in range(population_size):
            agent = NeuralAgent(input_dim, output_dim, hidden, self.device)
            agent.species_id = i % 5  # Start with 5 species
            self.agents.append(agent)

        # Evolution parameters
        self.mutation_rate = 0.1
        self.mutation_decay = 0.999
        self.crossover_rate = 0.3
        self.elite_fraction = 0.1
        self.tournament_size = 5

        # Tracking
        self.generation = 0
        self.history: List[Dict] = []
        self.best_ever_fitness = float('-inf')
        self.best_ever_agent: Optional[NeuralAgent] = None
        self.stagnation_counter = 0

    def evaluate(self, context: Optional[torch.Tensor] = None):
        """Evaluate fitness of all agents."""
        for agent in self.agents:
            agent.fitness = self.fitness_fn(agent, context)
            agent.best_fitness = max(agent.best_fitness, agent.fitness)

        # Track global best
        best = max(self.agents, key=lambda a: a.fitness)
        if best.fitness > self.best_ever_fitness:
            self.best_ever_fitness = best.fitness
            self.best_ever_agent = best.copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

    def _tournament_select(self) -> NeuralAgent:
        """Select agent via tournament selection."""
        candidates = random.sample(self.agents, min(self.tournament_size, len(self.agents)))
        return max(candidates, key=lambda a: a.fitness)

    def _mutate(self, agent: NeuralAgent, rate: float):
        """Gaussian mutation of agent's neural weights."""
        with torch.no_grad():
            for param in agent.brain.parameters():
                mask = torch.rand_like(param) < rate
                noise = torch.randn_like(param) * self.mutation_rate
                param.data += mask.float() * noise

    def _crossover(self, parent1: NeuralAgent, parent2: NeuralAgent) -> NeuralAgent:
        """Uniform crossover between two parents."""
        child = parent1.copy()
        with torch.no_grad():
            for p_child, p_parent2 in zip(child.brain.parameters(), parent2.brain.parameters()):
                mask = torch.rand_like(p_child) < 0.5
                p_child.data[mask] = p_parent2.data[mask]
        return child

    def _communicate(self):
        """Agents in the same species share information about good regions.
        Top performers influence others in their species."""
        species: Dict[int, List[NeuralAgent]] = {}
        for agent in self.agents:
            species.setdefault(agent.species_id, []).append(agent)

        for sid, members in species.items():
            if len(members) < 2:
                continue
            members.sort(key=lambda a: a.fitness, reverse=True)
            leader = members[0]
            # Bottom half of species gets pulled toward leader
            for follower in members[len(members) // 2:]:
                with torch.no_grad():
                    for p_f, p_l in zip(follower.brain.parameters(), leader.brain.parameters()):
                        p_f.data += 0.05 * (p_l.data - p_f.data)

    def evolve_generation(self, context: Optional[torch.Tensor] = None):
        """Run one generation of evolution."""
        self.generation += 1

        # Evaluate
        self.evaluate(context)

        # Sort by fitness
        self.agents.sort(key=lambda a: a.fitness, reverse=True)

        # Record stats
        fitnesses = [a.fitness for a in self.agents]
        species_counts: Dict[int, int] = {}
        for a in self.agents:
            species_counts[a.species_id] = species_counts.get(a.species_id, 0) + 1

        self.history.append({
            'generation': self.generation,
            'best': fitnesses[0],
            'worst': fitnesses[-1],
            'mean': sum(fitnesses) / len(fitnesses),
            'median': fitnesses[len(fitnesses) // 2],
            'std': (sum((f - sum(fitnesses)/len(fitnesses))**2 for f in fitnesses) / len(fitnesses)) ** 0.5,
            'n_species': len(species_counts),
            'best_ever': self.best_ever_fitness,
        })

        # Communication within species
        self._communicate()

        # Create next generation
        n_elite = max(2, int(self.pop_size * self.elite_fraction))
        new_pop = [a.copy() for a in self.agents[:n_elite]]  # Elitism

        # Always inject best-ever agent to prevent population degradation
        if self.best_ever_agent is not None:
            new_pop[0] = self.best_ever_agent.copy()

        while len(new_pop) < self.pop_size:
            if random.random() < self.crossover_rate:
                p1 = self._tournament_select()
                p2 = self._tournament_select()
                child = self._crossover(p1, p2)
            else:
                parent = self._tournament_select()
                child = parent.copy()

            # Mutation with adaptive rate
            effective_rate = self.mutation_rate
            if self.stagnation_counter > 10:
                effective_rate *= 3.0  # Bigger exploration when stuck
            elif self.stagnation_counter > 5:
                effective_rate *= 1.5
            self._mutate(child, effective_rate)

            child.age = 0
            child.fitness = 0
            new_pop.append(child)

        # Age all agents
        for a in new_pop:
            a.age += 1

        self.agents = new_pop

        # Decay mutation rate (slower decay)
        self.mutation_rate *= self.mutation_decay
        self.mutation_rate = max(0.02, self.mutation_rate)

        # Speciation: reassign species based on weight similarity
        if self.generation % 10 == 0:
            self._respeciate()

    def _respeciate(self):
        """Reassign species based on neural weight similarity."""
        if len(self.agents) < 5:
            return

        # Use first layer weights as species signature
        signatures = []
        for agent in self.agents:
            first_layer = list(agent.brain.parameters())[0]
            signatures.append(first_layer.data.flatten()[:100])  # First 100 weights

        sigs = torch.stack(signatures)
        # Simple k-means-ish: find 5 centroids from top agents
        n_species = min(5, len(self.agents))
        centroids = sigs[:n_species].clone()

        for agent, sig in zip(self.agents, sigs):
            dists = torch.cdist(sig.unsqueeze(0), centroids)
            agent.species_id = dists.argmin().item()

    def run(self, generations: int, context: Optional[torch.Tensor] = None,
            report_every: int = 10) -> Dict:
        """Run evolution for N generations."""
        for g in range(generations):
            self.evolve_generation(context)

            if (g + 1) % report_every == 0:
                h = self.history[-1]
                print(f"  Gen {h['generation']:4d} | "
                      f"Best: {h['best']:+.4f} | "
                      f"Mean: {h['mean']:+.4f} | "
                      f"Species: {h['n_species']} | "
                      f"All-time: {h['best_ever']:+.4f}")

        return {
            'best_fitness': self.best_ever_fitness,
            'best_agent': self.best_ever_agent,
            'generations': self.generation,
            'history': self.history,
            'final_population': self.agents,
        }

    def print_report(self):
        """Print evolution summary."""
        if not self.history:
            print("No evolution history yet.")
            return

        print(f"\n=== NEURAL SWARM EVOLUTION REPORT ===")
        print(f"  Generations:    {self.generation}")
        print(f"  Population:     {self.pop_size}")
        print(f"  Best fitness:   {self.best_ever_fitness:+.6f}")
        print(f"  Final mean:     {self.history[-1]['mean']:+.6f}")
        print(f"  Final std:      {self.history[-1]['std']:.6f}")
        print(f"  Species:        {self.history[-1]['n_species']}")
        print(f"  Mutation rate:  {self.mutation_rate:.4f}")

        # Show improvement curve
        checkpoints = [0, len(self.history)//4, len(self.history)//2,
                       3*len(self.history)//4, len(self.history)-1]
        print(f"\n  Evolution curve:")
        for idx in checkpoints:
            if 0 <= idx < len(self.history):
                h = self.history[idx]
                bar_len = int(max(0, (h['best'] + 5) * 4))  # Scale for display
                bar = "#" * min(bar_len, 40)
                print(f"    Gen {h['generation']:4d}: {bar} {h['best']:+.4f}")
