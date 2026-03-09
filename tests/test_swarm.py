"""Tests for neuroshift.swarm.ecosystem -- NeuralAgent and NeuralSwarm."""

import pytest
import torch

from neuroshift.swarm.ecosystem import NeuralAgent, NeuralSwarm


# ===========================================================================
# NeuralAgent
# ===========================================================================
class TestNeuralAgent:
    """Unit tests for individual neural agents."""

    @pytest.fixture
    def agent(self, device):
        dev = torch.device(device)
        return NeuralAgent(input_dim=4, output_dim=2, hidden=32, device=dev)

    # -- act -------------------------------------------------------------------

    @pytest.mark.parametrize("batch_shape", [(4,), (1, 4), (8, 4)])
    def test_act_returns_correct_shape(self, agent, batch_shape):
        """act() should return a tensor with output_dim as last dimension."""
        obs = torch.randn(*batch_shape, device=agent.device)
        result = agent.act(obs)
        expected_shape = batch_shape[:-1] + (2,)
        assert result.shape == expected_shape

    def test_act_no_grad(self, agent):
        """act() should not create a computation graph."""
        obs = torch.randn(4, device=agent.device)
        result = agent.act(obs)
        assert not result.requires_grad

    # -- copy ------------------------------------------------------------------

    def test_copy_preserves_weights(self, agent):
        """copy() should produce an agent with identical weights."""
        clone = agent.copy()
        for p_orig, p_clone in zip(agent.brain.parameters(), clone.brain.parameters()):
            assert torch.allclose(p_orig, p_clone)

    def test_copy_is_independent(self, agent):
        """Modifying the copy should not affect the original."""
        clone = agent.copy()
        with torch.no_grad():
            for p in clone.brain.parameters():
                p.data.fill_(999.0)
        for p in agent.brain.parameters():
            assert not (p.data == 999.0).all()

    def test_copy_preserves_species_id(self, agent):
        """copy() should preserve species_id."""
        agent.species_id = 42
        clone = agent.copy()
        assert clone.species_id == 42

    # -- get_genome_size -------------------------------------------------------

    def test_get_genome_size_positive(self, agent):
        """Genome size should be > 0."""
        assert agent.get_genome_size() > 0

    def test_get_genome_size_matches_parameters(self, agent):
        """get_genome_size should equal sum of all parameter elements."""
        expected = sum(p.numel() for p in agent.brain.parameters())
        assert agent.get_genome_size() == expected

    # -- initial state ---------------------------------------------------------

    def test_initial_fitness(self, agent):
        """Initial fitness should be 0."""
        assert agent.fitness == 0.0

    def test_initial_energy(self, agent):
        """Initial energy should be 1."""
        assert agent.energy == 1.0

    @pytest.mark.parametrize("hidden", [8, 16, 64])
    def test_various_hidden_sizes(self, device, hidden):
        """Agent should work with various hidden layer sizes."""
        dev = torch.device(device)
        agent = NeuralAgent(input_dim=4, output_dim=2, hidden=hidden, device=dev)
        obs = torch.randn(4, device=dev)
        result = agent.act(obs)
        assert result.shape == (2,)


# ===========================================================================
# NeuralSwarm
# ===========================================================================
class TestNeuralSwarm:
    """Unit tests for swarm-based neuroevolution."""

    @staticmethod
    def _sphere_fitness(agent, context):
        """Simple sphere function: minimize sum of output squared.
        Higher fitness = better, so negate the squared sum."""
        obs = torch.zeros(agent.input_dim, device=agent.device)
        output = agent.act(obs)
        return -output.pow(2).sum().item()

    @pytest.fixture
    def swarm(self, device):
        return NeuralSwarm(
            population_size=20,
            input_dim=4,
            output_dim=2,
            fitness_fn=self._sphere_fitness,
            hidden=16,
            device=device,
        )

    # -- constructor -----------------------------------------------------------

    def test_constructor_creates_population(self, swarm):
        """Swarm should have the requested number of agents."""
        assert len(swarm.agents) == 20

    def test_constructor_agents_are_neural_agents(self, swarm):
        """All swarm members should be NeuralAgent instances."""
        for agent in swarm.agents:
            assert isinstance(agent, NeuralAgent)

    def test_constructor_initial_state(self, swarm):
        """Generation counter and history should start empty."""
        assert swarm.generation == 0
        assert swarm.history == []
        assert swarm.best_ever_fitness == float('-inf')

    # -- evaluate --------------------------------------------------------------

    def test_evaluate_runs(self, swarm):
        """evaluate() should execute without errors."""
        swarm.evaluate()
        # After evaluate, at least one agent should have non-zero fitness
        fitnesses = [a.fitness for a in swarm.agents]
        assert any(f != 0.0 for f in fitnesses)

    def test_evaluate_sets_best_ever(self, swarm):
        """After evaluate, best_ever should be updated."""
        swarm.evaluate()
        assert swarm.best_ever_fitness > float('-inf')
        assert swarm.best_ever_agent is not None

    # -- evolve_generation -----------------------------------------------------

    def test_evolve_generation_runs(self, swarm):
        """evolve_generation should complete without errors."""
        swarm.evolve_generation()
        assert swarm.generation == 1
        assert len(swarm.history) == 1

    def test_evolve_generation_records_history(self, swarm):
        """History entry should contain the expected keys."""
        swarm.evolve_generation()
        h = swarm.history[0]
        expected_keys = {'generation', 'best', 'worst', 'mean', 'median', 'std',
                         'n_species', 'best_ever'}
        assert expected_keys.issubset(h.keys())

    def test_population_size_preserved(self, swarm):
        """Population size should remain constant across generations."""
        for _ in range(5):
            swarm.evolve_generation()
        assert len(swarm.agents) == 20

    # -- run -------------------------------------------------------------------

    def test_run_returns_dict_with_expected_keys(self, swarm):
        """run() should return a dict containing all expected keys."""
        result = swarm.run(generations=5, report_every=100)  # suppress output
        expected_keys = {'best_fitness', 'best_agent', 'generations', 'history',
                         'final_population'}
        assert expected_keys.issubset(result.keys())

    def test_run_correct_generation_count(self, swarm):
        """After run(N), swarm.generation should equal N."""
        swarm.run(generations=10, report_every=100)
        assert swarm.generation == 10

    def test_run_history_length(self, swarm):
        """History should contain one entry per generation."""
        swarm.run(generations=7, report_every=100)
        assert len(swarm.history) == 7

    # -- fitness improvement ---------------------------------------------------

    def test_best_fitness_improves_over_generations(self, device):
        """Best-ever fitness should generally improve (or at least not worsen)
        over multiple generations for a simple optimization problem."""
        swarm = NeuralSwarm(
            population_size=30,
            input_dim=4,
            output_dim=2,
            fitness_fn=self._sphere_fitness,
            hidden=16,
            device=device,
        )

        result = swarm.run(generations=30, report_every=100)
        # best_ever is tracked cumulatively so it can only stay same or improve
        first_best = swarm.history[0]['best_ever']
        last_best = swarm.history[-1]['best_ever']
        assert last_best >= first_best

    def test_best_ever_monotonically_nondecreasing(self, swarm):
        """best_ever in history should be monotonically non-decreasing."""
        swarm.run(generations=15, report_every=100)
        best_evers = [h['best_ever'] for h in swarm.history]
        for i in range(1, len(best_evers)):
            assert best_evers[i] >= best_evers[i - 1]

    # -- edge cases ------------------------------------------------------------

    @pytest.mark.parametrize("pop_size", [5, 10, 50])
    def test_various_population_sizes(self, device, pop_size):
        """Swarm should work with different population sizes."""
        swarm = NeuralSwarm(
            population_size=pop_size,
            input_dim=2,
            output_dim=1,
            fitness_fn=self._sphere_fitness,
            hidden=8,
            device=device,
        )
        swarm.run(generations=3, report_every=100)
        assert len(swarm.agents) == pop_size
        assert len(swarm.history) == 3
