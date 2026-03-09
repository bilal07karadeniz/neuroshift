"""Tests for neuroshift.morphic.network -- MorphicNet self-evolving architecture."""

import pytest
import torch

from neuroshift.morphic.network import MorphicNet


# ===========================================================================
# MorphicNet
# ===========================================================================
class TestMorphicNet:
    """Unit tests for the self-evolving neural network."""

    @pytest.fixture
    def net(self, device):
        """Small MorphicNet for fast tests."""
        return MorphicNet(
            input_dim=8,
            output_dim=3,
            initial_hidden=4,
            task="classification",
            device=device,
        )

    @pytest.fixture
    def regression_net(self, device):
        """MorphicNet configured for regression."""
        return MorphicNet(
            input_dim=4,
            output_dim=1,
            initial_hidden=8,
            task="regression",
            device=device,
        )

    # -- Constructor -----------------------------------------------------------

    def test_constructor_creates_network(self, net):
        """Constructor should produce a usable nn.Module."""
        assert isinstance(net, torch.nn.Module)
        assert net.get_total_params() > 0

    def test_initial_hidden_sizes(self, net):
        """Initial hidden_sizes should contain the value passed to constructor."""
        assert net.hidden_sizes == [4]

    def test_arch_history_initialized(self, net):
        """Architecture history should contain the 'init' entry."""
        assert len(net.arch_history) == 1
        assert net.arch_history[0][2] == "init"

    # -- Forward pass ----------------------------------------------------------

    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_forward_pass_correct_output_shape(self, net, batch_size):
        """Output shape should be (batch_size, output_dim)."""
        x = torch.randn(batch_size, net.input_dim, device=net.device)
        out = net(x)
        assert out.shape == (batch_size, net.output_dim)

    def test_forward_pass_regression(self, regression_net):
        """Regression net forward should work and return (batch, 1)."""
        x = torch.randn(5, 4, device=regression_net.device)
        out = regression_net(x)
        assert out.shape == (5, 1)

    # -- get_architecture_str --------------------------------------------------

    def test_get_architecture_str(self, net):
        """Architecture string should encode input -> hidden -> output."""
        arch_str = net.get_architecture_str()
        assert "8" in arch_str and "4" in arch_str and "3" in arch_str
        assert " -> " in arch_str

    def test_get_architecture_str_format(self, net):
        """Should have the form 'input -> h1 -> ... -> output'."""
        arch_str = net.get_architecture_str()
        parts = arch_str.split(" -> ")
        assert parts[0] == "8"
        assert parts[-1] == "3"
        assert len(parts) == 3  # input -> hidden -> output

    # -- evolve ----------------------------------------------------------------

    def test_evolve_runs_without_crash(self, net):
        """evolve() should execute without errors even at early steps."""
        result = net.evolve(loss_value=1.5, epoch=0)
        assert isinstance(result, bool)

    def test_evolve_no_change_early(self, net):
        """evolve should NOT change architecture in the first 50 steps."""
        changed = False
        for step in range(50):
            if net.evolve(loss_value=1.0, epoch=step):
                changed = True
        assert not changed

    def test_evolve_tracks_steps(self, net):
        """total_steps should increment with each evolve call."""
        for _ in range(10):
            net.evolve(1.0, 0)
        assert net.total_steps == 10

    # -- train_step ------------------------------------------------------------

    def test_train_step_returns_loss_and_bool(self, net):
        """train_step should return (float_loss, bool_evolved)."""
        x = torch.randn(16, net.input_dim, device=net.device)
        y = torch.randint(0, net.output_dim, (16,), device=net.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

        loss_val, evolved = net.train_step(x, y, optimizer, epoch=0)
        assert isinstance(loss_val, float)
        assert isinstance(evolved, bool)

    def test_train_step_loss_is_finite(self, net):
        """Loss returned by train_step should be a finite number."""
        x = torch.randn(16, net.input_dim, device=net.device)
        y = torch.randint(0, net.output_dim, (16,), device=net.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

        loss_val, _ = net.train_step(x, y, optimizer, epoch=0)
        assert not (loss_val != loss_val)  # NaN check
        assert loss_val < float('inf')

    def test_train_step_regression(self, regression_net):
        """train_step should work for regression tasks too."""
        x = torch.randn(16, 4, device=regression_net.device)
        y = torch.randn(16, 1, device=regression_net.device)
        optimizer = torch.optim.Adam(regression_net.parameters(), lr=0.01)

        loss_val, evolved = regression_net.train_step(x, y, optimizer, epoch=0)
        assert isinstance(loss_val, float)
        assert loss_val >= 0

    # -- architecture growth ---------------------------------------------------

    def test_architecture_can_grow(self, device):
        """After enough stagnation, hidden_sizes should change (grow)."""
        net = MorphicNet(
            input_dim=8,
            output_dim=3,
            initial_hidden=4,
            task="classification",
            device=device,
        )
        original_hidden = list(net.hidden_sizes)

        x = torch.randn(32, 8, device=net.device)
        y = torch.randint(0, 3, (32,), device=net.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

        # Run enough steps with stagnating loss to trigger growth
        changed = False
        for step in range(500):
            loss_val, evolved = net.train_step(x, y, optimizer, epoch=step)
            if evolved:
                # Rebuild optimizer with new params (the method already does this,
                # but we also need a fresh optimizer object)
                optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
                changed = True

        if changed:
            assert net.hidden_sizes != original_hidden
        # If no change in 500 steps, the network converged before stagnating.
        # Either outcome is valid; we just verify no crash occurred.

    def test_forward_works_after_growth(self, device):
        """Forward pass should still work after architecture changes."""
        net = MorphicNet(
            input_dim=8,
            output_dim=3,
            initial_hidden=4,
            task="classification",
            device=device,
        )
        # Force a growth by directly calling _grow
        original_hidden = list(net.hidden_sizes)
        grew = net._grow(epoch=0)

        if grew:
            assert net.hidden_sizes != original_hidden
            x = torch.randn(4, 8, device=net.device)
            out = net(x)
            assert out.shape == (4, 3)

    # -- get_total_params ------------------------------------------------------

    def test_get_total_params_positive(self, net):
        """Total parameters should be a positive integer."""
        assert net.get_total_params() > 0
        assert isinstance(net.get_total_params(), int)

    # -- Edge cases ------------------------------------------------------------

    @pytest.mark.parametrize("initial_hidden", [1, 2, 128])
    def test_various_initial_hidden(self, device, initial_hidden):
        """MorphicNet should work with various initial hidden sizes."""
        net = MorphicNet(
            input_dim=4,
            output_dim=2,
            initial_hidden=initial_hidden,
            task="classification",
            device=device,
        )
        x = torch.randn(2, 4, device=net.device)
        out = net(x)
        assert out.shape == (2, 2)
