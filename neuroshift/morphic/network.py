"""
MORPHICNET - SELF-EVOLVING NEURAL ARCHITECTURE
================================================
A neural network that rewrites its own topology during training.
Not just weight updates - it grows new neurons, prunes dead ones,
and adds entire layers when needed.

Key differences from Neural Architecture Search (NAS):
  - NAS searches OUTSIDE the model, then trains from scratch
  - MorphicNet evolves INSIDE the model, preserving learned weights
  - Architecture changes happen continuously during training
  - The network starts tiny and grows only what it needs

Use cases:
  - Continual learning (network adapts structure to new tasks)
  - Resource-efficient AI (minimum neurons for the job)
  - Automated model design (no manual architecture tuning)
  - Edge deployment (self-compressing networks)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import copy


class MorphicNet(nn.Module):
    """Neural network that evolves its own architecture during training.

    Starts with a minimal topology and grows/prunes based on:
      - Loss signal: high loss -> grow (need more capacity)
      - Neuron utility: low activation -> prune (wasted capacity)
      - Layer depth: all layers saturated -> add new layer
    """

    def __init__(self, input_dim: int, output_dim: int, initial_hidden: int = 4,
                 task: str = "classification", device: str = "auto"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task = task

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Start minimal
        self.hidden_sizes = [initial_hidden]
        self._build()

        # Evolution parameters
        self.max_neurons_per_layer = 256
        self.max_layers = 6

        # Tracking
        self.neuron_utility: List[torch.Tensor] = []
        self.arch_history: List[Tuple[int, List[int], str]] = []
        self.arch_history.append((0, list(self.hidden_sizes), "init"))
        self.total_steps = 0
        self.recent_losses: List[float] = []
        self._cooldown = 0  # Steps to wait after an architecture change
        self._best_loss = float('inf')
        self._stagnation = 0

    def _build(self):
        """Build the network from current hidden_sizes specification."""
        layers = []
        sizes = [self.input_dim] + self.hidden_sizes + [self.output_dim]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:  # No activation on output
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.net.to(self.device)

        # Reset utility tracking
        self.neuron_utility = [
            torch.zeros(s, device=self.device) for s in self.hidden_sizes
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with utility tracking."""
        current = x
        hidden_idx = 0
        for module in self.net:
            current = module(current)
            if isinstance(module, nn.ReLU) and hidden_idx < len(self.neuron_utility):
                with torch.no_grad():
                    magnitude = current.abs().mean(dim=0)
                    u = self.neuron_utility[hidden_idx]
                    self.neuron_utility[hidden_idx] = 0.95 * u + 0.05 * magnitude
                hidden_idx += 1
        return current

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_architecture_str(self) -> str:
        return " -> ".join(
            [str(self.input_dim)] +
            [str(s) for s in self.hidden_sizes] +
            [str(self.output_dim)]
        )

    def evolve(self, loss_value: float, epoch: int) -> bool:
        """Attempt to evolve architecture. Returns True if changed.

        Uses stagnation-driven evolution:
          - Track if loss is improving
          - Only grow when loss has stagnated (not just 'high')
          - Prune conservatively and rarely
          - Cooldown period after each change to let weights settle
        """
        self.total_steps += 1
        self.recent_losses.append(loss_value)
        if len(self.recent_losses) > 200:
            self.recent_losses = self.recent_losses[-200:]

        # Cooldown: let the network train for a while after architecture changes
        if self._cooldown > 0:
            self._cooldown -= 1
            return False

        # Need at least 50 steps of data before evolving
        if self.total_steps < 50:
            return False

        # Check for stagnation: compare recent loss to older loss
        if len(self.recent_losses) >= 40:
            old_avg = sum(self.recent_losses[-40:-20]) / 20
            new_avg = sum(self.recent_losses[-20:]) / 20
            improvement = (old_avg - new_avg) / (abs(old_avg) + 1e-8)

            if improvement < 0.01:  # Less than 1% improvement
                self._stagnation += 1
            else:
                self._stagnation = 0

        # Track best loss
        if loss_value < self._best_loss:
            self._best_loss = loss_value

        changed = False

        # --- GROWTH: only when stagnated ---
        if self._stagnation >= 3:
            changed = self._grow(epoch)
            if changed:
                self._stagnation = 0
                self._cooldown = 80  # Let it train for 80 steps

        # --- DEEPEN: all layers wide but still stagnated ---
        if (self._stagnation >= 6
                and all(s >= 64 for s in self.hidden_sizes)
                and len(self.hidden_sizes) < self.max_layers):
            added = self._add_layer(epoch)
            changed = changed or added
            if added:
                self._stagnation = 0
                self._cooldown = 100

        # --- PRUNING: only after significant training, and very conservatively ---
        if (self.total_steps >= 200
                and self.total_steps % 100 == 0
                and self._stagnation < 2):  # Don't prune when already struggling
            pruned = self._prune(epoch)
            if pruned:
                changed = True
                self._cooldown = 60

        return changed

    def _grow(self, epoch: int) -> bool:
        """Widen the narrowest hidden layer."""
        min_idx = min(range(len(self.hidden_sizes)), key=lambda i: self.hidden_sizes[i])
        old_size = self.hidden_sizes[min_idx]
        new_size = min(old_size + max(2, old_size // 2), self.max_neurons_per_layer)

        if new_size <= old_size:
            return False

        old_weights = self._save_weights()
        self.hidden_sizes[min_idx] = new_size
        self._build()
        self._restore_weights(old_weights)
        self.arch_history.append((epoch, list(self.hidden_sizes), f"grow layer {min_idx}: {old_size}->{new_size}"))
        return True

    def _prune(self, epoch: int) -> bool:
        """Remove truly dead neurons (near-zero utility). Very conservative."""
        changed = False
        for i, utility in enumerate(self.neuron_utility):
            if self.hidden_sizes[i] <= 8:
                continue

            # Only prune neurons with near-zero utility (truly dead)
            mean_util = utility.mean()
            if mean_util < 1e-8:
                continue
            dead_mask = utility < (mean_util * 0.01)  # Less than 1% of mean
            dead = dead_mask.sum().item()
            if dead == 0:
                continue

            # Remove at most 25% of neurons at once
            max_remove = max(1, self.hidden_sizes[i] // 4)
            dead = min(dead, max_remove)
            new_size = max(8, self.hidden_sizes[i] - dead)
            if new_size == self.hidden_sizes[i]:
                continue

            old_size = self.hidden_sizes[i]
            old_weights = self._save_weights()
            self.hidden_sizes[i] = new_size
            self._build()
            self._restore_weights(old_weights)
            self.arch_history.append((epoch, list(self.hidden_sizes), f"prune layer {i}: {old_size}->{new_size}"))
            changed = True

        return changed

    def _add_layer(self, epoch: int) -> bool:
        """Insert a new narrow hidden layer in the middle."""
        insert_pos = len(self.hidden_sizes) // 2
        old_weights = self._save_weights()
        self.hidden_sizes.insert(insert_pos, 8)
        self._build()
        # Can't restore weights cleanly when adding layers - partial restore
        self._restore_weights_partial(old_weights, inserted_at=insert_pos)
        self.arch_history.append((epoch, list(self.hidden_sizes), f"add layer at pos {insert_pos}"))
        return True

    def _save_weights(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Save weight matrices from all Linear layers."""
        weights = []
        for module in self.net:
            if isinstance(module, nn.Linear):
                weights.append((module.weight.data.clone(), module.bias.data.clone()))
        return weights

    def _restore_weights(self, old_weights: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Restore weights, copying the overlapping region."""
        new_linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for i, layer in enumerate(new_linears):
            if i >= len(old_weights):
                break
            old_w, old_b = old_weights[i]
            min_out = min(layer.weight.shape[0], old_w.shape[0])
            min_in = min(layer.weight.shape[1], old_w.shape[1])
            layer.weight.data[:min_out, :min_in] = old_w[:min_out, :min_in]
            layer.bias.data[:min_out] = old_b[:min_out]

    def _restore_weights_partial(self, old_weights, inserted_at: int):
        """Restore weights when a new layer was inserted."""
        new_linears = [m for m in self.net if isinstance(m, nn.Linear)]
        old_idx = 0
        for new_idx, layer in enumerate(new_linears):
            # The inserted layer and the one after it are new
            if new_idx == inserted_at or new_idx == inserted_at + 1:
                # Initialize as near-identity for smooth insertion
                if layer.weight.shape[0] == layer.weight.shape[1]:
                    nn.init.eye_(layer.weight)
                else:
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)
                if new_idx == inserted_at + 1:
                    old_idx += 1  # Skip one old weight set
                continue

            if old_idx < len(old_weights):
                old_w, old_b = old_weights[old_idx]
                min_out = min(layer.weight.shape[0], old_w.shape[0])
                min_in = min(layer.weight.shape[1], old_w.shape[1])
                layer.weight.data[:min_out, :min_in] = old_w[:min_out, :min_in]
                layer.bias.data[:min_out] = old_b[:min_out]
                old_idx += 1

    def train_step(self, x: torch.Tensor, y: torch.Tensor,
                   optimizer: torch.optim.Optimizer, epoch: int) -> Tuple[float, bool]:
        """Single training step with potential architecture evolution."""
        self.train()
        optimizer.zero_grad()
        out = self(x)

        if self.task == "classification":
            loss = F.cross_entropy(out, y)
        else:
            loss = F.mse_loss(out, y)

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        evolved = self.evolve(loss_val, epoch)

        # If architecture changed, rebuild optimizer
        if evolved:
            for pg in optimizer.param_groups:
                pg['params'] = list(self.parameters())

        return loss_val, evolved

    def print_evolution_report(self):
        """Print the architecture evolution history."""
        print("\n=== MORPHICNET EVOLUTION HISTORY ===")
        for epoch, arch, reason in self.arch_history:
            total = self.input_dim
            for i, s in enumerate(arch):
                if i == 0:
                    total = self.input_dim * s + s
                else:
                    total += arch[i - 1] * s + s
            total += arch[-1] * self.output_dim + self.output_dim
            print(f"  Epoch {epoch:4d} | {' -> '.join(str(s) for s in arch):30s} | {reason}")
        print(f"  Final: {self.get_architecture_str()} ({self.get_total_params()} params)")
