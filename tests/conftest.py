"""Shared fixtures for the neuroshift test suite."""

import pytest
import torch


@pytest.fixture
def device():
    """Return the best available torch device for testing."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@pytest.fixture
def cpu_device():
    """Always return CPU device for deterministic tests."""
    return "cpu"
