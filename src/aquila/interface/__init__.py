"""Quantum-classical interface layer: problem representation and pluggable backends."""

from aquila.interface.backend import AerBackend, depolarizing_noise_backend
from aquila.interface.problem import TaskAllocationProblem

__all__ = ["AerBackend", "TaskAllocationProblem", "depolarizing_noise_backend"]
