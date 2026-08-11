"""Problem representation for agentic task/resource-allocation workloads.

An agent submits a set of tasks, each carrying a resource-cost weight, to be
split across two compute clusters while minimizing load imbalance. This is
the classical number-partitioning problem: NP-hard in general, and it maps
directly onto an Ising cost Hamiltonian, making it a natural first workload
for the quantum-classical interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskAllocationProblem:
    """A two-cluster agentic task/resource allocation problem.

    Spins s_i in {-1, +1} denote which cluster task i is assigned to.
    Minimizing (sum_i w_i * s_i)^2 minimizes the load imbalance between
    the two clusters.
    """

    task_weights: list[float]
    task_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.task_weights:
            raise ValueError("task_weights must be non-empty")
        if not self.task_ids:
            self.task_ids = [f"task-{i}" for i in range(len(self.task_weights))]
        if len(self.task_ids) != len(self.task_weights):
            raise ValueError("task_ids and task_weights must be the same length")
        if any(w <= 0 for w in self.task_weights):
            raise ValueError("task_weights must be strictly positive")

    @property
    def num_tasks(self) -> int:
        return len(self.task_weights)

    def ising_coefficients(self) -> dict[tuple[int, int], float]:
        """Pairwise coupling coefficients J_ij for cost = (sum_i w_i s_i)^2.

        The constant term sum_i w_i^2 is dropped since it is independent of
        the spin assignment and does not change the argmin.
        """
        w = self.task_weights
        n = self.num_tasks
        return {(i, j): 2.0 * w[i] * w[j] for i in range(n) for j in range(i + 1, n)}

    def cost_of_assignment(self, spins: list[int]) -> float:
        """Evaluate (sum_i w_i s_i)^2 for a +-1 spin assignment (lower is better)."""
        if len(spins) != self.num_tasks:
            raise ValueError("spins length must match num_tasks")
        if any(s not in (-1, 1) for s in spins):
            raise ValueError("spins must be -1 or +1")
        total = sum(w * s for w, s in zip(self.task_weights, spins, strict=True))
        return total * total

    def clusters_from_assignment(self, spins: list[int]) -> tuple[list[str], list[str]]:
        """Split task ids into (cluster A, cluster B) given a spin assignment."""
        cluster_a = [tid for tid, s in zip(self.task_ids, spins, strict=True) if s == 1]
        cluster_b = [tid for tid, s in zip(self.task_ids, spins, strict=True) if s == -1]
        return cluster_a, cluster_b

    @staticmethod
    def bitstring_to_spins(bitstring: int, num_bits: int) -> list[int]:
        """Convert a Qiskit little-endian measurement outcome to +-1 spins.

        Qubit 0 (task 0) is the least-significant bit.
        """
        bits = [(bitstring >> i) & 1 for i in range(num_bits)]
        return [1 if b == 1 else -1 for b in bits]
