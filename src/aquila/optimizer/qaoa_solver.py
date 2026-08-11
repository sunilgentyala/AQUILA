"""QAOA solver core: builds the Ising cost Hamiltonian for a task-allocation
problem, optimizes a QAOA ansatz against it via a classical outer loop, and
post-selects the best sampled assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from aquila.interface.backend import AerBackend
from aquila.interface.problem import TaskAllocationProblem


@dataclass
class QAOAResult:
    best_spins: list[int]
    best_cost: float
    optimal_parameters: list[float]
    objective_history: list[float] = field(default_factory=list)
    num_evaluations: int = 0


def build_cost_operator(problem: TaskAllocationProblem) -> SparsePauliOp:
    """Build the Ising cost Hamiltonian sum_{i<j} J_ij Z_i Z_j as a SparsePauliOp."""
    n = problem.num_tasks
    coefficients = problem.ising_coefficients()
    paulis: list[str] = []
    coeffs: list[float] = []
    for (i, j), value in coefficients.items():
        term = ["I"] * n
        term[i] = "Z"
        term[j] = "Z"
        # Qiskit Pauli strings are ordered with qubit 0 as the rightmost character.
        paulis.append("".join(reversed(term)))
        coeffs.append(value)
    return SparsePauliOp(paulis, coeffs)


def solve_qaoa(
    problem: TaskAllocationProblem,
    backend: AerBackend,
    reps: int = 2,
    maxiter: int = 150,
    num_restarts: int = 3,
    seed: int | None = None,
) -> QAOAResult:
    """Solve a task-allocation problem via QAOA and return the best assignment found.

    QAOA's classical outer loop is sensitive to parameter initialization,
    especially as qubit count grows, so `num_restarts` independent
    optimization runs (from different random initial parameters) are
    performed and the best result across all of them is kept - the same
    principle used for the simulated-annealing baseline.
    """
    cost_operator = build_cost_operator(problem)
    ansatz = QAOAAnsatz(cost_operator=cost_operator, reps=reps)

    simulator = backend.simulator()
    transpiled_ansatz = transpile(ansatz, simulator)
    mapped_cost_operator = cost_operator.apply_layout(transpiled_ansatz.layout)

    sampling_circuit = transpiled_ansatz.copy()
    sampling_circuit.measure_all()

    estimator = backend.estimator()
    sampler = backend.sampler()
    rng = np.random.default_rng(seed)

    best_result: QAOAResult | None = None
    for _ in range(num_restarts):
        restart_seed = int(rng.integers(0, 2**32 - 1))
        result = _solve_qaoa_once(
            problem,
            transpiled_ansatz,
            sampling_circuit,
            mapped_cost_operator,
            estimator,
            sampler,
            maxiter=maxiter,
            seed=restart_seed,
        )
        if best_result is None or result.best_cost < best_result.best_cost:
            best_result = result

    assert best_result is not None
    return best_result


def _solve_qaoa_once(
    problem: TaskAllocationProblem,
    transpiled_ansatz: QuantumCircuit,
    sampling_circuit: QuantumCircuit,
    mapped_cost_operator: SparsePauliOp,
    estimator: object,
    sampler: object,
    maxiter: int,
    seed: int,
) -> QAOAResult:
    history: list[float] = []

    def objective(params: np.ndarray) -> float:
        job = estimator.run([(transpiled_ansatz, mapped_cost_operator, params)])  # type: ignore[attr-defined]
        value = float(job.result()[0].data.evs)
        history.append(value)
        return value

    rng = np.random.default_rng(seed)
    initial_params = rng.uniform(0, np.pi, transpiled_ansatz.num_parameters)
    optimization = minimize(
        objective, initial_params, method="COBYLA", options={"maxiter": maxiter}
    )

    job = sampler.run([(sampling_circuit, optimization.x)])  # type: ignore[attr-defined]
    int_counts = job.result()[0].data.meas.get_int_counts()

    best_spins: list[int] | None = None
    best_cost = float("inf")
    for bitstring, _count in int_counts.items():
        spins = problem.bitstring_to_spins(bitstring, problem.num_tasks)
        cost = problem.cost_of_assignment(spins)
        if cost < best_cost:
            best_cost = cost
            best_spins = spins

    assert best_spins is not None
    return QAOAResult(
        best_spins=best_spins,
        best_cost=best_cost,
        optimal_parameters=list(optimization.x),
        objective_history=history,
        num_evaluations=len(history),
    )
