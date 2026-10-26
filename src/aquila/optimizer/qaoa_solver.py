"""QAOA solver core: builds the Ising cost Hamiltonian for a task-allocation
problem, optimizes a QAOA ansatz against it via a classical outer loop, and
post-selects the best sampled assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
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
    counts: dict[int, int] = field(default_factory=dict)
    restart_costs: list[float] = field(default_factory=list)
    circuit_depth: int = 0
    two_qubit_gates: int = 0
    total_shots: int = 0


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

    transpiled_ansatz = backend.transpile(ansatz)
    mapped_cost_operator = cost_operator.apply_layout(transpiled_ansatz.layout)

    sampling_circuit = transpiled_ansatz.copy()
    sampling_circuit.measure_all()

    estimator = backend.estimator()
    sampler = backend.sampler()
    rng = np.random.default_rng(seed)

    best_result: QAOAResult | None = None
    restart_costs: list[float] = []
    total_shots = 0
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
            shots=backend.shots,
        )
        restart_costs.append(result.best_cost)
        total_shots += sum(result.counts.values())
        if best_result is None or result.best_cost < best_result.best_cost:
            best_result = result

    assert best_result is not None
    # Sampling happens after every restart; the reported assignment is the
    # lowest-cost sample across all restarts.
    best_result.restart_costs = restart_costs
    best_result.total_shots = total_shots
    ops = transpiled_ansatz.count_ops()
    best_result.circuit_depth = transpiled_ansatz.depth()
    best_result.two_qubit_gates = int(ops.get("cx", 0) + ops.get("rzz", 0) + ops.get("cz", 0))
    return best_result


def sample_unoptimized(
    problem: TaskAllocationProblem,
    backend: AerBackend,
    reps: int = 3,
    num_restarts: int = 3,
    seed: int | None = None,
) -> QAOAResult:
    """Control: run the same QAOA circuit at random (unoptimized) parameters
    with the same total sampling budget, so the value added by the classical
    optimization loop can be isolated."""
    cost_operator = build_cost_operator(problem)
    ansatz = QAOAAnsatz(cost_operator=cost_operator, reps=reps)
    transpiled = backend.transpile(ansatz)
    sampling_circuit = transpiled.copy()
    sampling_circuit.measure_all()
    sampler = backend.sampler()
    rng = np.random.default_rng(seed)

    merged: dict[int, int] = {}
    best_spins: list[int] | None = None
    best_cost = float("inf")
    for _ in range(num_restarts):
        params = rng.uniform(0, np.pi, transpiled.num_parameters)
        counts = sampler.run([(sampling_circuit, params, backend.shots)]).result()[0].data.meas.get_int_counts()
        for bitstring, count in counts.items():
            merged[bitstring] = merged.get(bitstring, 0) + count
            spins = problem.bitstring_to_spins(bitstring, problem.num_tasks)
            cost = problem.cost_of_assignment(spins)
            if cost < best_cost:
                best_cost, best_spins = cost, spins
    assert best_spins is not None
    return QAOAResult(
        best_spins=best_spins,
        best_cost=best_cost,
        optimal_parameters=[],
        counts=merged,
        total_shots=sum(merged.values()),
    )


def _solve_qaoa_once(
    problem: TaskAllocationProblem,
    transpiled_ansatz: QuantumCircuit,
    sampling_circuit: QuantumCircuit,
    mapped_cost_operator: SparsePauliOp,
    estimator: object,
    sampler: object,
    maxiter: int,
    seed: int,
    shots: int,
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

    job = sampler.run([(sampling_circuit, optimization.x, shots)])  # type: ignore[attr-defined]
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
        counts={int(k): int(v) for k, v in int_counts.items()},
    )
