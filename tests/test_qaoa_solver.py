import pytest

from aquila.interface.backend import AerBackend
from aquila.interface.problem import TaskAllocationProblem
from aquila.optimizer.classical_baselines import brute_force_optimal
from aquila.optimizer.qaoa_solver import build_cost_operator, solve_qaoa


def test_cost_operator_matches_manual_ising_expectation():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    problem = TaskAllocationProblem(task_weights=[1.0, 2.0, 3.0])
    cost_operator = build_cost_operator(problem)

    spins = [1, -1, -1]  # qubit0=X(applied), qubit1,2 = |0>
    qc = QuantumCircuit(problem.num_tasks)
    qc.x(0)
    expectation = Statevector.from_instruction(qc).expectation_value(cost_operator).real

    constant_term = sum(w**2 for w in problem.task_weights)
    full_cost = problem.cost_of_assignment(spins)
    assert expectation == pytest.approx(full_cost - constant_term)


def test_solve_qaoa_finds_optimal_on_small_instance():
    problem = TaskAllocationProblem(task_weights=[3.0, 5.0, 7.0, 2.0])
    backend = AerBackend(seed=7, shots=4096)
    result = solve_qaoa(problem, backend, reps=3, maxiter=150, seed=7)

    _, optimal_cost = brute_force_optimal(problem)
    assert result.best_cost == pytest.approx(optimal_cost)
    assert len(result.best_spins) == problem.num_tasks
    assert all(s in (-1, 1) for s in result.best_spins)
    assert result.num_evaluations > 0


def test_solve_qaoa_result_cost_matches_recomputation():
    problem = TaskAllocationProblem(task_weights=[4.0, 6.0, 5.0])
    backend = AerBackend(seed=3, shots=2048)
    result = solve_qaoa(problem, backend, reps=2, maxiter=80, seed=3)

    assert result.best_cost == pytest.approx(problem.cost_of_assignment(result.best_spins))
