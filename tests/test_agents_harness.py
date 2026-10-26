import pytest

from aquila.agents.harness import AllocationAgent, generate_workload
from aquila.interface.backend import AerBackend
from aquila.optimizer.classical_baselines import brute_force_optimal


def test_generate_workload_reproducible_with_seed():
    a = generate_workload(num_tasks=5, seed=1)
    b = generate_workload(num_tasks=5, seed=1)
    assert a.task_weights == b.task_weights


def test_generate_workload_respects_task_count():
    problem = generate_workload(num_tasks=7, seed=2)
    assert problem.num_tasks == 7


def test_end_to_end_secured_allocation_matches_optimal():
    requester = AllocationAgent.create("scheduler-agent")
    solver = AllocationAgent.create("quantum-solver-agent")
    problem = generate_workload(num_tasks=4, seed=5)

    backend = AerBackend(seed=13, shots=4096)
    result = requester.request_allocation(problem, solver, backend, reps=3, maxiter=150, seed=13)

    _, optimal_cost = brute_force_optimal(problem)
    assert result.best_cost == pytest.approx(optimal_cost)


def test_end_to_end_uses_distinct_agent_identities():
    requester = AllocationAgent.create("scheduler-agent")
    solver = AllocationAgent.create("quantum-solver-agent")
    assert requester.identity.kem_public_key != solver.identity.kem_public_key
    assert requester.name != solver.name
