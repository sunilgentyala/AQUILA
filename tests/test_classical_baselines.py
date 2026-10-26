import pytest

from aquila.interface.problem import TaskAllocationProblem
from aquila.optimizer.classical_baselines import (
    brute_force_optimal,
    greedy_partition,
    simulated_annealing_partition,
)


@pytest.fixture
def small_problem():
    return TaskAllocationProblem(task_weights=[3.0, 5.0, 7.0, 2.0, 9.0])


def test_brute_force_finds_known_optimum(small_problem):
    _, cost = brute_force_optimal(small_problem)
    assert cost == pytest.approx(4.0)


def test_brute_force_perfect_partition_is_zero_cost():
    # {1, 2} vs {3} both sum to 3
    problem = TaskAllocationProblem(task_weights=[1.0, 2.0, 3.0])
    _, cost = brute_force_optimal(problem)
    assert cost == pytest.approx(0.0)


def test_greedy_partition_is_valid_and_reasonable(small_problem):
    spins, cost = greedy_partition(small_problem)
    assert len(spins) == small_problem.num_tasks
    assert all(s in (-1, 1) for s in spins)
    assert cost == small_problem.cost_of_assignment(spins)
    _, optimal_cost = brute_force_optimal(small_problem)
    assert cost >= optimal_cost


def test_simulated_annealing_matches_or_beats_greedy(small_problem):
    _, greedy_cost = greedy_partition(small_problem)
    _, sa_cost = simulated_annealing_partition(small_problem, iterations=1000, restarts=5, seed=42)
    _, optimal_cost = brute_force_optimal(small_problem)
    assert sa_cost >= optimal_cost
    assert sa_cost <= greedy_cost


def test_simulated_annealing_reproducible_with_seed(small_problem):
    _, cost_a = simulated_annealing_partition(small_problem, seed=99)
    _, cost_b = simulated_annealing_partition(small_problem, seed=99)
    assert cost_a == cost_b


@pytest.mark.parametrize("seed", range(5))
def test_simulated_annealing_multi_restart_finds_optimum_across_seeds(small_problem, seed):
    _, cost = simulated_annealing_partition(small_problem, iterations=2000, restarts=5, seed=seed)
    _, optimal_cost = brute_force_optimal(small_problem)
    assert cost == pytest.approx(optimal_cost)
