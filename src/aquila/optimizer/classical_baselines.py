"""Classical baselines for the task-allocation (number-partitioning) problem.

These give AQUILA's evaluation something concrete to compare QAOA against:
an exact solver for ground truth on small instances, a standard greedy
heuristic, and simulated annealing as a stronger classical metaheuristic.
"""

from __future__ import annotations

import itertools
import random

from aquila.interface.problem import TaskAllocationProblem


def brute_force_optimal(problem: TaskAllocationProblem) -> tuple[list[int], float]:
    """Exact solver by exhaustive search, used as ground truth on small instances.

    Fixes the first spin to +1 to exploit the global sign symmetry of the
    cost function, halving the search space.
    """
    n = problem.num_tasks
    best_spins: list[int] | None = None
    best_cost = float("inf")
    for combo in itertools.product((-1, 1), repeat=n - 1):
        spins = [1, *combo]
        cost = problem.cost_of_assignment(spins)
        if cost < best_cost:
            best_cost = cost
            best_spins = spins
    assert best_spins is not None
    return best_spins, best_cost


def greedy_partition(problem: TaskAllocationProblem) -> tuple[list[int], float]:
    """Longest-processing-time greedy heuristic: assign each task, largest
    weight first, to whichever cluster currently has the smaller load.
    """
    order = sorted(range(problem.num_tasks), key=lambda i: problem.task_weights[i], reverse=True)
    spins = [0] * problem.num_tasks
    load_a, load_b = 0.0, 0.0
    for i in order:
        w = problem.task_weights[i]
        if load_a <= load_b:
            spins[i] = 1
            load_a += w
        else:
            spins[i] = -1
            load_b += w
    return spins, problem.cost_of_assignment(spins)


def simulated_annealing_partition(
    problem: TaskAllocationProblem,
    iterations: int = 2000,
    initial_temp: float = 10.0,
    cooling_rate: float = 0.995,
    restarts: int = 5,
    seed: int | None = None,
) -> tuple[list[int], float]:
    """Single-spin-flip simulated annealing over the partition cost landscape.

    Runs `restarts` independent chains (standard practice for SA on rugged
    small-instance landscapes, where a single chain can get trapped in a
    local optimum) and returns the best result found across all of them.
    """
    rng = random.Random(seed)
    best_overall_spins: list[int] | None = None
    best_overall_cost = float("inf")

    for _ in range(restarts):
        chain_seed = rng.randrange(2**32)
        spins, cost = _anneal_once(problem, iterations, initial_temp, cooling_rate, chain_seed)
        if cost < best_overall_cost:
            best_overall_spins, best_overall_cost = spins, cost

    assert best_overall_spins is not None
    return best_overall_spins, best_overall_cost


def _anneal_once(
    problem: TaskAllocationProblem,
    iterations: int,
    initial_temp: float,
    cooling_rate: float,
    seed: int,
) -> tuple[list[int], float]:
    rng = random.Random(seed)
    n = problem.num_tasks
    spins = [rng.choice((-1, 1)) for _ in range(n)]
    cost = problem.cost_of_assignment(spins)
    best_spins, best_cost = list(spins), cost
    temp = initial_temp

    for _ in range(iterations):
        idx = rng.randrange(n)
        spins[idx] *= -1
        new_cost = problem.cost_of_assignment(spins)
        delta = new_cost - cost
        if delta <= 0 or rng.random() < _acceptance_probability(delta, temp):
            cost = new_cost
            if cost < best_cost:
                best_spins, best_cost = list(spins), cost
        else:
            spins[idx] *= -1  # revert
        temp *= cooling_rate

    return best_spins, best_cost


def _acceptance_probability(delta: float, temp: float) -> float:
    if temp <= 0:
        return 0.0
    try:
        import math

        return math.exp(-delta / temp)
    except OverflowError:
        return 0.0
