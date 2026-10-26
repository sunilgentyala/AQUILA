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


def karmarkar_karp_partition(problem: TaskAllocationProblem) -> tuple[list[int], float]:
    """Karmarkar-Karp largest-differencing heuristic, the standard strong
    polynomial-time baseline for number partitioning.

    Repeatedly replaces the two largest items by their difference, then
    unwinds the merges to recover a +-1 assignment.
    """
    import heapq

    n = problem.num_tasks
    # heap entries: (-value, tiebreak, positive_members, negative_members)
    heap = [(-w, i, [i], []) for i, w in enumerate(problem.task_weights)]
    heapq.heapify(heap)
    counter = n
    while len(heap) > 1:
        v1, _, pos1, neg1 = heapq.heappop(heap)
        v2, _, pos2, neg2 = heapq.heappop(heap)
        # largest minus second largest: second group is flipped to the other side
        merged = (v1 - v2, counter, pos1 + neg2, neg1 + pos2)
        counter += 1
        heapq.heappush(heap, merged)
    _, _, pos, neg = heap[0]
    spins = [0] * n
    for i in pos:
        spins[i] = 1
    for i in neg:
        spins[i] = -1
    return spins, problem.cost_of_assignment(spins)


def dp_subset_sum_cost(problem: TaskAllocationProblem, decimals: int = 3) -> float:
    """Exact pseudo-polynomial dynamic program (reachable-sum bitset).

    Weights are scaled to integers at `decimals` places. Returns the optimal
    squared imbalance in the original units. The scaled problem is exact for
    weights specified to at most `decimals` places.
    """
    scale = 10**decimals
    ints = [round(w * scale) for w in problem.task_weights]
    total = sum(ints)
    reachable = 1
    for w in ints:
        reachable |= reachable << w
    best = None
    for s in range(total // 2, -1, -1):
        if (reachable >> s) & 1:
            best = s
            break
    assert best is not None
    imbalance = (total - 2 * best) / scale
    return imbalance * imbalance


def uniform_random_sampling(
    problem: TaskAllocationProblem, num_samples: int, seed: int | None = None
) -> tuple[list[int], float]:
    """Control: best of `num_samples` independent uniform random assignments."""
    rng = random.Random(seed)
    n = problem.num_tasks
    best_spins: list[int] | None = None
    best_cost = float("inf")
    for _ in range(num_samples):
        spins = [rng.choice((-1, 1)) for _ in range(n)]
        cost = problem.cost_of_assignment(spins)
        if cost < best_cost:
            best_spins, best_cost = spins, cost
    assert best_spins is not None
    return best_spins, best_cost


def simulated_annealing_tuned(
    problem: TaskAllocationProblem, total_flips: int, seed: int | None = None, restarts: int = 3
) -> tuple[list[int], float]:
    """Scale-aware simulated annealing with a fixed total evaluation budget.

    Works on the absolute imbalance |sum w_i s_i| so the temperature is in
    load units, starts at the mean weight and cools geometrically to 1e-3
    of it, with single-spin-flip moves. `total_flips` is split across
    `restarts` chains so the budget is matched to a sampling budget.
    """
    import math

    rng = random.Random(seed)
    n = problem.num_tasks
    w = problem.task_weights
    t0 = sum(w) / n
    t_end = t0 * 1e-3
    per_chain = max(1, total_flips // restarts)
    cooling = (t_end / t0) ** (1.0 / per_chain)
    best_spins: list[int] | None = None
    best_cost = float("inf")
    for _ in range(restarts):
        spins = [rng.choice((-1, 1)) for _ in range(n)]
        d = sum(wi * si for wi, si in zip(w, spins, strict=True))
        temp = t0
        for _ in range(per_chain):
            idx = rng.randrange(n)
            new_d = d - 2 * spins[idx] * w[idx]
            delta = abs(new_d) - abs(d)
            if delta <= 0 or rng.random() < math.exp(-delta / temp):
                spins[idx] = -spins[idx]
                d = new_d
                if d * d < best_cost:
                    best_cost = d * d
                    best_spins = list(spins)
            temp *= cooling
        if best_spins is None:
            best_spins, best_cost = list(spins), d * d
    return best_spins, best_cost
