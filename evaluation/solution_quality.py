"""Experiment 1: QAOA solution quality vs. classical baselines across problem sizes.

For each problem size, several random workload instances are solved by QAOA
and by each classical baseline; the approximation ratio (optimal cost /
found cost, 1.0 = optimal) is recorded for every run.

Run with: python evaluation/solution_quality.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import write_csv  # noqa: E402

from aquila.agents.harness import generate_workload  # noqa: E402
from aquila.interface.backend import AerBackend  # noqa: E402
from aquila.optimizer.classical_baselines import (  # noqa: E402
    brute_force_optimal,
    greedy_partition,
    simulated_annealing_partition,
)
from aquila.optimizer.qaoa_solver import solve_qaoa  # noqa: E402

PROBLEM_SIZES = [4, 6, 8, 10, 12, 14]
SEEDS_PER_SIZE = 5


def approx_ratio(optimal_cost: float, found_cost: float) -> float:
    """1.0 = matched the optimum. Handles the zero-cost (perfect partition) case."""
    if optimal_cost == 0.0:
        return 1.0 if found_cost == 0.0 else 0.0
    return optimal_cost / found_cost if found_cost > 0 else 1.0


def main() -> None:
    rows = []
    for n in PROBLEM_SIZES:
        for seed in range(SEEDS_PER_SIZE):
            problem = generate_workload(n, seed=seed * 1000 + n)
            _, optimal_cost = brute_force_optimal(problem)

            _, greedy_cost = greedy_partition(problem)
            _, sa_cost = simulated_annealing_partition(
                problem, iterations=2000, restarts=5, seed=seed
            )

            backend = AerBackend(seed=seed, shots=4096)
            t0 = time.perf_counter()
            qaoa_result = solve_qaoa(
                problem, backend, reps=3, maxiter=150, num_restarts=3, seed=seed
            )
            qaoa_time = time.perf_counter() - t0

            row = {
                "num_tasks": n,
                "seed": seed,
                "optimal_cost": optimal_cost,
                "greedy_cost": greedy_cost,
                "greedy_ratio": approx_ratio(optimal_cost, greedy_cost),
                "sa_cost": sa_cost,
                "sa_ratio": approx_ratio(optimal_cost, sa_cost),
                "qaoa_cost": qaoa_result.best_cost,
                "qaoa_ratio": approx_ratio(optimal_cost, qaoa_result.best_cost),
                "qaoa_time_sec": qaoa_time,
                "qaoa_evaluations": qaoa_result.num_evaluations,
            }
            rows.append(row)
            print(
                f"n={n:2d} seed={seed}  optimal={optimal_cost:.4f}  "
                f"greedy_ratio={row['greedy_ratio']:.3f}  sa_ratio={row['sa_ratio']:.3f}  "
                f"qaoa_ratio={row['qaoa_ratio']:.3f}  ({qaoa_time:.2f}s)"
            )

    path = write_csv("solution_quality.csv", list(rows[0].keys()), rows)
    print(f"\nWrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
