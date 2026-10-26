"""Experiment 1 (revised): QAOA vs. matched-budget controls and strong classical baselines.

Per instance it records, for every method: the achieved squared imbalance,
the normalized imbalance |sum w_i s_i| / sum w_i, exact-optimum success,
and wall-clock time. For the QAOA variants it also records the
probability mass on optimal assignments *before* best-sample selection,
circuit depth, two-qubit gate count and optimizer evaluations.

Instance seeds and optimizer seeds are drawn independently.

Run with: python evaluation/quality_controls.py [--instances 30] [--workers 4]
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import write_csv  # noqa: E402

from aquila.interface.backend import AerBackend  # noqa: E402
from aquila.interface.problem import TaskAllocationProblem  # noqa: E402
from aquila.optimizer.classical_baselines import (  # noqa: E402
    brute_force_optimal,
    dp_subset_sum_cost,
    greedy_partition,
    karmarkar_karp_partition,
    simulated_annealing_partition,
    simulated_annealing_tuned,
    uniform_random_sampling,
)
from aquila.optimizer.qaoa_solver import sample_unoptimized, solve_qaoa  # noqa: E402

SIZES = [6, 8, 10, 12, 14]
FAMILIES = ("real_1_20", "int_1_100")
SHOTS = 1024
RESTARTS = 3
REPS = 3
MAXITER = 150
TOL = 1e-9


def make_problem(family: str, n: int, instance_seed: int) -> TaskAllocationProblem:
    import random

    rng = random.Random(instance_seed)
    if family == "real_1_20":
        weights = [round(rng.uniform(1.0, 20.0), 3) for _ in range(n)]
    else:
        weights = [float(rng.randint(1, 100)) for _ in range(n)]
    return TaskAllocationProblem(task_weights=weights)


def optimal_assignments(problem: TaskAllocationProblem, opt_cost: float) -> set[int]:
    """All bitstrings (little-endian ints) within TOL of the optimum."""
    n = problem.num_tasks
    w = problem.task_weights
    found = set()
    for b in range(2**n):
        d = sum(w[i] if (b >> i) & 1 else -w[i] for i in range(n))
        if d * d <= opt_cost + TOL * max(1.0, opt_cost):
            found.add(b)
    return found


def norm_imb(cost: float, total: float) -> float:
    return math.sqrt(max(cost, 0.0)) / total


def run_instance(args: tuple[str, int, int]) -> dict:
    family, n, k = args
    instance_seed = 10_000 * n + k
    optimizer_seed = 7_000_003 + 31 * k + n  # independent of the instance seed
    problem = make_problem(family, n, instance_seed)
    total = sum(problem.task_weights)
    budget = SHOTS * RESTARTS

    t = time.perf_counter()
    _, opt_cost = brute_force_optimal(problem)
    t_exact = time.perf_counter() - t
    t = time.perf_counter()
    dp_cost = dp_subset_sum_cost(problem, decimals=3)
    t_dp = time.perf_counter() - t
    assert abs(dp_cost - opt_cost) <= 1e-6 * max(1.0, opt_cost), (dp_cost, opt_cost)
    opt_set = optimal_assignments(problem, opt_cost)

    def timed(fn):
        t0 = time.perf_counter()
        r = fn()
        return r, time.perf_counter() - t0

    (_, c_greedy), t_greedy = timed(lambda: greedy_partition(problem))
    (_, c_kk), t_kk = timed(lambda: karmarkar_karp_partition(problem))
    (_, c_sa_paper), t_sa_paper = timed(
        lambda: simulated_annealing_partition(problem, iterations=2000, restarts=5, seed=optimizer_seed)
    )
    (_, c_sa), t_sa = timed(lambda: simulated_annealing_tuned(problem, budget, seed=optimizer_seed))
    (_, c_uni), t_uni = timed(lambda: uniform_random_sampling(problem, budget, seed=optimizer_seed))

    backend = AerBackend(seed=optimizer_seed % 2**31, shots=SHOTS)
    t0 = time.perf_counter()
    q = solve_qaoa(problem, backend, reps=REPS, maxiter=MAXITER, num_restarts=RESTARTS, seed=optimizer_seed)
    t_qaoa = time.perf_counter() - t0
    t0 = time.perf_counter()
    u = sample_unoptimized(problem, backend, reps=REPS, num_restarts=RESTARTS, seed=optimizer_seed)
    t_unopt = time.perf_counter() - t0

    def p_opt(counts: dict[int, int]) -> float:
        tot = sum(counts.values())
        return sum(c for b, c in counts.items() if b in opt_set) / tot if tot else float("nan")

    def ok(c: float) -> int:
        return int(c <= opt_cost + TOL * max(1.0, opt_cost))

    row = {
        "family": family, "n": n, "instance": k, "instance_seed": instance_seed,
        "optimizer_seed": optimizer_seed, "total_weight": total, "opt_cost": opt_cost,
        "num_optimal_assignments": len(opt_set),
        "uniform_p_opt_per_shot": len(opt_set) / 2**n,
        "sample_budget": budget,
    }
    for name, c, tm in (
        ("qaoa", q.best_cost, t_qaoa), ("qaoa_unopt", u.best_cost, t_unopt),
        ("uniform", c_uni, t_uni), ("sa_tuned", c_sa, t_sa), ("sa_paper", c_sa_paper, t_sa_paper),
        ("kk", c_kk, t_kk), ("greedy", c_greedy, t_greedy),
    ):
        row[f"{name}_cost"] = c
        row[f"{name}_norm_imb"] = norm_imb(c, total)
        row[f"{name}_success"] = ok(c)
        row[f"{name}_time_s"] = tm
    row["exact_time_s"] = t_exact
    row["dp_time_s"] = t_dp
    row["qaoa_p_opt_best_restart"] = p_opt(q.counts)
    row["qaoa_unopt_p_opt"] = p_opt(u.counts)
    row["qaoa_evaluations"] = q.num_evaluations
    row["qaoa_depth"] = q.circuit_depth
    row["qaoa_two_qubit_gates"] = q.two_qubit_gates
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = [(f, n, k) for f in FAMILIES for n in SIZES for k in range(a.instances)]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(run_instance, jobs, chunksize=1), 1):
            rows.append(r)
            print(f"[{i}/{len(jobs)}] {r['family']} n={r['n']} k={r['instance']} "
                  f"qaoa_ok={r['qaoa_success']} uni_ok={r['uniform_success']} sa_ok={r['sa_tuned_success']} "
                  f"kk_ok={r['kk_success']}", flush=True)
    path = write_csv("quality_controls.csv", list(rows[0].keys()), rows)
    print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
