"""Extension study (v0.3): depth sweep, scaling to n = 20, and shot-cost accounting.

Runs QAOA on the exact NumPy simulator (validated against Aer in
validate_fast_qaoa.py) so that n = 16..20 is tractable, and records for every
instance the exact probability mass on optimal assignments. From that mass the
analysis derives success-within-budget curves and the total quantum shot cost
of reaching a target success probability, counting the variational loop.

Objective: exact expectation (infinite shots). This favours QAOA, so a
negative result here is conservative.

Run: python evaluation/extension_study.py --workers 14
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import fast_qaoa as fq  # noqa: E402
from common import write_csv  # noqa: E402

from aquila.interface.problem import TaskAllocationProblem  # noqa: E402
from aquila.optimizer.classical_baselines import simulated_annealing_tuned  # noqa: E402

SHOTS, RESTARTS, MAXITER, TOL = 1024, 3, 150, 1e-9
BUDGET = SHOTS * RESTARTS


def weights_for(n: int, k: int) -> list[float]:
    rng = random.Random(10_000 * n + k)  # same instance seeds as quality_controls.py
    return [round(rng.uniform(1.0, 20.0), 3) for _ in range(n)]


NORM = os.environ.get("AQUILA_NORM") == "1"


def run(job: tuple[int, int, int]) -> dict:
    n, p, k = job
    w = weights_for(n, k)
    total = sum(w)
    seed = 7_000_003 + 31 * k + n + 100_000 * p
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    cost = fq.cost_diagonal(w)
    opt = float(cost.min())
    t_exact = time.perf_counter() - t0
    is_opt = cost <= opt + TOL * max(1.0, opt)
    n_opt = int(is_opt.sum())
    p_uniform = n_opt / 2**n
    h = fq.ising_diagonal(w)
    if NORM:  # rescale so the cost spectrum has unit standard deviation (gamma then lives on a sensible scale)
        h = h / float(h.std())

    # Optimized QAOA: 3 random restarts, COBYLA on the exact expectation.
    evals, best_cost, p_opts, exps = 0, float("inf"), [], []
    t0 = time.perf_counter()
    for _ in range(RESTARTS):
        x0 = rng.uniform(0, np.pi, 2 * p)
        count = [0]

        def f(x: np.ndarray) -> float:
            count[0] += 1
            return fq.expectation(h, n, x)

        res = minimize(f, x0, method="COBYLA", options={"maxiter": MAXITER})
        evals += count[0]
        pr = fq.probabilities(h, n, res.x)
        samples = rng.choice(2**n, size=SHOTS, p=pr)
        best_cost = min(best_cost, float(cost[samples].min()))
        p_opts.append(float(pr[is_opt].sum()))
        exps.append(float(res.fun))
    t_qaoa = time.perf_counter() - t0
    i_best = int(np.argmin(exps))

    # Control: same circuit, random (unoptimized) parameters, same sample budget.
    ctrl_cost, ctrl_p = float("inf"), []
    for _ in range(RESTARTS):
        x0 = rng.uniform(0, np.pi, 2 * p)
        pr = fq.probabilities(h, n, x0)
        samples = rng.choice(2**n, size=SHOTS, p=pr)
        ctrl_cost = min(ctrl_cost, float(cost[samples].min()))
        ctrl_p.append(float(pr[is_opt].sum()))

    # Uniform sampling and tuned simulated annealing at the same budget.
    uni = float(cost[rng.integers(0, 2**n, size=BUDGET)].min())
    prob = TaskAllocationProblem(task_weights=w)
    _, sa_cost = simulated_annealing_tuned(prob, BUDGET, seed=seed)

    def ok(c: float) -> int:
        return int(c <= opt + TOL * max(1.0, opt))

    return {
        "n": n, "p": p, "instance": k, "normalized": int(NORM), "opt_cost": opt, "total_weight": total,
        "num_optimal": n_opt, "p_uniform": p_uniform,
        "qaoa_p_opt_lowest_energy_restart": p_opts[i_best],
        "qaoa_p_opt_mean_restarts": float(np.mean(p_opts)),
        "ctrl_p_opt_mean": float(np.mean(ctrl_p)),
        "qaoa_success": ok(best_cost), "ctrl_success": ok(ctrl_cost),
        "uniform_success": ok(uni), "sa_success": ok(sa_cost),
        "qaoa_norm_imb": float(np.sqrt(max(best_cost, 0)) / total),
        "uniform_norm_imb": float(np.sqrt(max(uni, 0)) / total),
        "evaluations": evals, "qaoa_time_s": t_qaoa, "exact_time_s": t_exact,
    }


def main() -> None:
    import csv

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--small", type=int, default=30)
    ap.add_argument("--large", type=int, default=20)
    ap.add_argument("--huge", type=int, default=8, help="instances at n = 20")
    ap.add_argument("--only", default="all")
    a = ap.parse_args()
    jobs = []
    if a.only in ("all", "depth"):
        jobs += [(n, p, k) for n in (10, 12, 14) for p in (1, 2, 3, 4, 6) for k in range(a.small)]
    if a.only in ("all", "scale"):
        jobs += [(n, 3, k) for n in (16, 18) for k in range(a.large)]
        jobs += [(20, 3, k) for k in range(a.huge)]
    jobs.sort(key=lambda j: -(j[0] * 10 + j[1]))  # slowest first
    path = Path(__file__).parent / "results" / f"extension_study_{a.only}{'_norm' if NORM else ''}.partial.csv"
    rows, writer, fh = [], None, path.open("w", newline="")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(run, jobs, chunksize=1), 1):  # ordered, flushed as each arrives
            rows.append(r)
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(r.keys()))
                writer.writeheader()
            writer.writerow(r)
            fh.flush()
            print(f"[{i}/{len(jobs)}] n={r['n']} p={r['p']} k={r['instance']} t={r['qaoa_time_s']:.0f}s", flush=True)
    fh.close()
    rows.sort(key=lambda r: (r["n"], r["p"], r["instance"]))
    out = write_csv(f"extension_study_{a.only}{'_norm' if NORM else ''}.csv", list(rows[0].keys()), rows)
    print("Wrote", len(rows), "rows to", out)


if __name__ == "__main__":
    main()
