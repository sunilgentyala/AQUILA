"""Confirm on the Aer path that cost normalization changes QAOA's behaviour (n = 8, 10, 12).

For each instance: raw vs normalized cost, same seeds, p = 3, 3 restarts x 1,024 shots.
Records the probability mass on optimal states of the returned (best-cost) restart, the
exact-optimum success, and the optimizer evaluations. Run: python evaluation/normalization_aer_check.py
"""

from __future__ import annotations

import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import write_csv  # noqa: E402

from aquila.interface.backend import AerBackend  # noqa: E402
from aquila.interface.problem import TaskAllocationProblem  # noqa: E402
from aquila.optimizer.classical_baselines import brute_force_optimal  # noqa: E402
from aquila.optimizer.qaoa_solver import solve_qaoa  # noqa: E402

INSTANCES = 12


def job(args: tuple[int, int, bool]) -> dict:
    n, k, norm = args
    rng = random.Random(10_000 * n + k)
    w = [round(rng.uniform(1.0, 20.0), 3) for _ in range(n)]
    prob = TaskAllocationProblem(task_weights=w)
    _, opt = brute_force_optimal(prob)
    opt_set = set()
    for b in range(2**n):
        d = sum(w[i] if (b >> i) & 1 else -w[i] for i in range(n))
        if d * d <= opt + 1e-9 * max(1.0, opt):
            opt_set.add(b)
    seed = 7_000_003 + 31 * k + n
    res = solve_qaoa(prob, AerBackend(seed=seed % 2**31, shots=1024), reps=3, maxiter=150,
                     num_restarts=3, seed=seed, normalize=norm)
    tot = sum(res.counts.values())
    p_opt = sum(c for b, c in res.counts.items() if b in opt_set) / tot
    return {"n": n, "instance": k, "normalized": int(norm), "p_opt": p_opt,
            "p_uniform": len(opt_set) / 2**n, "success": int(res.best_cost <= opt + 1e-9 * max(1.0, opt)),
            "evaluations": res.num_evaluations}


if __name__ == "__main__":
    jobs = [(n, k, nm) for n in (8, 10, 12) for k in range(INSTANCES) for nm in (False, True)]
    with ProcessPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(job, jobs, chunksize=1))
    write_csv("normalization_aer_check.csv", list(rows[0].keys()), rows)
    for n in (8, 10, 12):
        for nm in (0, 1):
            g = [r for r in rows if r["n"] == n and r["normalized"] == nm]
            e = sum(r["p_opt"] for r in g) / sum(r["p_uniform"] for r in g)
            print(n, "norm" if nm else "raw ", "enrichment %.2f" % e, "success %d/%d" % (sum(r["success"] for r in g), len(g)),
                  "mean evals %.0f" % (sum(r["evaluations"] for r in g) / len(g)))
