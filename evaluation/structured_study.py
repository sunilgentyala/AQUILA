"""Structured-workload study (v0.4): communication-aware allocation, not number partitioning.

The earlier studies use a two-cluster number-partitioning cost, a single-quadratic
problem with a huge degenerate optimum set. Agent workloads are not like that: tasks
exchange data, so a placement pays for every message that crosses a cluster boundary.
Here the allocation cost is

    c(s) = sum_{i<j} C_ij [s_i != s_j]  +  lam * (sum_i w_i s_i)^2 / W^2 * Cmax

(cut communication cost plus a load-imbalance penalty), on three structured families:
  community  planted-partition graph (dense inside groups, sparse between)
  pipeline   chain with skip links, the shape of a multi-stage agent workflow
  random     Erdos-Renyi graph with random message volumes

QAOA runs on the exact NumPy simulator (fast_qaoa) with the exact expectation
(infinite shots, which favours QAOA). Every method gets the same sample budget of
3 x 1,024 cost evaluations. Controls: uniform sampling, unoptimized circuit, tuned
simulated annealing, multi-start 1-flip local search, exact enumeration.

Run: python evaluation/structured_study.py --workers 14
"""

from __future__ import annotations

import argparse
import csv
import os
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

SHOTS, RESTARTS, MAXITER, TOL = 1024, 3, 150, 1e-9
BUDGET = SHOTS * RESTARTS
FAMILIES = ("community", "pipeline", "random")


def make_instance(family: str, n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (communication matrix C, load vector w) from a fixed seed."""
    rng = np.random.default_rng(20_000 * n + 97 * k + FAMILIES.index(family))
    c = np.zeros((n, n))
    if family == "community":
        grp = rng.integers(0, 3, n)
        for i in range(n):
            for j in range(i + 1, n):
                pr = 0.75 if grp[i] == grp[j] else 0.12
                if rng.random() < pr:
                    c[i, j] = rng.uniform(1.0, 10.0)
    elif family == "pipeline":
        for i in range(n - 1):
            c[i, i + 1] = rng.uniform(4.0, 10.0)
        for i in range(n - 2):
            if rng.random() < 0.5:
                c[i, i + 2] = rng.uniform(1.0, 5.0)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.4:
                    c[i, j] = rng.uniform(1.0, 10.0)
    c = c + c.T
    w = np.round(rng.uniform(1.0, 10.0, n), 3)
    return c, w


def cost_diagonal(c: np.ndarray, w: np.ndarray, lam: float = 0.5) -> np.ndarray:
    n = len(w)
    idx = np.arange(2**n, dtype=np.int64)
    s = ((idx[:, None] >> np.arange(n)) & 1) * 2 - 1  # (2^n, n) in {-1,+1}
    cut = np.zeros(2**n)
    for i in range(n):
        for j in range(i + 1, n):
            if c[i, j] > 0:
                cut += c[i, j] * (s[:, i] != s[:, j])
    load = (s * w).sum(axis=1)
    cmax = float(np.triu(c).sum()) or 1.0
    return cut + lam * cmax * (load / w.sum()) ** 2


def local_search(cost: np.ndarray, n: int, rng: np.random.Generator, budget: int) -> float:
    """Multi-start steepest 1-flip descent; every cost lookup counts against the budget."""
    used, best = 0, float("inf")
    while used < budget:
        x = int(rng.integers(0, 2**n))
        cx = cost[x]
        used += 1
        while used < budget:
            nb = [x ^ (1 << i) for i in range(n)]
            cn = cost[nb]
            used += n
            j = int(np.argmin(cn))
            if cn[j] < cx:
                x, cx = nb[j], cn[j]
            else:
                break
        best = min(best, float(cx))
    return best


def anneal(cost: np.ndarray, n: int, rng: np.random.Generator, budget: int) -> float:
    best = float("inf")
    spread = float(cost.std()) or 1.0
    for _ in range(RESTARTS):
        x = int(rng.integers(0, 2**n))
        cx = cost[x]
        steps = budget // RESTARTS
        t0, t1 = 0.5 * spread, 0.005 * spread
        for t in range(steps):
            temp = t0 * (t1 / t0) ** (t / steps)
            y = x ^ (1 << int(rng.integers(0, n)))
            cy = cost[y]
            if cy <= cx or rng.random() < np.exp(-(cy - cx) / temp):
                x, cx = y, cy
            best = min(best, float(cx))
    return best


def run(job: tuple[str, int, int, int]) -> dict:
    fam, n, p, k = job
    c, w = make_instance(fam, n, k)
    cost = cost_diagonal(c, w)
    opt, worst = float(cost.min()), float(cost.max())
    span = max(worst - opt, 1e-12)
    is_opt = cost <= opt + TOL * max(1.0, abs(opt))
    n_opt = int(is_opt.sum())
    h = (cost - cost.mean()) / (cost.std() or 1.0)
    rng = np.random.default_rng(9_000_011 + 131 * k + 7 * n + 100_000 * p + 17 * FAMILIES.index(fam))

    t0 = time.perf_counter()
    evals, q_best, p_opts, exps = 0, float("inf"), [], []
    for _ in range(RESTARTS):
        x0 = rng.uniform(0, np.pi, 2 * p)
        count = [0]

        def f(x: np.ndarray) -> float:
            count[0] += 1
            return fq.expectation(h, n, x)

        res = minimize(f, x0, method="COBYLA", options={"maxiter": MAXITER})
        evals += count[0]
        pr = fq.probabilities(h, n, res.x)
        q_best = min(q_best, float(cost[rng.choice(2**n, size=SHOTS, p=pr)].min()))
        p_opts.append(float(pr[is_opt].sum()))
        exps.append(float(res.fun))
    t_q = time.perf_counter() - t0
    i_best = int(np.argmin(exps))

    ctrl_best, ctrl_p = float("inf"), []
    for _ in range(RESTARTS):
        pr = fq.probabilities(h, n, rng.uniform(0, np.pi, 2 * p))
        ctrl_best = min(ctrl_best, float(cost[rng.choice(2**n, size=SHOTS, p=pr)].min()))
        ctrl_p.append(float(pr[is_opt].sum()))

    uni = float(cost[rng.integers(0, 2**n, size=BUDGET)].min())
    sa = anneal(cost, n, rng, BUDGET)
    ls = local_search(cost, n, rng, BUDGET)

    def ok(v: float) -> int:
        return int(v <= opt + TOL * max(1.0, abs(opt)))

    def gap(v: float) -> float:
        return (v - opt) / span

    return {
        "family": fam, "n": n, "p": p, "instance": k, "opt_cost": opt, "num_optimal": n_opt,
        "p_uniform": n_opt / 2**n,
        "qaoa_p_opt_lowest_energy_restart": p_opts[i_best], "ctrl_p_opt_mean": float(np.mean(ctrl_p)),
        "qaoa_success": ok(q_best), "ctrl_success": ok(ctrl_best), "uniform_success": ok(uni),
        "sa_success": ok(sa), "ls_success": ok(ls),
        "qaoa_gap": gap(q_best), "ctrl_gap": gap(ctrl_best), "uniform_gap": gap(uni),
        "sa_gap": gap(sa), "ls_gap": gap(ls),
        "evaluations": evals, "qaoa_time_s": t_q,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--sizes", default="10,12,14,16")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--big-n", type=int, default=18, help="sizes >= this use --big-instances")
    ap.add_argument("--big-instances", type=int, default=12)
    ap.add_argument("--out", default="structured_study.csv")
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]
    jobs = [(f, n, a.p, k) for f in FAMILIES for n in sizes for k in range(a.big_instances if n >= a.big_n else a.instances)]
    jobs.sort(key=lambda j: -j[1])
    outdir = Path(__file__).parent / "results"
    outdir.mkdir(exist_ok=True)
    rows, writer = [], None
    fh = (outdir / (a.out + ".partial")).open("w", newline="")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(run, jobs, chunksize=1), 1):
            rows.append(r)
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(r.keys()))
                writer.writeheader()
            writer.writerow(r)
            fh.flush()
            print(f"[{i}/{len(jobs)}] {r['family']} n={r['n']} k={r['instance']} t={r['qaoa_time_s']:.0f}s", flush=True)
    fh.close()
    rows.sort(key=lambda r: (r["family"], r["n"], r["instance"]))
    with (outdir / a.out).open("w", newline="") as f2:
        wr = csv.DictWriter(f2, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("Wrote", len(rows), "rows")


if __name__ == "__main__":
    main()
