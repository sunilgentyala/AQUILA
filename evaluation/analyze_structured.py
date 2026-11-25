"""Analyze results/structured_study.csv: success, gap, enrichment, exact McNemar tests.

Writes results/analysis_structured.csv and prints a summary.
Run: python evaluation/analyze_structured.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

RES = Path(__file__).parent / "results"
METHODS = ("qaoa", "ctrl", "uniform", "sa", "ls")


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def boot(v: np.ndarray, rng: np.random.Generator, f=np.mean, B: int = 4000) -> tuple[float, float]:
    idx = rng.integers(0, len(v), size=(B, len(v)))
    s = np.array([f(v[i]) for i in idx])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main() -> None:
    rows = list(csv.DictReader((RES / "structured_study.csv").open()))
    g = defaultdict(list)
    for r in rows:
        g[(r["family"], int(r["n"]))].append(r)
    rng = np.random.default_rng(2026)
    out = []
    for (fam, n), v in sorted(g.items()):
        N = len(v)
        rec = {"family": fam, "n": n, "N": N, "median_num_optimal": float(np.median([int(x["num_optimal"]) for x in v]))}
        for m in METHODS:
            s = np.array([int(x[m + "_success"]) for x in v])
            gp = np.array([float(x[m + "_gap"]) for x in v])
            rec[f"{m}_k"] = int(s.sum())
            rec[f"{m}_gap_mean"] = float(gp.mean())
            lo, hi = boot(gp, rng)
            rec[f"{m}_gap_lo"], rec[f"{m}_gap_hi"] = lo, hi
        enr = np.array([float(x["qaoa_p_opt_lowest_energy_restart"]) / float(x["p_uniform"]) for x in v])
        cen = np.array([float(x["ctrl_p_opt_mean"]) / float(x["p_uniform"]) for x in v])
        rec["enrichment"] = float(np.exp(np.mean(np.log(enr))))
        rec["enr_lo"], rec["enr_hi"] = (float(np.exp(a)) for a in boot(np.log(enr), rng))
        rec["ctrl_enrichment"] = float(np.exp(np.mean(np.log(np.maximum(cen, 1e-12)))))
        for m in ("uniform", "sa", "ls", "ctrl"):
            a = np.array([int(x["qaoa_success"]) for x in v])
            b = np.array([int(x[m + "_success"]) for x in v])
            rec[f"p_qaoa_vs_{m}"] = mcnemar_exact(int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum()))
        out.append(rec)
    with (RES / "analysis_structured.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(r["family"], r["n"], r["N"], "succ q/ctrl/uni/sa/ls", *(r[f"{m}_k"] for m in METHODS),
              "gap q %.4f uni %.4f sa %.4f ls %.4f" % (r["qaoa_gap_mean"], r["uniform_gap_mean"], r["sa_gap_mean"], r["ls_gap_mean"]),
              "enr %.1f p(uni) %.3f p(ls) %.3f" % (r["enrichment"], r["p_qaoa_vs_uniform"], r["p_qaoa_vs_ls"]))


if __name__ == "__main__":
    main()
