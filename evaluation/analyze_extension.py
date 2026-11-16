"""Analysis of extension_study_all.csv: depth sweep, scaling to n = 20 and shot-cost accounting.

Writes analysis_ext_depth.csv, analysis_ext_cost.csv and two figures.
Run: python evaluation/analyze_extension.py
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS_DIR, write_csv  # noqa: E402

import os
SUF = "_norm" if os.environ.get("AQUILA_NORM") == "1" else ""
rows = list(csv.DictReader((RESULTS_DIR / f"extension_study_all{SUF}.csv").open()))
for r in rows:
    for k in r:
        r[k] = float(r[k])
    r["n"], r["p"], r["instance"] = int(r["n"]), int(r["p"]), int(r["instance"])
rng = np.random.default_rng(2026)
SHOT_PER_EVAL = 1024
TARGET = 0.90


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def boot_ratio(num: np.ndarray, den: np.ndarray, reps: int = 4000) -> tuple[float, float, float]:
    idx = rng.integers(0, len(num), size=(reps, len(num)))
    v = num[idx].sum(1) / den[idx].sum(1)
    return float(num.sum() / den.sum()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    x, y = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    if x + y == 0:
        return x, y, 1.0
    return x, y, float(binomtest(x, x + y, 0.5).pvalue)


def samples_for(p: float, target: float = TARGET) -> float:
    if p <= 0:
        return float("inf")
    if p >= 1:
        return 1.0
    return math.log(1 - target) / math.log(1 - p)


depth_rows, cost_rows = [], []
keys = sorted({(r["n"], r["p"]) for r in rows})
for n, p in keys:
    g = sorted((r for r in rows if r["n"] == n and r["p"] == p), key=lambda r: r["instance"])
    N = len(g)
    q = np.array([r["qaoa_success"] for r in g])
    u = np.array([r["uniform_success"] for r in g])
    c = np.array([r["ctrl_success"] for r in g])
    s = np.array([r["sa_success"] for r in g])
    pq = np.array([r["qaoa_p_opt_lowest_energy_restart"] for r in g])
    pc = np.array([r["ctrl_p_opt_mean"] for r in g])
    pu = np.array([r["p_uniform"] for r in g])
    e, elo, ehi = boot_ratio(pq, pu)
    ec, eclo, echi = boot_ratio(pc, pu)
    qu = mcnemar(q, u)
    qs = mcnemar(q, s)
    depth_rows.append({
        "n": n, "p": p, "N": N,
        "qaoa_k": int(q.sum()), "qaoa_ci_lo": wilson(int(q.sum()), N)[0], "qaoa_ci_hi": wilson(int(q.sum()), N)[1],
        "ctrl_k": int(c.sum()), "uniform_k": int(u.sum()), "sa_k": int(s.sum()),
        "enrichment": e, "enr_lo": elo, "enr_hi": ehi,
        "ctrl_enrichment": ec, "ctrl_enr_lo": eclo, "ctrl_enr_hi": echi,
        "mcnemar_q_only_vs_uniform": qu[0], "mcnemar_u_only": qu[1], "mcnemar_p_vs_uniform": qu[2],
        "mcnemar_q_only_vs_sa": qs[0], "mcnemar_sa_only": qs[1], "mcnemar_p_vs_sa": qs[2],
        "mean_evaluations": float(np.mean([r["evaluations"] for r in g])),
        "mean_qaoa_time_s": float(np.mean([r["qaoa_time_s"] for r in g])),
        "mean_exact_time_s": float(np.mean([r["exact_time_s"] for r in g])),
    })
    # shot-cost accounting (median over instances)
    n_u = np.array([samples_for(x) for x in pu])
    n_q = np.array([samples_for(x) for x in pq])
    loop = np.array([r["evaluations"] * SHOT_PER_EVAL for r in g])
    total = loop + n_q
    cost_rows.append({
        "n": n, "p": p, "N": N,
        "median_uniform_samples_90": float(np.median(n_u)),
        "median_qaoa_samples_90": float(np.median(n_q)),
        "median_loop_shots": float(np.median(loop)),
        "median_qaoa_total_shots_90": float(np.median(total)),
        "frac_sampling_only_cheaper": float((n_q < n_u).mean()),
        "frac_total_cheaper": float((total < n_u).mean()),
        "median_total_over_uniform": float(np.median(total / n_u)),
    })
write_csv(f"analysis_ext_depth{SUF}.csv", list(depth_rows[0].keys()), depth_rows)
write_csv(f"analysis_ext_cost{SUF}.csv", list(cost_rows[0].keys()), cost_rows)

# Figure 1: enrichment against n for each depth (with random-parameter control at p = 3).
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
for p in (1, 2, 3, 4, 6):
    pts = [d for d in depth_rows if d["p"] == p]
    if pts:
        ax[0].errorbar([d["n"] for d in pts], [d["enrichment"] for d in pts],
                       yerr=[[d["enrichment"] - d["enr_lo"] for d in pts], [d["enr_hi"] - d["enrichment"] for d in pts]],
                       marker="o", ms=3, lw=1, capsize=2, label=f"p = {p}")
ctrl = [d for d in depth_rows if d["p"] == 3]
ax[0].plot([d["n"] for d in ctrl], [d["ctrl_enrichment"] for d in ctrl], "k--", marker="s", ms=3, lw=1, label="random, p = 3")
ax[0].axhline(1, color="grey", lw=0.6)
ax[0].set_xlabel("tasks n"); ax[0].set_ylabel("enrichment over uniform"); ax[0].legend(fontsize=6, ncol=2)
ax[0].set_title("(a) probability enrichment", fontsize=8)
p3 = [d for d in depth_rows if d["p"] == 3]
for lab, key, mk in (("QAOA p = 3", "qaoa_k", "o"), ("random params", "ctrl_k", "s"), ("uniform", "uniform_k", "^"), ("SA tuned", "sa_k", "v")):
    ax[1].plot([d["n"] for d in p3], [d[key] / d["N"] for d in p3], marker=mk, ms=3, lw=1, label=lab)
ax[1].set_xlabel("tasks n"); ax[1].set_ylabel("exact-optimum success"); ax[1].legend(fontsize=6)
ax[1].set_title("(b) success at equal budget (3,072 samples)", fontsize=8)
for a_ in ax:
    a_.tick_params(labelsize=7)
    a_.xaxis.label.set_size(8); a_.yaxis.label.set_size(8)
fig.tight_layout()
fig.savefig(RESULTS_DIR / f"fig_ext_depth_scale{SUF}.png", dpi=300)

# Figure 2: median samples to reach 90% success.
fig, ax = plt.subplots(figsize=(3.4, 2.7))
p3c = [d for d in cost_rows if d["p"] == 3]
ns = [d["n"] for d in p3c]
ax.semilogy(ns, [d["median_uniform_samples_90"] for d in p3c], "^-", ms=3, lw=1, label="uniform sampling")
ax.semilogy(ns, [d["median_qaoa_samples_90"] for d in p3c], "o-", ms=3, lw=1, label="QAOA, sampling only")
ax.semilogy(ns, [d["median_qaoa_total_shots_90"] for d in p3c], "s-", ms=3, lw=1, label="QAOA incl. variational loop")
ax.set_xlabel("tasks n", fontsize=8); ax.set_ylabel("median samples/shots to 90% success", fontsize=7)
ax.tick_params(labelsize=7); ax.legend(fontsize=6)
fig.tight_layout()
fig.savefig(RESULTS_DIR / f"fig_ext_cost{SUF}.png", dpi=300)
print("done", len(depth_rows), "depth rows")
for d in depth_rows:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k in ("n", "p", "qaoa_k", "ctrl_k", "uniform_k", "sa_k", "enrichment", "ctrl_enrichment", "mcnemar_p_vs_uniform")})
for d in cost_rows:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()})
