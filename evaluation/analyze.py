"""Statistics and figures for the revised evaluation.

Reads evaluation/results/*.csv and writes:
  - analysis_quality.csv        success rates with Wilson 95% CIs and paired tests
  - analysis_pdelta.csv         optimal-probability enrichment with bootstrap CIs
  - analysis_noise.csv          noise study aggregates (if noise_controls.csv exists)
  - figures: fig_quality.png, fig_enrichment.png, fig_runtime.png, fig_noise.png,
             fig_security.png, fig_resources.png

Run with: python evaluation/analyze.py
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).parent / "results"
rng = np.random.default_rng(12345)

# colorblind-safe palette with a secondary marker encoding
STYLE = {
    "qaoa": ("QAOA (optimized)", "#2a78d6", "o", "-"),
    "qaoa_unopt": ("QAOA (random params)", "#7a5cc7", "s", "--"),
    "uniform": ("Uniform sampling", "#1baf7a", "^", "-"),
    "sa_tuned": ("SA (tuned, matched budget)", "#eb6834", "D", "-"),
    "kk": ("Karmarkar-Karp", "#a0a0a0", "v", "-"),
    "greedy": ("Greedy LPT", "#555555", "x", ":"),
}
FAMILY_TITLE = {"real_1_20": "Real-valued weights in [1, 20]", "int_1_100": "Integer weights in [1, 100]"}


def load(name: str) -> list[dict]:
    with (R / name).open() as f:
        return list(csv.DictReader(f))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def boot_ci(x: np.ndarray, fn=np.mean, b: int = 4000) -> tuple[float, float]:
    idx = rng.integers(0, len(x), (b, len(x)))
    vals = fn(x[idx], axis=1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def style_axes(ax):
    ax.grid(alpha=0.25, linewidth=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def quality() -> None:
    rows = load("quality_controls.csv")
    methods = ["qaoa", "qaoa_unopt", "uniform", "sa_tuned", "kk", "greedy"]
    sizes = sorted({int(r["n"]) for r in rows})
    out = []
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, fam in zip(axes, FAMILY_TITLE):
        for m in methods:
            label, color, marker, ls = STYLE[m]
            ys, lo, hi = [], [], []
            for n in sizes:
                g = [r for r in rows if r["family"] == fam and int(r["n"]) == n]
                k = sum(int(r[f"{m}_success"]) for r in g)
                a, b = wilson(k, len(g))
                ys.append(k / len(g))
                lo.append(a)
                hi.append(b)
                out.append({"family": fam, "n": n, "method": m, "successes": k, "N": len(g),
                            "rate": k / len(g), "ci_lo": a, "ci_hi": b})
            ax.errorbar(sizes, ys, yerr=[np.array(ys) - lo, np.array(hi) - ys], label=label,
                        color=color, marker=marker, linestyle=ls, capsize=2, linewidth=1.4, markersize=5)
        ax.set_title(FAMILY_TITLE[fam], fontsize=10)
        ax.set_xlabel("Number of tasks n")
        ax.set_xticks(sizes)
        style_axes(ax)
    axes[0].set_ylabel("Exact-optimum success rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(R / "fig_quality.png", dpi=200)
    plt.close(fig)

    # paired tests: QAOA vs uniform, QAOA vs unoptimized (exact McNemar, two-sided)
    paired = []
    for fam in FAMILY_TITLE:
        for n in sizes:
            g = [r for r in rows if r["family"] == fam and int(r["n"]) == n]
            for other in ("uniform", "qaoa_unopt", "sa_tuned", "kk"):
                b = sum(1 for r in g if int(r["qaoa_success"]) and not int(r[f"{other}_success"]))
                c = sum(1 for r in g if not int(r["qaoa_success"]) and int(r[f"{other}_success"]))
                p = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
                paired.append({"family": fam, "n": n, "vs": other, "qaoa_only": b,
                               "other_only": c, "mcnemar_exact_p": p})
    with (R / "analysis_quality.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    with (R / "analysis_paired_tests.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0].keys()))
        w.writeheader()
        w.writerows(paired)

    # enrichment: optimal-probability mass relative to uniform per-shot probability
    enr = []
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for ax, fam in zip(axes, FAMILY_TITLE):
        for m, key in (("qaoa", "qaoa_p_opt_best_restart"), ("qaoa_unopt", "qaoa_unopt_p_opt")):
            label, color, marker, ls = STYLE[m]
            ys, lo, hi = [], [], []
            for n in sizes:
                g = [r for r in rows if r["family"] == fam and int(r["n"]) == n]
                num = np.array([float(r[key]) for r in g])
                den = np.array([float(r["uniform_p_opt_per_shot"]) for r in g])
                pooled = float(num.sum() / den.sum())  # pooled ratio; robust to zero-mass instances
                idx = rng.integers(0, len(g), (4000, len(g)))
                bs = num[idx].sum(axis=1) / den[idx].sum(axis=1)
                a, b = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
                ys.append(pooled)
                lo.append(a)
                hi.append(b)
                enr.append({"family": fam, "n": n, "method": m, "pooled_enrichment": pooled,
                            "ci_lo": a, "ci_hi": b})
            ax.errorbar(sizes, ys, yerr=[np.array(ys) - lo, np.array(hi) - ys], label=label,
                        color=color, marker=marker, linestyle=ls, capsize=2, linewidth=1.4)
        ax.axhline(1.0, color="#555555", linewidth=0.8, linestyle=":")
        ax.set_title(FAMILY_TITLE[fam], fontsize=10)
        ax.set_xlabel("Number of tasks n")
        ax.set_xticks(sizes)
        style_axes(ax)
    axes[0].set_ylabel("P(optimal) / uniform P(optimal)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(R / "fig_enrichment.png", dpi=200)
    plt.close(fig)
    with (R / "analysis_enrichment.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(enr[0].keys()))
        w.writeheader()
        w.writerows(enr)

    # runtime
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    series = (("qaoa_time_s", "QAOA (3 restarts)", "#2a78d6", "o"),
              ("exact_time_s", "Exact enumeration", "#1baf7a", "^"),
              ("dp_time_s", "Exact DP (subset sum)", "#eb6834", "D"),
              ("uniform_time_s", "Uniform sampling", "#7a5cc7", "s"),
              ("kk_time_s", "Karmarkar-Karp", "#a0a0a0", "v"))
    for key, label, color, marker in series:
        ys = [float(np.mean([float(r[key]) for r in rows if int(r["n"]) == n])) for n in sizes]
        ax.plot(sizes, ys, label=label, color=color, marker=marker, linewidth=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("Number of tasks n")
    ax.set_ylabel("Mean wall-clock time (s)")
    ax.set_xticks(sizes)
    ax.legend(fontsize=7.5, frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(R / "fig_runtime.png", dpi=200)
    plt.close(fig)

    # normalized imbalance, instance-level: what the original ratio hid
    lines = []
    for fam in FAMILY_TITLE:
        for n in sizes:
            g = [r for r in rows if r["family"] == fam and int(r["n"]) == n]
            for m in methods:
                v = np.array([float(r[f"{m}_norm_imb"]) for r in g])
                lines.append({"family": fam, "n": n, "method": m, "median_norm_imbalance": float(np.median(v)),
                              "p90_norm_imbalance": float(np.percentile(v, 90)),
                              "mean_norm_imbalance": float(v.mean())})
    with (R / "analysis_imbalance.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lines[0].keys()))
        w.writeheader()
        w.writerows(lines)


def noise() -> None:
    p = R / "noise_controls.csv"
    if not p.exists():
        print("noise_controls.csv not found; skipping noise analysis")
        return
    rows = load("noise_controls.csv")
    sizes = sorted({int(r["n"]) for r in rows})
    rates = sorted({float(r["error_rate"]) for r in rows})
    out = []
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for n, color, marker in zip(sizes, ("#2a78d6", "#eb6834", "#1baf7a"), ("o", "s", "^")):
        ys, lo, hi, ey = [], [], [], []
        for rt in rates:
            g = [r for r in rows if int(r["n"]) == n and float(r["error_rate"]) == rt and int(r["shots"]) == 1024]
            v = np.array([float(r["p_opt_qaoa"]) / float(r["p_opt_uniform"]) for r in g])
            ys.append(float(v.mean()))
            a, b = boot_ci(v)
            lo.append(a)
            hi.append(b)
            ey.append(float(np.mean([float(r["expectation_ratio"]) for r in g])))
            out.append({"n": n, "error_rate": rt, "shots": 1024, "enrichment_mean": v.mean(),
                        "ci_lo": a, "ci_hi": b, "expectation_ratio": ey[-1], "N": len(g)})
        xs = [100 * r for r in rates]
        axes[0].plot(xs, ey, marker=marker, color=color, label=f"n={n}")
        axes[1].errorbar(xs, ys, yerr=[np.array(ys) - lo, np.array(hi) - ys], marker=marker,
                         color=color, capsize=2, label=f"n={n}")
    axes[0].set_ylabel("Expectation ratio  <H>noisy / <H>ideal")
    axes[1].set_ylabel("P(optimal) / uniform P(optimal)")
    axes[1].axhline(1.0, color="#555555", linewidth=0.8, linestyle=":")
    for ax in axes[:2]:
        ax.set_xlabel("Single-qubit depolarizing error (%)")
        ax.legend(fontsize=8, frameon=False)
        style_axes(ax)
    # best-of-shots success vs shot budget, QAOA at 5% noise vs analytic uniform
    shots = sorted({int(r["shots"]) for r in rows})
    for n, color, marker in zip(sizes, ("#2a78d6", "#eb6834", "#1baf7a"), ("o", "s", "^")):
        q = [np.mean([int(r["success_qaoa"]) for r in rows if int(r["n"]) == n and float(r["error_rate"]) == rates[-1]
                      and int(r["shots"]) == s]) for s in shots]
        u = [np.mean([float(r["uniform_success_prob"]) for r in rows if int(r["n"]) == n and float(r["error_rate"]) == rates[-1]
                      and int(r["shots"]) == s]) for s in shots]
        axes[2].plot(shots, q, marker=marker, color=color, label=f"QAOA n={n}")
        axes[2].plot(shots, u, color=color, linestyle="--", alpha=0.8)
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("Shot budget")
    axes[2].set_ylabel(f"Success at {100 * rates[-1]:.0f}% error (dashed: uniform)")
    axes[2].legend(fontsize=8, frameon=False)
    style_axes(axes[2])
    fig.tight_layout()
    fig.savefig(R / "fig_noise.png", dpi=200)
    plt.close(fig)
    if out:
        with (R / "analysis_noise.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)


def security() -> None:
    rows = load("security_overhead_v2.csv")
    pick = [("ML-KEM-768 + ML-DSA-65 envelope", "PQC envelope", "cat3"),
            ("ECDH+ECDSA P-256", "ECDH+ECDSA P-256", "128-bit"),
            ("RSA-3072 OAEP+PSS", "RSA-3072 (OAEP+PSS)", "128-bit"),
            ("ECDH+ECDSA P-384", "ECDH+ECDSA P-384", "192-bit")]

    def val(scheme, level, metric):
        for r in rows:
            if r["scheme"] == scheme and r["security_level"] == level and r["metric"] == metric:
                return float(r["mean"]), float(r["stdev"])
        return None

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    colors = ["#2a78d6", "#1baf7a", "#eb6834", "#7a5cc7"]
    xs = np.arange(2)
    w = 0.2
    for i, (label, scheme, level) in enumerate(pick):
        vals = [val(scheme, level, "seal"), val(scheme, level, "open")]
        axes[0].bar(xs + (i - 1.5) * w, [v[0] for v in vals], w, yerr=[v[1] for v in vals],
                    color=colors[i], label=label, capsize=2)
        wire = val(scheme, level, "wire_size_512B_payload")
        axes[1].bar(i, wire[0], 0.6, color=colors[i])
    axes[0].set_xticks(xs, ["seal", "open"])
    axes[0].set_ylabel("Per-message time (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].set_xticks(range(4), ["PQC\ncat. 3", "P-256\n128-bit", "RSA-3072\n128-bit", "P-384\n192-bit"], fontsize=8)
    axes[1].set_ylabel("Wire size, 512-byte payload (bytes)")
    for ax in axes:
        style_axes(ax)
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(R / "fig_security.png", dpi=200)
    plt.close(fig)


def resources() -> None:
    p = R / "backend_resources.csv"
    if not p.exists():
        return
    rows = load("backend_resources.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
    for name, color, marker, label in (("aer_all_to_all", "#2a78d6", "o", "All-to-all"),
                                       ("heavy_hex_d3", "#eb6834", "s", "Heavy-hex (19 qubits)")):
        g = [r for r in rows if r["backend"] == name]
        ns = [int(r["n"]) for r in g]
        axes[0].plot(ns, [int(r["depth"]) for r in g], color=color, marker=marker, label=label)
        axes[1].plot(ns, [int(r["two_qubit_gates"]) for r in g], color=color, marker=marker, label=label)
    axes[0].set_ylabel("Transpiled depth (p = 3)")
    axes[1].set_ylabel("Two-qubit gates (p = 3)")
    for ax in axes:
        ax.set_xlabel("Number of tasks n")
        style_axes(ax)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(R / "fig_resources.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    quality()
    noise()
    security()
    resources()
    print("analysis complete")
