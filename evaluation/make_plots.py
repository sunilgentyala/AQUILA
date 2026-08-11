"""Generate the paper's evaluation figures from the CSVs in evaluation/results/.

Sized for single-column IEEE placement (~3.5in wide), colorblind-validated
3-series palette (blue/orange/aqua), with markers/linestyles as a secondary
encoding so figures stay legible in grayscale print.

Run with: python evaluation/make_plots.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"

COLOR_QAOA = "#2a78d6"  # categorical slot 1 (blue)
COLOR_GREEDY = "#eb6834"  # categorical slot 2 (orange)
COLOR_SA = "#1baf7a"  # categorical slot 3 (aqua)
COLOR_CLASSICAL = "#eb6834"

FIGSIZE = (3.5, 2.6)
DPI = 300


def read_csv(name: str) -> list[dict]:
    with (RESULTS_DIR / name).open() as f:
        return list(csv.DictReader(f))


def group_mean(rows: list[dict], key: str, value: str) -> tuple[list[float], list[float]]:
    groups: dict[float, list[float]] = {}
    for row in rows:
        k = float(row[key])
        groups.setdefault(k, []).append(float(row[value]))
    xs = sorted(groups)
    ys = [sum(groups[x]) / len(groups[x]) for x in xs]
    return xs, ys


def plot_solution_quality() -> None:
    rows = read_csv("solution_quality.csv")
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    for value_key, label, color, marker in [
        ("qaoa_ratio", "AQUILA (QAOA)", COLOR_QAOA, "o"),
        ("sa_ratio", "Simulated annealing", COLOR_SA, "^"),
        ("greedy_ratio", "Greedy (LPT)", COLOR_GREEDY, "s"),
    ]:
        xs, ys = group_mean(rows, "num_tasks", value_key)
        ax.plot(xs, ys, color=color, marker=marker, markersize=5, linewidth=2, label=label)

    ax.set_xlabel("Number of tasks")
    ax.set_ylabel("Mean approximation ratio")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.grid(True, color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "solution_quality.png")
    plt.close(fig)


def plot_runtime_scaling() -> None:
    rows = read_csv("solution_quality.csv")
    xs, ys = group_mean(rows, "num_tasks", "qaoa_time_sec")
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(xs, ys, color=COLOR_QAOA, marker="o", markersize=5, linewidth=2)
    ax.set_xlabel("Number of tasks")
    ax.set_ylabel("Mean QAOA solve time (s)")
    ax.set_yscale("log")
    ax.grid(True, which="both", color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "runtime_scaling.png")
    plt.close(fig)


def plot_security_overhead() -> None:
    rows = read_csv("security_overhead.csv")
    pqc = next(r for r in rows if "PQC" in r["scheme"])
    classical = next(r for r in rows if "classical" in r["scheme"])

    metrics = ["kem_keygen_ms", "kem_encap_ms", "kem_decap_ms", "sign_ms", "verify_ms"]
    labels = ["Keygen", "Encap/\nEncrypt", "Decap/\nDecrypt", "Sign", "Verify"]

    import numpy as np

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.bar(
        x - width / 2, [float(pqc[m]) for m in metrics], width,
        color=COLOR_QAOA, label="ML-KEM-768 + ML-DSA-65",
    )
    ax.bar(
        x + width / 2, [float(classical[m]) for m in metrics], width,
        color=COLOR_CLASSICAL, label="RSA-3072 + ECDSA-P256",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Mean time (ms, log scale)")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.grid(True, axis="y", which="both", color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "security_overhead.png")
    plt.close(fig)


def plot_noise_degradation() -> None:
    rows = read_csv("simulation_methodology.csv")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.2), dpi=DPI, sharex=True)

    xs, ys = group_mean(rows, "error_rate", "final_expectation")
    ax1.plot(xs, ys, color=COLOR_QAOA, marker="o", markersize=5, linewidth=2)
    ax1.set_ylabel("Raw expectation\n(circuit fidelity)", fontsize=8)
    ax1.grid(True, color="#e1e0d9", linewidth=0.6)
    ax1.set_axisbelow(True)

    xs2, ys2 = group_mean(rows, "error_rate", "approx_ratio")
    ax2.plot(xs2, ys2, color=COLOR_SA, marker="^", markersize=5, linewidth=2)
    ax2.set_ylabel("Post-selected\napprox. ratio", fontsize=8)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("Depolarizing error rate")
    ax2.grid(True, color="#e1e0d9", linewidth=0.6)
    ax2.set_axisbelow(True)

    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "noise_degradation.png")
    plt.close(fig)


def main() -> None:
    plot_solution_quality()
    plot_runtime_scaling()
    plot_security_overhead()
    plot_noise_degradation()
    print(f"Wrote 4 figures to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
