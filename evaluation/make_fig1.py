"""Generate Figure 1: AQUILA architecture diagram (interface/optimizer/security/backend).

Run with: python evaluation/make_fig1.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"

COLOR_QAOA = "#2a78d6"
COLOR_SA = "#1baf7a"
INK = "#0b0b0b"


def box(ax, x, y, w, h, text, color, fontsize=7.2, textcolor="white"):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.1, edgecolor=INK, facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=textcolor)


def arrow(ax, x1, y1, x2, y2, label=None, label_dy=0.15):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, linewidth=1.2, color=INK
    )
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2 + label_dy, label, fontsize=6, color="#52514e")


def main() -> None:
    fig, ax = plt.subplots(figsize=(3.5, 4.1), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0.8, 13.3)
    ax.axis("off")

    box(ax, 0.3, 11.6, 4.2, 1.1, "Requester Agent\n(AllocationAgent)", "#898781")
    box(ax, 5.5, 11.6, 4.2, 1.1, "Solver Agent\n(AllocationAgent)", "#898781")
    arrow(ax, 4.5, 12.15, 5.5, 12.15)
    ax.text(5.0, 12.95, "PQC-secured envelope", fontsize=6, ha="center", color="#52514e")

    box(
        ax, 1.6, 9.5, 6.8, 1.2,
        "Security layer: ML-KEM-768 encapsulation +\nAES-256-GCM + ML-DSA-65 signature",
        "#4a3aa7", fontsize=6.8,
    )
    arrow(ax, 2.4, 11.6, 2.4, 10.7)
    arrow(ax, 7.6, 11.6, 7.6, 10.7)

    box(
        ax, 1.6, 7.3, 6.8, 1.4,
        "Interface layer\nTaskAllocationProblem -> Ising Hamiltonian\n(EstimatorV2 / SamplerV2 protocol)",
        COLOR_QAOA, fontsize=6.8,
    )
    arrow(ax, 5.0, 9.5, 5.0, 8.7)

    box(
        ax, 1.6, 5.1, 6.8, 1.6,
        "Optimizer layer\nMulti-restart QAOA solver\n(vs. greedy / simulated-annealing baselines)",
        COLOR_QAOA, fontsize=6.8,
    )
    arrow(ax, 5.0, 7.3, 5.0, 6.7)

    box(
        ax, 1.6, 3.0, 6.8, 1.6,
        "Backend (pluggable)\nAer simulator today; real QPU later,\nsame primitive API",
        COLOR_SA, fontsize=6.8, textcolor="black",
    )
    arrow(ax, 5.0, 5.1, 5.0, 4.6)

    ax.text(
        5.0, 1.6,
        "Nothing above the backend boundary refers\nto the simulator directly.",
        fontsize=6.3, ha="center", color="#52514e", style="italic",
    )

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig1_architecture.png", bbox_inches="tight")
    print(f"Wrote {RESULTS_DIR / 'fig1_architecture.png'}")


if __name__ == "__main__":
    main()
