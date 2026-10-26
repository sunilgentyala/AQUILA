"""Draw the AQUILA architecture diagram and an animated request walkthrough.

Writes:
  docs/assets/architecture.png   static diagram for the README and site
  docs/assets/aquila_flow.gif    animation of one request/response cycle

Run with: python evaluation/make_architecture.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets"

INK = "#1a1a1a"
MUTED = "#5b5a56"
DIM = "#d9d8d3"
SECURITY = "#4a3aa7"
INTERFACE = "#2a78d6"
OPTIMIZER = "#1f8a70"
BACKEND = "#eb6834"
AGENT = "#6b6a66"

# (key, label lines, x, color)
LAYERS = [
    ("security", ["Security layer", "verify signature,", "recipient, role,", "freshness, replay;", "then decrypt"], 4, SECURITY),
    ("interface", ["Interface layer", "TaskAllocationProblem", "to Ising Hamiltonian", "Estimator / Sampler", "primitives"], 28, INTERFACE),
    ("optimizer", ["Optimizer layer", "multi-restart QAOA", "best-of-samples", "controls: uniform,", "random-parameter QAOA"], 52, OPTIMIZER),
    ("backend", ["Backend (pluggable)", "owns transpilation", "", "", ""], 76, BACKEND),
]
BOX_W, BOX_Y, BOX_H = 20, 9, 22

STEPS = [
    ("1  Requester seals the problem", ["requester"], "ML-KEM-768 encapsulation, AES-256-GCM, ML-DSA-65 signature; header binds sender, recipient, role, sequence, time"),
    ("2  Solver checks the envelope", ["security"], "Signature, recipient, role, freshness window and replay counter are checked before any decryption"),
    ("3  Problem becomes a Hamiltonian", ["interface"], "Two-cluster number partitioning as pairwise ZZ terms, one qubit per task"),
    ("4  QAOA optimizes and samples", ["optimizer"], "COBYLA outer loop, three restarts, samples after every restart, best sample kept"),
    ("5  Backend executes", ["backend"], "All-to-all Aer simulator or heavy-hex lattice with a {rz, sx, x, cx} basis; same primitives either way"),
    ("6  Result is sealed and returned", ["security", "requester"], "Response is bound to its own role and sequence number, so it cannot be replayed as a request"),
]


def rbox(ax, x, y, w, h, color, alpha=1.0, lw=1.2, ec=INK):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.2",
            linewidth=lw, edgecolor=ec, facecolor=color, alpha=alpha,
        )
    )


def arrow(ax, p1, p2, color=INK, lw=1.5, style="-|>", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p1, p2, arrowstyle=style, mutation_scale=12, linewidth=lw, color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def draw(ax, active: set[str] | None = None, caption: tuple[str, str] | None = None):
    ax.clear()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    def on(key: str) -> bool:
        return active is None or key in active

    # solver container
    rbox(ax, 1.5, 5.5, 97, 31, "#f6f5f1", lw=1.0, ec=MUTED)
    ax.text(96.8, 34.4, "Solver agent", fontsize=10, color=MUTED, fontweight="bold",
            va="center", ha="right")

    # requester
    req_on = on("requester")
    rbox(ax, 3, 43.5, 26, 11, AGENT if req_on else DIM)
    ax.text(16, 49, "Requester agent\nAllocationAgent", ha="center", va="center",
            fontsize=10, color="white" if req_on else MUTED, fontweight="bold")

    # layers
    for key, lines, x, color in LAYERS:
        is_on = on(key)
        rbox(ax, x, BOX_Y, BOX_W, BOX_H, color if is_on else DIM)
        tc = "white" if is_on else MUTED
        ax.text(x + BOX_W / 2, BOX_Y + BOX_H - 2.6, lines[0], ha="center", va="center",
                fontsize=9.5, color=tc, fontweight="bold")
        for i, line in enumerate(lines[1:]):
            ax.text(x + BOX_W / 2, BOX_Y + BOX_H - 6.2 - i * 3.0, line, ha="center",
                    va="center", fontsize=8.2, color=tc)
        if key != "backend":
            arrow(ax, (x + BOX_W + 0.3, BOX_Y + BOX_H / 2), (x + 24 - 0.3, BOX_Y + BOX_H / 2),
                  color=INK if is_on else DIM)

    # two backends inside the backend box
    bx = 76
    bk_on = on("backend")
    for j, name in enumerate(["Aer, all-to-all", "Heavy-hex, 19 qubits"]):
        yy = BOX_Y + 2.0 + j * 6.2
        rbox(ax, bx + 1.6, yy, BOX_W - 3.2, 5.2, "#fbd9c8" if bk_on else "#ecebe7", lw=0.9)
        ax.text(bx + BOX_W / 2, yy + 2.6, name, ha="center", va="center", fontsize=8.2,
                color=INK if bk_on else MUTED)

    # agent <-> security arrows
    send_on = on("requester") or on("security")
    arrow(ax, (12, 43.5), (12, BOX_Y + BOX_H + 0.3), color=INK if send_on else DIM, lw=2)
    arrow(ax, (21, BOX_Y + BOX_H + 0.3), (21, 43.5), color=INK if send_on else DIM, lw=2)
    ax.text(10.6, 38.6, "request", fontsize=7.5, color=MUTED, ha="right")
    ax.text(22.4, 38.6, "response", fontsize=7.5, color=MUTED, ha="left")
    ax.text(40, 49, "Post-quantum-secured envelope in both directions", fontsize=9.5, color=SECURITY,
            va="center", fontweight="bold")
    ax.text(40, 45.6, "ML-KEM-768 + AES-256-GCM + ML-DSA-65", fontsize=8.4, color=MUTED, va="center")
    ax.text(40, 42.6, "header: sender, recipient, role, sequence, timestamp", fontsize=8.4,
            color=MUTED, va="center")

    ax.text(50, 3.4, "Nothing above the backend boundary refers to a simulator directly.",
            ha="center", fontsize=8.6, color=MUTED, style="italic")

    if caption:
        title, detail = caption
        ax.text(50, 58.3, title, ha="center", va="center", fontsize=11.5, color=INK,
                fontweight="bold")
        ax.text(50, 56.0, detail, ha="center", va="center", fontsize=8.3, color=MUTED)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=110)
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)

    draw(ax, None, ("AQUILA architecture", "One request-response cycle through four layers and a pluggable backend"))
    fig.savefig(OUT / "architecture.png", dpi=160, facecolor="white")
    print(f"Wrote {OUT / 'architecture.png'}")

    frames = [(s[1], (s[0], s[2])) for s in STEPS]

    def update(i: int):
        active, cap = frames[i]
        draw(ax, set(active), cap)
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(OUT / "aquila_flow.gif", writer=PillowWriter(fps=0.55), savefig_kwargs={"facecolor": "white"})
    print(f"Wrote {OUT / 'aquila_flow.gif'}")


if __name__ == "__main__":
    main()
