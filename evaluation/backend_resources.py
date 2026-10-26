"""Experiment 4: backend portability and hardware resource accounting.

(a) Transpile-only accounting for the QAOA circuit (p = 3 layers, dense
    all-to-all cost Hamiltonian) on the all-to-all Aer backend and on the
    heavy-hex backend: depth, two-qubit gate count, and the number of
    pairwise interactions per cost layer.
(b) End-to-end portability check: the same optimizer code runs unchanged
    on both backends; per-instance best cost and success are compared.

Run with: python evaluation/backend_resources.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from qiskit.circuit.library import QAOAAnsatz

sys.path.insert(0, str(Path(__file__).parent))
from common import write_csv  # noqa: E402
from quality_controls import TOL, make_problem  # noqa: E402

from aquila.interface.backend import AerBackend, HeavyHexBackend  # noqa: E402
from aquila.optimizer.classical_baselines import brute_force_optimal  # noqa: E402
from aquila.optimizer.qaoa_solver import build_cost_operator, solve_qaoa  # noqa: E402

REPS = 3


def resources() -> list[dict]:
    rows = []
    for n in (4, 6, 8, 10, 12, 14):
        problem = make_problem("real_1_20", n, 10_000 * n)
        ansatz = QAOAAnsatz(cost_operator=build_cost_operator(problem), reps=REPS)
        for name, backend in (("aer_all_to_all", AerBackend(seed=1)), ("heavy_hex_d3", HeavyHexBackend(3, seed=1))):
            t = time.perf_counter()
            tc = backend.transpile(ansatz)
            ops = tc.count_ops()
            rows.append(
                {
                    "n": n,
                    "backend": name,
                    "pair_interactions_per_layer": n * (n - 1) // 2,
                    "depth": tc.depth(),
                    "two_qubit_gates": int(ops.get("cx", 0) + ops.get("rzz", 0)),
                    "total_gates": int(sum(ops.values())),
                    "transpile_s": time.perf_counter() - t,
                }
            )
            print(rows[-1], flush=True)
    return rows


def portability() -> list[dict]:
    rows = []
    for n in (4, 6):
        for k in range(3):
            problem = make_problem("real_1_20", n, 30_000 * n + k)
            _, opt = brute_force_optimal(problem)
            seed = 5_000_011 + 13 * k + n
            for name, backend in (
                ("aer_all_to_all", AerBackend(seed=seed, shots=1024)),
                ("heavy_hex_d3", HeavyHexBackend(3, seed=seed, shots=1024)),
            ):
                t = time.perf_counter()
                r = solve_qaoa(problem, backend, reps=2, maxiter=40, num_restarts=2, seed=seed)
                rows.append(
                    {
                        "n": n,
                        "instance": k,
                        "backend": name,
                        "opt_cost": opt,
                        "found_cost": r.best_cost,
                        "success": int(r.best_cost <= opt + TOL * max(1.0, opt)),
                        "depth": r.circuit_depth,
                        "two_qubit_gates": r.two_qubit_gates,
                        "evaluations": r.num_evaluations,
                        "time_s": time.perf_counter() - t,
                    }
                )
                print(rows[-1], flush=True)
    return rows


def main() -> None:
    r1 = resources()
    write_csv("backend_resources.csv", list(r1[0].keys()), r1)
    r2 = portability()
    write_csv("backend_portability.csv", list(r2[0].keys()), r2)


if __name__ == "__main__":
    main()
