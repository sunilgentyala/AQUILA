"""Experiment 2 (revised): noise study with the random-sampling control.

For several instances per size, QAOA is optimized under depolarizing noise
(single-qubit rate p, two-qubit rate 2p). From the optimized parameters we
re-sample at several shot budgets and record:
  - p_opt: probability mass on optimal assignments (before best-sample selection)
  - success: whether best-of-shots found an optimum
  - the analytic uniform-sampling success probability at the same budget
  - expectation ratio: best <H> under noise / noiseless best <H> on the same
    instance (a Hamiltonian expectation ratio, not a state fidelity)

Run with: python evaluation/noise_controls.py [--instances 8]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from qiskit.circuit.library import QAOAAnsatz

sys.path.insert(0, str(Path(__file__).parent))
from common import write_csv  # noqa: E402
from quality_controls import TOL, make_problem, optimal_assignments  # noqa: E402

from aquila.interface.backend import AerBackend, depolarizing_noise_backend  # noqa: E402
from aquila.optimizer.classical_baselines import brute_force_optimal  # noqa: E402
from aquila.optimizer.qaoa_solver import build_cost_operator, solve_qaoa  # noqa: E402

SIZES = [6, 8, 10]
RATES = [0.0, 0.005, 0.01, 0.02, 0.05]
SHOT_BUDGETS = [64, 256, 1024, 4096]
REPS, MAXITER, RESTARTS, OPT_SHOTS = 3, 80, 2, 1024


def run_one(args: tuple[int, int, float]) -> list[dict]:
    n, k, rate = args
    problem = make_problem("real_1_20", n, 20_000 * n + k)
    _, opt_cost = brute_force_optimal(problem)
    opt_set = optimal_assignments(problem, opt_cost)
    p_uniform = len(opt_set) / 2**n
    seed = 9_100_007 + 17 * k + n

    def mk(s: int, shots: int) -> AerBackend:
        if rate == 0.0:
            return AerBackend(seed=s, shots=shots)
        return depolarizing_noise_backend(rate, seed=s, shots=shots)

    res = solve_qaoa(
        problem, mk(seed, OPT_SHOTS), reps=REPS, maxiter=MAXITER, num_restarts=RESTARTS, seed=seed
    )

    ansatz = QAOAAnsatz(cost_operator=build_cost_operator(problem), reps=REPS)
    circ = mk(seed, OPT_SHOTS).transpile(ansatz)
    circ.measure_all()
    rows = []
    for shots in SHOT_BUDGETS:
        sampler = mk(seed + shots, shots).sampler()
        pub = (circ, np.array(res.optimal_parameters), shots)
        counts = sampler.run([pub]).result()[0].data.meas.get_int_counts()
        tot = sum(counts.values())
        p_opt = sum(c for b, c in counts.items() if b in opt_set) / tot
        best = min(problem.cost_of_assignment(problem.bitstring_to_spins(b, n)) for b in counts)
        rows.append(
            {
                "n": n,
                "instance": k,
                "error_rate": rate,
                "shots": shots,
                "p_opt_qaoa": p_opt,
                "p_opt_uniform": p_uniform,
                "success_qaoa": int(best <= opt_cost + TOL * max(1.0, opt_cost)),
                "uniform_success_prob": 1 - (1 - p_uniform) ** shots,
                "best_expectation": min(res.objective_history),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=8)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    jobs = [(n, k, r) for n in SIZES for k in range(a.instances) for r in RATES]
    rows: list[dict] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(run_one, jobs, chunksize=1), 1):
            rows.extend(r)
            print(
                f"[{i}/{len(jobs)}] n={r[0]['n']} k={r[0]['instance']} "
                f"p={r[0]['error_rate']} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    base = {(r["n"], r["instance"]): r["best_expectation"] for r in rows if r["error_rate"] == 0.0}
    for r in rows:
        r["expectation_ratio"] = r["best_expectation"] / base[(r["n"], r["instance"])]
    write_csv("noise_controls.csv", list(rows[0].keys()), rows)
    print("done")


if __name__ == "__main__":
    main()
