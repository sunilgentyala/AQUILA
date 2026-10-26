"""Experiment 3: quantum-simulation methodology - noise-model degradation study.

Solves the same set of problem instances under an ideal (noiseless) Aer
backend and under Aer backends carrying depolarizing noise models at
increasing error rates, to characterize how QAOA solution quality degrades
as the simulated hardware becomes noisier. This is the paper's quantum-
simulation-methodology contribution: using the noise model as a controlled
proxy for NISQ-era hardware behavior, in the absence of real QPU access.

Run with: python evaluation/simulation_methodology.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import write_csv  # noqa: E402
from solution_quality import approx_ratio  # noqa: E402

from aquila.agents.harness import generate_workload  # noqa: E402
from aquila.interface.backend import AerBackend, depolarizing_noise_backend  # noqa: E402
from aquila.optimizer.classical_baselines import brute_force_optimal  # noqa: E402
from aquila.optimizer.qaoa_solver import solve_qaoa  # noqa: E402

NUM_TASKS = 8
SEEDS = range(5)
ERROR_RATES = [0.0, 0.005, 0.01, 0.02, 0.05]


def main() -> None:
    rows = []
    for error_rate in ERROR_RATES:
        for seed in SEEDS:
            problem = generate_workload(NUM_TASKS, seed=seed * 1000 + NUM_TASKS)
            _, optimal_cost = brute_force_optimal(problem)

            backend = (
                AerBackend(seed=seed, shots=4096)
                if error_rate == 0.0
                else depolarizing_noise_backend(error_rate=error_rate, seed=seed, shots=4096)
            )
            result = solve_qaoa(problem, backend, reps=3, maxiter=150, num_restarts=3, seed=seed)
            ratio = approx_ratio(optimal_cost, result.best_cost)

            rows.append(
                {
                    "error_rate": error_rate,
                    "seed": seed,
                    "optimal_cost": optimal_cost,
                    "qaoa_cost": result.best_cost,
                    "approx_ratio": ratio,
                    # Raw Hamiltonian expectation value at the converged parameters -
                    # a circuit-fidelity signal, distinct from post-selected solution
                    # quality. Confirms the noise model has a real physical effect
                    # even when post-selection masks it at the solution-quality level.
                    "final_expectation": result.objective_history[-1],
                }
            )
            print(f"error_rate={error_rate:.3f} seed={seed}  approx_ratio={ratio:.3f}")

    path = write_csv("simulation_methodology.csv", list(rows[0].keys()), rows)
    print(f"\nWrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
