"""End-to-end AQUILA demo.

A scheduler agent generates a synthetic task-allocation workload, submits it
to a solver agent over a post-quantum-secured channel, the solver agent
offloads the optimization to a QAOA solver via the quantum-classical
interface, and the result is returned and compared against classical
baselines.

Run with: python examples/demo_end_to_end.py
"""

from __future__ import annotations

from aquila.agents.harness import AllocationAgent, generate_workload
from aquila.interface.backend import AerBackend
from aquila.optimizer.classical_baselines import (
    brute_force_optimal,
    greedy_partition,
    simulated_annealing_partition,
)


def main() -> None:
    print("=== AQUILA: Agentic QUantum Interface for Load Allocation ===\n")

    scheduler = AllocationAgent.create("scheduler-agent")
    solver = AllocationAgent.create("quantum-solver-agent")
    print(f"Created agents: '{scheduler.name}' and '{solver.name}'")
    print("Each agent generated its own ML-KEM-768 / ML-DSA-65 identity keypair.\n")

    problem = generate_workload(num_tasks=8, seed=42)
    print(f"Workload: {problem.num_tasks} tasks, weights={problem.task_weights}\n")

    backend = AerBackend(seed=42, shots=4096)

    print("--- Solving via AQUILA (QAOA offload over a PQC-secured channel) ---")
    result = scheduler.request_allocation(problem, solver, backend, reps=3, maxiter=200, seed=42)
    cluster_a, cluster_b = problem.clusters_from_assignment(result.best_spins)
    print(f"QAOA cost: {result.best_cost:.4f}  (evaluations: {result.num_evaluations})")
    print(f"Cluster A: {cluster_a}")
    print(f"Cluster B: {cluster_b}\n")

    print("--- Classical baselines ---")
    _, optimal_cost = brute_force_optimal(problem)
    _, greedy_cost = greedy_partition(problem)
    _, sa_cost = simulated_annealing_partition(problem, seed=42)
    print(f"Brute-force optimal cost: {optimal_cost:.4f}")
    print(f"Greedy (LPT) cost:        {greedy_cost:.4f}")
    print(f"Simulated annealing cost: {sa_cost:.4f}")

    if result.best_cost > 0:
        print(f"\nQAOA approximation ratio vs. optimal: {optimal_cost / result.best_cost:.4f}")
    else:
        print("\nQAOA found the perfect (zero-cost) partition.")


if __name__ == "__main__":
    main()
