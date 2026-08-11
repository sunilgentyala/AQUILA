"""Minimal multi-agent harness wiring together the interface, optimizer, and
security layers into a single request/response cycle: a requester agent
submits a task-allocation problem, a solver agent solves it via QAOA over
the quantum-classical interface, and the result is returned over a
post-quantum-secured channel. The channel is simulated in-process (no real
network transport) - sufficient for studying the interface and security
properties in isolation, which is the scope of this evaluation.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from aquila.interface.backend import AerBackend
from aquila.interface.problem import TaskAllocationProblem
from aquila.optimizer.qaoa_solver import QAOAResult, solve_qaoa
from aquila.security.pqc_channel import AgentIdentity, SecureAgentChannel


def generate_workload(
    num_tasks: int,
    seed: int | None = None,
    weight_range: tuple[float, float] = (1.0, 20.0),
) -> TaskAllocationProblem:
    """Generate a synthetic agentic task-allocation workload."""
    rng = random.Random(seed)
    weights = [round(rng.uniform(*weight_range), 3) for _ in range(num_tasks)]
    return TaskAllocationProblem(task_weights=weights)


@dataclass
class AllocationAgent:
    """An agent that can request a task allocation be solved over the AQUILA interface."""

    name: str
    identity: AgentIdentity

    @classmethod
    def create(cls, name: str) -> AllocationAgent:
        return cls(name=name, identity=AgentIdentity.generate())

    def request_allocation(
        self,
        problem: TaskAllocationProblem,
        solver: AllocationAgent,
        backend: AerBackend,
        reps: int = 2,
        maxiter: int = 150,
        seed: int | None = None,
    ) -> QAOAResult:
        """Submit `problem` to `solver` over a PQC-secured channel and get the result back."""
        request_payload = json.dumps(
            {"task_weights": problem.task_weights, "task_ids": problem.task_ids}
        ).encode()
        request_envelope = SecureAgentChannel.seal(
            request_payload, recipient=solver.identity, sender=self.identity
        )

        # --- solver agent receives and processes the request ---
        decoded_request = SecureAgentChannel.open_envelope(
            request_envelope,
            recipient=solver.identity,
            sender_dsa_public_key=self.identity.dsa_public_key,
        )
        received = json.loads(decoded_request.decode())
        received_problem = TaskAllocationProblem(
            task_weights=received["task_weights"], task_ids=received["task_ids"]
        )
        result = solve_qaoa(received_problem, backend, reps=reps, maxiter=maxiter, seed=seed)

        response_payload = json.dumps(
            {"best_spins": result.best_spins, "best_cost": result.best_cost}
        ).encode()
        response_envelope = SecureAgentChannel.seal(
            response_payload, recipient=self.identity, sender=solver.identity
        )

        # --- requester agent receives the solved allocation ---
        decoded_response = SecureAgentChannel.open_envelope(
            response_envelope,
            recipient=self.identity,
            sender_dsa_public_key=solver.identity.dsa_public_key,
        )
        response = json.loads(decoded_response.decode())
        return QAOAResult(
            best_spins=response["best_spins"],
            best_cost=response["best_cost"],
            optimal_parameters=result.optimal_parameters,
            objective_history=result.objective_history,
            num_evaluations=result.num_evaluations,
        )
