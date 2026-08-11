"""Pluggable quantum backend abstraction.

AQUILA's interface layer talks to backends purely through the Qiskit
EstimatorV2/SamplerV2 primitive protocol. Today that resolves to the local
Aer simulator; a real QPU backend (e.g. Qiskit Runtime's EstimatorV2/
SamplerV2) implements the same primitive interface, so the optimizer and
agent layers above it run unmodified when a physical device becomes
available. This is the "scalable quantum-classical interface" contribution:
the boundary between classical orchestration and quantum execution is fixed
at the primitive level, not at the simulator.
"""

from __future__ import annotations

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer.primitives import SamplerV2 as AerSampler


class AerBackend:
    """Local Aer-simulator backend, optionally carrying a noise model."""

    def __init__(
        self,
        noise_model: NoiseModel | None = None,
        seed: int | None = None,
        shots: int = 4096,
    ) -> None:
        self._noise_model = noise_model
        self._seed = seed
        self._shots = shots

    def _backend_options(self) -> dict[str, object]:
        options: dict[str, object] = {}
        if self._seed is not None:
            options["seed_simulator"] = self._seed
        if self._noise_model is not None:
            options["noise_model"] = self._noise_model
        return options

    def simulator(self) -> AerSimulator:
        """The underlying AerSimulator, used as the transpile target so
        circuits are lowered to a basis the primitives can execute."""
        return AerSimulator(**self._backend_options())

    def estimator(self) -> AerEstimator:
        return AerEstimator(
            options={
                "backend_options": self._backend_options(),
                "run_options": {"shots": self._shots},
            }
        )

    def sampler(self) -> AerSampler:
        return AerSampler(options={"backend_options": self._backend_options()})


def depolarizing_noise_backend(
    error_rate: float = 0.01, seed: int | None = None, shots: int = 4096
) -> AerBackend:
    """Convenience factory for a simple depolarizing-noise Aer backend.

    Used by the evaluation suite to measure QAOA solution-quality
    degradation under NISQ-era noise (the quantum-simulation-methodology
    contribution), against the ideal noiseless backend as a control.
    """
    single_qubit_error = depolarizing_error(error_rate, 1)
    two_qubit_error = depolarizing_error(error_rate * 2, 2)

    noise_model = NoiseModel()
    single_qubit_gates = ["rz", "rx", "ry", "sx", "x", "h", "u", "u1", "u2", "u3"]
    two_qubit_gates = ["cx", "cz", "rzz"]
    noise_model.add_all_qubit_quantum_error(single_qubit_error, single_qubit_gates)
    noise_model.add_all_qubit_quantum_error(two_qubit_error, two_qubit_gates)
    return AerBackend(noise_model=noise_model, seed=seed, shots=shots)
