# AQUILA

[![AQUILA CI](https://github.com/sunilgentyala/AQUILA/actions/workflows/ci.yaml/badge.svg)](https://github.com/sunilgentyala/AQUILA/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-orange)](https://github.com/sunilgentyala/AQUILA/releases)

**Agentic QUantum Interface for Load Allocation.**

AQUILA lets an agentic AI system offload combinatorial subtasks, such as task and resource allocation across compute clusters, to a QAOA solver through a backend-agnostic quantum-classical interface, hardened with a post-quantum-secured (ML-KEM-768 / ML-DSA-65) agent messaging layer.

> Companion paper (in preparation, targeting the Sept 10, 2026 deadline): *"AQUILA: A Scalable Quantum-Classical Interface for Optimization Offload in Agentic AI Systems"* — Sunil Gentyala, HCLTech, Dallas TX. To be submitted to IEEE MCSoC 2026 (Quantum and Hybrid Classical-Quantum Computing SoCs track).

---

## Architecture

```
  Requester Agent                                    Solver Agent
  ┌───────────────┐   PQC-secured envelope            ┌───────────────┐
  │ AllocationAgent│ ──(ML-KEM-768 + ML-DSA-65)──────> │ AllocationAgent│
  └───────────────┘   (tamper/impersonation-checked)   └───────┬───────┘
                                                                │
                                                                v
                                                     ┌─────────────────────┐
                                                     │   Interface layer    │
                                                     │  TaskAllocationProblem│
                                                     │  -> Ising Hamiltonian │
                                                     └──────────┬──────────┘
                                                                │
                                                                v
                                                     ┌─────────────────────┐
                                                     │   Optimizer layer    │
                                                     │  QAOA (multi-restart) │
                                                     │  vs. classical        │
                                                     │  baselines             │
                                                     └──────────┬──────────┘
                                                                │
                                                                v
                                                     ┌─────────────────────┐
                                                     │   Backend (pluggable) │
                                                     │  Aer simulator today; │
                                                     │  real QPU later,       │
                                                     │  same primitive API    │
                                                     └─────────────────────┘
```

The interface layer talks to backends purely through Qiskit's `EstimatorV2`/`SamplerV2` primitive protocol. Nothing above that boundary refers to the simulator directly, so a hardware-backed primitive implementation would satisfy the interface without changes to the optimizer or agent code.

## Results summary

Evaluated across problem sizes 4-14 tasks (5 seeds each) against greedy (LPT) and multi-restart simulated-annealing baselines:

| n | QAOA ratio | SA ratio | Greedy ratio |
|---|---|---|---|
| 4 | 1.000 | 1.000 | 1.000 |
| 8 | 1.000 | 0.525 | 0.070 |
| 12 | 0.957 | 0.016 | 0.017 |
| 14 | 0.737 | 0.002 | 0.000 |

PQC hardening layer (30-trial mean vs. RSA-3072/ECDSA-P256): ML-KEM-768 key generation is ~1000x faster than RSA-3072 keygen (0.30ms vs. 322.70ms), at the cost of larger keys/signatures. Full results and figures: [`evaluation/results/`](evaluation/results/).

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.12+. Core dependencies: `qiskit`, `qiskit-aer`, `pqcrypto`, `cryptography`.

## Usage

```bash
python examples/demo_end_to_end.py   # full request -> QAOA solve -> secured response cycle
pytest tests/                         # 35 tests covering all four layers
python evaluation/solution_quality.py # reproduce the solution-quality sweep
python evaluation/make_plots.py       # regenerate figures from evaluation/results/*.csv
```

## Citation

If you use AQUILA in research, please cite:

```bibtex
@misc{gentyala2026aquila,
  title        = {{AQUILA}: A Scalable Quantum-Classical Interface for
                  Optimization Offload in Agentic {AI} Systems},
  author       = {Gentyala, Sunil},
  year         = {2026},
  institution  = {HCLTech, Dallas TX},
  note         = {In preparation for IEEE MCSoC 2026. \url{https://github.com/sunilgentyala/AQUILA}}
}
```

---

## Author

**Sunil Gentyala** — IEEE Senior Member
Lead Cybersecurity and AI Security Consultant, HCLTech, Dallas, TX, USA
sunil.gentyala@ieee.org

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin)](https://www.linkedin.com/in/sunil-gentyala/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/sunilgentyala)

---

## License

Apache-2.0 — see [LICENSE](LICENSE).
