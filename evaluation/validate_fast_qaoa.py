"""Cross-check fast_qaoa against the Aer EstimatorV2 / QAOAAnsatz path."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fast_qaoa as fq  # noqa: E402
from qiskit.circuit.library import QAOAAnsatz  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402

from aquila.interface.problem import TaskAllocationProblem  # noqa: E402
from aquila.optimizer.qaoa_solver import build_cost_operator  # noqa: E402

rng = np.random.default_rng(1)
worst = 0.0
for n in (4, 6, 8, 10):
    for reps in (1, 2, 3):
        w = [round(float(x), 3) for x in rng.uniform(1, 20, n)]
        prob = TaskAllocationProblem(task_weights=w)
        op = build_cost_operator(prob)
        ans = QAOAAnsatz(cost_operator=op, reps=reps)
        params = rng.uniform(0, np.pi, ans.num_parameters)
        sv = Statevector(ans.assign_parameters(params))
        ref = float(np.real(sv.expectation_value(op)))
        h = fq.ising_diagonal(w)
        mine = fq.expectation(h, n, params)
        pref = sv.probabilities()
        pmine = fq.probabilities(h, n, params)
        err = max(abs(ref - mine) / max(1, abs(ref)), float(np.max(np.abs(pref - pmine))))
        worst = max(worst, err)
        print(f"n={n} p={reps} E_qiskit={ref:.6f} E_numpy={mine:.6f} err={err:.2e}")
print("worst error", worst)
assert worst < 1e-8
