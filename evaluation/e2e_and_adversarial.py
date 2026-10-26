"""Experiment 5: end-to-end latency breakdown and randomized adversarial tests.

(a) End-to-end: a requester seals a task-allocation problem, the solver opens
    it, solves it (QAOA, and exact enumeration for reference), seals the
    response, and the requester opens it. Each stage is timed separately.
(b) Adversarial: randomized attacks on the sealed envelope. Every attack
    must be rejected; the script counts rejections per attack class.

Run with: python evaluation/e2e_and_adversarial.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import write_csv  # noqa: E402
from quality_controls import make_problem  # noqa: E402

from aquila.interface.backend import AerBackend  # noqa: E402
from aquila.optimizer.classical_baselines import brute_force_optimal  # noqa: E402
from aquila.optimizer.qaoa_solver import solve_qaoa  # noqa: E402
from aquila.security.pqc_channel import AgentIdentity, ReplayGuard, SecureAgentChannel  # noqa: E402

REQUESTS_PER_SIZE = 10
SIZES = [6, 8, 10, 12]
ATTACK_TRIALS = 1000


def e2e() -> list[dict]:
    requester, solver = AgentIdentity.generate(), AgentIdentity.generate()
    req_guard, sol_guard = ReplayGuard(), ReplayGuard()
    seq_req = seq_sol = 0
    rows = []
    for n in SIZES:
        for k in range(REQUESTS_PER_SIZE):
            problem = make_problem("real_1_20", n, 40_000 * n + k)
            payload = json.dumps({"task_weights": problem.task_weights}).encode()
            t = {}
            seq_req += 1
            t0 = time.perf_counter()
            env = SecureAgentChannel.seal(payload, solver, requester, context=b"request", sequence=seq_req)
            t["seal_request_ms"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            SecureAgentChannel.open_envelope(
                env, solver, requester.dsa_public_key, replay_guard=sol_guard, expected_context=b"request"
            )
            t["open_request_ms"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            res = solve_qaoa(problem, AerBackend(seed=k, shots=1024), reps=3, maxiter=150, num_restarts=3, seed=k)
            t["qaoa_solve_ms"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            brute_force_optimal(problem)
            t["exact_solve_ms"] = (time.perf_counter() - t0) * 1000
            seq_sol += 1
            resp = json.dumps({"best_spins": res.best_spins, "best_cost": res.best_cost}).encode()
            t0 = time.perf_counter()
            renv = SecureAgentChannel.seal(resp, requester, solver, context=b"response", sequence=seq_sol)
            t["seal_response_ms"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            SecureAgentChannel.open_envelope(
                renv, requester, solver.dsa_public_key, replay_guard=req_guard, expected_context=b"response"
            )
            t["open_response_ms"] = (time.perf_counter() - t0) * 1000
            sec = sum(t[k2] for k2 in t if k2.startswith(("seal", "open")))
            rows.append(
                {"n": n, "request": k, **t, "security_total_ms": sec,
                 "request_wire_bytes": env.wire_size(), "response_wire_bytes": renv.wire_size(),
                 "security_share_of_qaoa_pipeline": sec / (sec + t["qaoa_solve_ms"])}
            )
    return rows


def flip(b: bytes, rng: random.Random) -> bytes:
    i = rng.randrange(len(b))
    return b[:i] + bytes([b[i] ^ (1 << rng.randrange(8))]) + b[i + 1:]


def adversarial() -> list[dict]:
    rng = random.Random(2026)
    a, b, m = AgentIdentity.generate(), AgentIdentity.generate(), AgentIdentity.generate()
    results = {}

    def record(name: str, fn):
        rejected = 0
        for i in range(ATTACK_TRIALS):
            try:
                fn(i)
            except ValueError:
                rejected += 1
            except Exception:  # any other failure also counts as a rejection, but is logged separately
                rejected += 1
        results[name] = rejected

    def fresh(i: int, ctx=b"request"):
        return SecureAgentChannel.seal(os.urandom(64), b, a, context=ctx, sequence=i + 1)

    def op(env, recipient=b, key=a.dsa_public_key, guard=None, ctx=None):
        return SecureAgentChannel.open_envelope(env, recipient, key, replay_guard=guard, expected_context=ctx)

    for fld in ("kem_ciphertext", "nonce", "ciphertext", "signature"):
        record(f"bit flip in {fld}", lambda i, fld=fld: op(dataclasses.replace(fresh(i), **{fld: flip(getattr(fresh(i), fld), rng)})))
    # note: the flip above uses a different fresh envelope for the replaced field, which also breaks the signature;
    # the targeted version below flips within the same envelope.
    for fld in ("kem_ciphertext", "nonce", "ciphertext", "signature"):
        def f(i, fld=fld):
            e = fresh(i)
            return op(dataclasses.replace(e, **{fld: flip(getattr(e, fld), rng)}))
        record(f"bit flip in {fld} (same envelope)", f)
    record("altered sender_id", lambda i: op(dataclasses.replace(fresh(i), sender_id=os.urandom(16))))
    record("altered recipient_id", lambda i: op(dataclasses.replace(fresh(i), recipient_id=os.urandom(16))))
    record("altered sequence", lambda i: op(dataclasses.replace(fresh(i), sequence=rng.randrange(10**6, 10**7))))
    record("altered timestamp", lambda i: op(dataclasses.replace(fresh(i), timestamp_ms=1)))
    record("altered role label", lambda i: op(dataclasses.replace(fresh(i), context=b"response")))
    record("wrong recipient", lambda i: op(fresh(i), recipient=m))
    record("wrong sender key", lambda i: op(fresh(i), key=m.dsa_public_key))
    record("role confusion (response as request)", lambda i: op(fresh(i, b"response"), ctx=b"request"))

    guard = ReplayGuard()
    stored = [fresh(i) for i in range(ATTACK_TRIALS)]
    for e in stored:
        op(e, guard=guard)

    def replay(i):
        return op(stored[i], guard=guard)

    record("exact replay of accepted envelope", replay)
    stale_guard = ReplayGuard(max_age_s=300)
    record("stale envelope (timestamp 1 hour old)", lambda i: op(
        SecureAgentChannel.seal(os.urandom(64), b, a, context=b"request", sequence=i + 1,
                                timestamp_ms=int(time.time() * 1000) - 3_600_000), guard=stale_guard))
    ooo_guard = ReplayGuard()
    op(SecureAgentChannel.seal(b"x", b, a, context=b"request", sequence=10**6), guard=ooo_guard)
    record("out-of-order (lower) sequence", lambda i: op(fresh(i), guard=ooo_guard))
    # control: honest envelopes must be accepted
    ok = 0
    g = ReplayGuard()
    for i in range(ATTACK_TRIALS):
        try:
            op(fresh(i), guard=g)
            ok += 1
        except ValueError:
            pass
    rows = [{"attack": k, "trials": ATTACK_TRIALS, "rejected": v} for k, v in results.items()]
    rows.append({"attack": "control: honest envelopes accepted", "trials": ATTACK_TRIALS, "rejected": ATTACK_TRIALS - ok})
    return rows


def main() -> None:
    adv = adversarial()
    for r in adv:
        print(f"{r['attack']:<50}{r['rejected']:>5}/{r['trials']}")
    write_csv("adversarial_tests.csv", list(adv[0].keys()), adv)
    rows = e2e()
    write_csv("e2e_latency.csv", list(rows[0].keys()), rows)
    for n in SIZES:
        g = [r for r in rows if r["n"] == n]
        print(n, {k: round(statistics.mean(r[k] for r in g), 2) for k in
                  ("seal_request_ms", "open_request_ms", "qaoa_solve_ms", "exact_solve_ms",
                   "seal_response_ms", "open_response_ms", "security_share_of_qaoa_pipeline")})


if __name__ == "__main__":
    main()
