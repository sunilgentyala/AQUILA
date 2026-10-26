"""Experiment 2: PQC hardening-layer overhead vs. a classical baseline.

Compares AQUILA's ML-KEM-768 + ML-DSA-65 channel against a classical
RSA-3072 (key transport) + ECDSA-P256 (signatures) baseline: wall-clock cost
of each primitive operation, and on-the-wire size overhead. This quantifies
the cost of the PQC hardening layer relative to what it replaces.

Run with: python evaluation/security_overhead.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pqcrypto.kem.ml_kem_768 as kem  # noqa: E402
import pqcrypto.sign.ml_dsa_65 as dsa  # noqa: E402
from common import write_csv  # noqa: E402
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa  # noqa: E402

TRIALS = 30
MESSAGE = b"AQUILA: allocate task-7 to cluster-A" * 4  # representative agent message
KEM_PAYLOAD = b"0" * 32  # analogous to the 32-byte shared secret ML-KEM transports


def measure_pqc() -> dict:
    keygen_times, encap_times, decap_times = [], [], []
    sign_keygen_times, sign_times, verify_times = [], [], []
    pk_sizes, ct_sizes, sig_sizes = [], [], []

    for _ in range(TRIALS):
        t0 = time.perf_counter()
        pk, sk = kem.generate_keypair()
        keygen_times.append(time.perf_counter() - t0)
        pk_sizes.append(len(pk))

        t0 = time.perf_counter()
        ct, ss = kem.encrypt(pk)
        encap_times.append(time.perf_counter() - t0)
        ct_sizes.append(len(ct))

        t0 = time.perf_counter()
        kem.decrypt(sk, ct)
        decap_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        vk, dsk = dsa.generate_keypair()
        sign_keygen_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        signature = dsa.sign(dsk, MESSAGE)
        sign_times.append(time.perf_counter() - t0)
        sig_sizes.append(len(signature))

        t0 = time.perf_counter()
        dsa.verify(vk, MESSAGE, signature)
        verify_times.append(time.perf_counter() - t0)

    return {
        "scheme": "ML-KEM-768 + ML-DSA-65 (PQC)",
        "kem_keygen_ms": statistics.mean(keygen_times) * 1000,
        "kem_encap_ms": statistics.mean(encap_times) * 1000,
        "kem_decap_ms": statistics.mean(decap_times) * 1000,
        "sig_keygen_ms": statistics.mean(sign_keygen_times) * 1000,
        "sign_ms": statistics.mean(sign_times) * 1000,
        "verify_ms": statistics.mean(verify_times) * 1000,
        "kem_public_key_bytes": statistics.mean(pk_sizes),
        "kem_ciphertext_bytes": statistics.mean(ct_sizes),
        "signature_bytes": statistics.mean(sig_sizes),
    }


def measure_classical() -> dict:
    keygen_times, encap_times, decap_times = [], [], []
    sign_keygen_times, sign_times, verify_times = [], [], []
    pk_sizes, ct_sizes, sig_sizes = [], [], []
    oaep = padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None
    )

    for _ in range(TRIALS):
        t0 = time.perf_counter()
        rsa_sk = rsa.generate_private_key(public_exponent=65537, key_size=3072)
        keygen_times.append(time.perf_counter() - t0)
        rsa_pk = rsa_sk.public_key()
        rsa_pub_der = rsa_pk.public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        pk_sizes.append(len(rsa_pub_der))

        t0 = time.perf_counter()
        ct = rsa_pk.encrypt(KEM_PAYLOAD, oaep)
        encap_times.append(time.perf_counter() - t0)
        ct_sizes.append(len(ct))

        t0 = time.perf_counter()
        rsa_sk.decrypt(ct, oaep)
        decap_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        ec_sk = ec.generate_private_key(ec.SECP256R1())
        sign_keygen_times.append(time.perf_counter() - t0)
        ec_pk = ec_sk.public_key()

        t0 = time.perf_counter()
        signature = ec_sk.sign(MESSAGE, ec.ECDSA(hashes.SHA256()))
        sign_times.append(time.perf_counter() - t0)
        sig_sizes.append(len(signature))

        t0 = time.perf_counter()
        ec_pk.verify(signature, MESSAGE, ec.ECDSA(hashes.SHA256()))
        verify_times.append(time.perf_counter() - t0)

    return {
        "scheme": "RSA-3072 + ECDSA-P256 (classical)",
        "kem_keygen_ms": statistics.mean(keygen_times) * 1000,
        "kem_encap_ms": statistics.mean(encap_times) * 1000,
        "kem_decap_ms": statistics.mean(decap_times) * 1000,
        "sig_keygen_ms": statistics.mean(sign_keygen_times) * 1000,
        "sign_ms": statistics.mean(sign_times) * 1000,
        "verify_ms": statistics.mean(verify_times) * 1000,
        "kem_public_key_bytes": statistics.mean(pk_sizes),
        "kem_ciphertext_bytes": statistics.mean(ct_sizes),
        "signature_bytes": statistics.mean(sig_sizes),
    }


def main() -> None:
    print(f"Running {TRIALS} trials per scheme...")
    pqc_row = measure_pqc()
    classical_row = measure_classical()

    for row in (pqc_row, classical_row):
        print(f"\n{row['scheme']}:")
        for key, value in row.items():
            if key != "scheme":
                print(f"  {key}: {value:.3f}")

    path = write_csv("security_overhead.csv", list(pqc_row.keys()), [pqc_row, classical_row])
    print(f"\nWrote 2 rows to {path}")


if __name__ == "__main__":
    main()
