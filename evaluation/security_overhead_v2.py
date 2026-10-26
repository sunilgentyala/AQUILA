"""Experiment 3 (revised): security-layer accounting.

Fixes the original comparison's problems:
  - KEM and signature key sizes and times are reported separately, plus the
    combined identity cost
  - classical baselines at the 128-bit level (RSA-3072, P-256) and the
    192-bit level (P-384); the PQC sets are NIST category 3
  - key generation is reported per identity (amortized over messages); the
    per-message cost is seal plus open only
  - complete envelope wire size and request-response latency

Run with: python evaluation/security_overhead_v2.py [--trials 200]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import pqcrypto.kem.ml_kem_768 as kem
import pqcrypto.sign.ml_dsa_65 as dsa
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

sys.path.insert(0, str(Path(__file__).parent))
from common import write_csv  # noqa: E402

from aquila.security.pqc_channel import AgentIdentity, SecureAgentChannel  # noqa: E402

PAYLOAD = b"x" * 512  # representative allocation request/response payload
AEAD_OVERHEAD = 12 + 16  # nonce + GCM tag


def bench(fn, trials: int) -> tuple[float, float]:
    xs = []
    for _ in range(trials):
        t = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t) * 1000)
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def spki(pub) -> int:
    fmt = serialization.PublicFormat.SubjectPublicKeyInfo
    return len(pub.public_bytes(serialization.Encoding.DER, fmt))


def ec_baseline(curve, sig_hash):
    """Ephemeral ECDH + HKDF + AES-GCM, with an ECDSA signature."""
    r_priv = ec.generate_private_key(curve())
    s_priv = ec.generate_private_key(curve())
    x962 = (serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)

    def seal():
        eph = ec.generate_private_key(curve())
        key = HKDF(hashes.SHA256(), 32, None, b"baseline").derive(
            eph.exchange(ec.ECDH(), r_priv.public_key())
        )
        ct = AESGCM(key).encrypt(b"\0" * 12, PAYLOAD, None)
        eph_pub = eph.public_key().public_bytes(*x962)
        return eph_pub, ct, s_priv.sign(eph_pub + ct, ec.ECDSA(sig_hash()))

    eph_pub, ct, sig = seal()

    def open_():
        s_priv.public_key().verify(sig, eph_pub + ct, ec.ECDSA(sig_hash()))
        peer = ec.EllipticCurvePublicKey.from_encoded_point(curve(), eph_pub)
        key = HKDF(hashes.SHA256(), 32, None, b"baseline").derive(r_priv.exchange(ec.ECDH(), peer))
        return AESGCM(key).decrypt(b"\0" * 12, ct, None)

    wire = len(eph_pub) + len(ct) + len(sig) + 12
    return seal, open_, wire, spki(r_priv.public_key()) + spki(s_priv.public_key())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--rsa-keygen-trials", type=int, default=20)
    a = ap.parse_args()
    n = a.trials
    rows: list[dict] = []

    def add(scheme, level, metric, mean, sd, unit):
        rows.append(
            {"scheme": scheme, "security_level": level, "metric": metric,
             "mean": mean, "stdev": sd, "unit": unit, "trials": n}
        )

    # --- PQC, NIST category 3 ---
    alice, bob = AgentIdentity.generate(), AgentIdentity.generate()
    m, s = bench(kem.generate_keypair, n)
    add("ML-KEM-768", "cat3", "keygen", m, s, "ms")
    m, s = bench(dsa.generate_keypair, n)
    add("ML-DSA-65", "cat3", "keygen", m, s, "ms")
    m, s = bench(AgentIdentity.generate, n)
    add("PQC identity (KEM+DSA)", "cat3", "keygen", m, s, "ms")
    add("ML-KEM-768", "cat3", "public_key", len(alice.kem_public_key), 0, "bytes")
    add("ML-DSA-65", "cat3", "public_key", len(alice.dsa_public_key), 0, "bytes")
    add("PQC identity (KEM+DSA)", "cat3", "public_key",
        len(alice.kem_public_key) + len(alice.dsa_public_key), 0, "bytes")
    env = SecureAgentChannel.seal(PAYLOAD, recipient=bob, sender=alice, context=b"request", sequence=1)
    add("PQC envelope", "cat3", "wire_size_512B_payload", env.wire_size(), 0, "bytes")
    add("PQC envelope", "cat3", "kem_ciphertext", len(env.kem_ciphertext), 0, "bytes")
    add("PQC envelope", "cat3", "signature", len(env.signature), 0, "bytes")
    add("PQC envelope", "cat3", "header_nonce_tag_overhead",
        env.wire_size() - len(env.kem_ciphertext) - len(env.signature) - len(PAYLOAD), 0, "bytes")

    seq = [0]

    def pqc_seal():
        seq[0] += 1
        return SecureAgentChannel.seal(
            PAYLOAD, recipient=bob, sender=alice, context=b"request", sequence=seq[0]
        )

    m, s = bench(pqc_seal, n)
    add("PQC envelope", "cat3", "seal", m, s, "ms")
    fixed = SecureAgentChannel.seal(PAYLOAD, recipient=bob, sender=alice, context=b"request", sequence=1)
    m, s = bench(lambda: SecureAgentChannel.open_envelope(fixed, bob, alice.dsa_public_key), n)
    add("PQC envelope", "cat3", "open", m, s, "ms")

    def roundtrip():
        req = SecureAgentChannel.seal(PAYLOAD, recipient=bob, sender=alice, context=b"request", sequence=1)
        SecureAgentChannel.open_envelope(req, bob, alice.dsa_public_key)
        rsp = SecureAgentChannel.seal(PAYLOAD, recipient=alice, sender=bob, context=b"response", sequence=1)
        SecureAgentChannel.open_envelope(rsp, alice, bob.dsa_public_key)

    m, s = bench(roundtrip, n)
    add("PQC envelope", "cat3", "request_response_roundtrip", m, s, "ms")

    # --- RSA-3072, 128-bit level ---
    def rsa_key():
        return rsa.generate_private_key(65537, 3072)

    m, s = bench(rsa_key, a.rsa_keygen_trials)
    add("RSA-3072", "128-bit", "keygen", m, s, "ms")
    rk = rsa_key()
    oaep = padding.OAEP(padding.MGF1(hashes.SHA256()), hashes.SHA256(), None)
    pss = padding.PSS(padding.MGF1(hashes.SHA256()), padding.PSS.MAX_LENGTH)
    ct = rk.public_key().encrypt(b"k" * 32, oaep)
    sig = rk.sign(ct, pss, hashes.SHA256())
    m, s = bench(lambda: (rk.public_key().encrypt(b"k" * 32, oaep), rk.sign(ct, pss, hashes.SHA256())), n)
    add("RSA-3072 (OAEP+PSS)", "128-bit", "seal", m, s, "ms")
    m, s = bench(lambda: (rk.public_key().verify(sig, ct, pss, hashes.SHA256()), rk.decrypt(ct, oaep)), n)
    add("RSA-3072 (OAEP+PSS)", "128-bit", "open", m, s, "ms")
    add("RSA-3072 (OAEP+PSS)", "128-bit", "public_key", spki(rk.public_key()), 0, "bytes")
    add("RSA-3072 (OAEP+PSS)", "128-bit", "wire_size_512B_payload",
        len(ct) + len(sig) + AEAD_OVERHEAD + len(PAYLOAD), 0, "bytes")

    # --- elliptic-curve baselines at 128-bit and 192-bit levels ---
    for name, curve, h, level in (
        ("ECDH+ECDSA P-256", ec.SECP256R1, hashes.SHA256, "128-bit"),
        ("ECDH+ECDSA P-384", ec.SECP384R1, hashes.SHA384, "192-bit"),
    ):
        seal_f, open_f, wire, pk = ec_baseline(curve, h)
        m, s = bench(seal_f, n)
        add(name, level, "seal", m, s, "ms")
        m, s = bench(open_f, n)
        add(name, level, "open", m, s, "ms")
        add(name, level, "public_key", pk, 0, "bytes")
        add(name, level, "wire_size_512B_payload", wire + 16 + len(PAYLOAD), 0, "bytes")
        m, s = bench(lambda c=curve: ec.generate_private_key(c()), n)
        add(name, level, "keygen", m, s, "ms")

    write_csv("security_overhead_v2.csv", list(rows[0].keys()), rows)
    for r in rows:
        print(f"{r['scheme']:<26}{r['security_level']:<8}{r['metric']:<28}"
              f"{r['mean']:>12.3f} +/-{r['stdev']:.3f} {r['unit']}")


if __name__ == "__main__":
    main()
