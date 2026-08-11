"""Post-quantum-secured agent-to-agent messaging.

Uses ML-KEM-768 to establish a per-message shared secret (encapsulated
against the recipient's long-term KEM public key), AES-256-GCM to encrypt
the payload under that secret, and ML-DSA-65 to sign the ciphertext so the
recipient can authenticate the sender. This is AQUILA's hardening layer: it
protects agent-to-agent traffic against classical and harvest-now-decrypt-
later quantum adversaries, secondary to the quantum-classical optimization
interface that is the paper's primary contribution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pqcrypto.kem.ml_kem_768 as _kem
import pqcrypto.sign.ml_dsa_65 as _dsa
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_NONCE_LENGTH_BYTES = 12


@dataclass
class AgentIdentity:
    """An agent's long-term post-quantum keypairs.

    `kem_public_key` is published so other agents can seal messages to this
    agent; `dsa_public_key` is published so other agents can verify this
    agent's signatures. The corresponding secret keys never leave the
    identity object.
    """

    kem_public_key: bytes
    kem_secret_key: bytes
    dsa_public_key: bytes
    dsa_secret_key: bytes

    @classmethod
    def generate(cls) -> AgentIdentity:
        kem_pk, kem_sk = _kem.generate_keypair()
        dsa_pk, dsa_sk = _dsa.generate_keypair()
        return cls(
            kem_public_key=kem_pk,
            kem_secret_key=kem_sk,
            dsa_public_key=dsa_pk,
            dsa_secret_key=dsa_sk,
        )


@dataclass
class SecureEnvelope:
    """A sealed, authenticated message ready for transport over an untrusted channel."""

    kem_ciphertext: bytes
    nonce: bytes
    ciphertext: bytes
    signature: bytes


class SecureAgentChannel:
    """Pairwise post-quantum-secured channel between two agent identities."""

    @staticmethod
    def seal(message: bytes, recipient: AgentIdentity, sender: AgentIdentity) -> SecureEnvelope:
        """Encrypt `message` for `recipient` and sign it as `sender`."""
        kem_ciphertext, shared_secret = _kem.encrypt(recipient.kem_public_key)
        nonce = os.urandom(_NONCE_LENGTH_BYTES)
        aead = AESGCM(shared_secret)
        ciphertext = aead.encrypt(nonce, message, associated_data=None)
        signature = _dsa.sign(sender.dsa_secret_key, kem_ciphertext + nonce + ciphertext)
        return SecureEnvelope(
            kem_ciphertext=kem_ciphertext,
            nonce=nonce,
            ciphertext=ciphertext,
            signature=signature,
        )

    @staticmethod
    def open_envelope(
        envelope: SecureEnvelope, recipient: AgentIdentity, sender_dsa_public_key: bytes
    ) -> bytes:
        """Verify `envelope`'s signature and decrypt it as `recipient`.

        Raises ValueError if the signature does not verify or the AEAD tag
        does not match (tampering, wrong recipient, or wrong sender key).
        """
        signed_payload = envelope.kem_ciphertext + envelope.nonce + envelope.ciphertext
        if not _dsa.verify(sender_dsa_public_key, signed_payload, envelope.signature):
            raise ValueError("signature verification failed")

        shared_secret = _kem.decrypt(recipient.kem_secret_key, envelope.kem_ciphertext)
        aead = AESGCM(shared_secret)
        try:
            return aead.decrypt(envelope.nonce, envelope.ciphertext, associated_data=None)
        except InvalidTag as exc:
            raise ValueError("AEAD authentication failed") from exc
