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

import hashlib
import os
import struct
import time
from dataclasses import dataclass, field

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

    @property
    def identity_id(self) -> bytes:
        """Stable 16-byte identifier: SHA-256 over both public keys, truncated.

        Binding both keys means an identifier can only be claimed by the
        holder of the matching KEM and signature public keys.
        """
        return hashlib.sha256(self.kem_public_key + self.dsa_public_key).digest()[:16]

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
    sender_id: bytes = b""
    recipient_id: bytes = b""
    context: bytes = b""
    sequence: int = 0
    timestamp_ms: int = 0

    def header(self) -> bytes:
        """Canonical, length-prefixed header. It is both the AEAD associated
        data and the first part of the signed payload, so sender, recipient,
        request/response role, sequence number and timestamp cannot be
        altered or replayed under a different binding."""
        return (
            b"AQUILA-v2"
            + struct.pack(">H", len(self.sender_id)) + self.sender_id
            + struct.pack(">H", len(self.recipient_id)) + self.recipient_id
            + struct.pack(">H", len(self.context)) + self.context
            + struct.pack(">QQ", self.sequence, self.timestamp_ms)
        )

    def wire_size(self) -> int:
        """Total serialized size in bytes (header + KEM ct + nonce + AEAD ct + signature)."""
        return (
            len(self.header()) + len(self.kem_ciphertext) + len(self.nonce)
            + len(self.ciphertext) + len(self.signature)
        )


@dataclass
class ReplayGuard:
    """Receiver-side replay and freshness protection.

    Accepts an envelope only if its timestamp is within `max_age_s` of the
    receiver's clock and its sequence number is strictly greater than the
    highest accepted for that (sender, context) pair.
    """

    max_age_s: float = 300.0
    _highest: dict[tuple[bytes, bytes], int] = field(default_factory=dict)

    def check_and_update(self, envelope: SecureEnvelope, now_ms: int | None = None) -> None:
        now = int(time.time() * 1000) if now_ms is None else now_ms
        if abs(now - envelope.timestamp_ms) > self.max_age_s * 1000:
            raise ValueError("envelope timestamp outside freshness window")
        key = (envelope.sender_id, envelope.context)
        if envelope.sequence <= self._highest.get(key, -1):
            raise ValueError("replayed or out-of-order sequence number")
        self._highest[key] = envelope.sequence


class SecureAgentChannel:
    """Pairwise post-quantum-secured channel between two agent identities."""

    @staticmethod
    def seal(
        message: bytes,
        recipient: AgentIdentity,
        sender: AgentIdentity,
        context: bytes = b"",
        sequence: int = 0,
        timestamp_ms: int | None = None,
    ) -> SecureEnvelope:
        """Encrypt `message` for `recipient` and sign it as `sender`.

        `context` labels the message role (for example b"request" or
        b"response"), so a response cannot be replayed as a request.
        """
        kem_ciphertext, shared_secret = _kem.encrypt(recipient.kem_public_key)
        nonce = os.urandom(_NONCE_LENGTH_BYTES)
        header_only = SecureEnvelope(
            kem_ciphertext=kem_ciphertext, nonce=nonce, ciphertext=b"", signature=b"",
            sender_id=sender.identity_id, recipient_id=recipient.identity_id,
            context=context, sequence=sequence,
            timestamp_ms=int(time.time() * 1000) if timestamp_ms is None else timestamp_ms,
        )
        header = header_only.header()
        ciphertext = AESGCM(shared_secret).encrypt(nonce, message, associated_data=header)
        signature = _dsa.sign(sender.dsa_secret_key, header + kem_ciphertext + nonce + ciphertext)
        return SecureEnvelope(
            kem_ciphertext=kem_ciphertext, nonce=nonce, ciphertext=ciphertext, signature=signature,
            sender_id=header_only.sender_id, recipient_id=header_only.recipient_id,
            context=context, sequence=sequence, timestamp_ms=header_only.timestamp_ms,
        )

    @staticmethod
    def open_envelope(
        envelope: SecureEnvelope,
        recipient: AgentIdentity,
        sender_dsa_public_key: bytes,
        replay_guard: ReplayGuard | None = None,
        expected_context: bytes | None = None,
    ) -> bytes:
        """Verify `envelope`'s signature and decrypt it as `recipient`.

        Raises ValueError if the signature does not verify or the AEAD tag
        does not match (tampering, wrong recipient, or wrong sender key).
        """
        header = envelope.header()
        signed_payload = header + envelope.kem_ciphertext + envelope.nonce + envelope.ciphertext
        if not _dsa.verify(sender_dsa_public_key, signed_payload, envelope.signature):
            raise ValueError("signature verification failed")
        if envelope.recipient_id and envelope.recipient_id != recipient.identity_id:
            raise ValueError("envelope addressed to a different recipient")
        if expected_context is not None and envelope.context != expected_context:
            raise ValueError("unexpected message context")
        if replay_guard is not None:
            replay_guard.check_and_update(envelope)

        shared_secret = _kem.decrypt(recipient.kem_secret_key, envelope.kem_ciphertext)
        aead = AESGCM(shared_secret)
        try:
            return aead.decrypt(envelope.nonce, envelope.ciphertext, associated_data=header)
        except InvalidTag as exc:
            raise ValueError("AEAD authentication failed") from exc
