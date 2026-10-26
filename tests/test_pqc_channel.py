import dataclasses

import pytest

from aquila.security.pqc_channel import AgentIdentity, ReplayGuard, SecureAgentChannel


@pytest.fixture
def identities():
    return AgentIdentity.generate(), AgentIdentity.generate(), AgentIdentity.generate()


def test_round_trip(identities):
    alice, bob, _mallory = identities
    message = b"allocate task-7 to cluster-A"
    envelope = SecureAgentChannel.seal(message, recipient=bob, sender=alice)
    recovered = SecureAgentChannel.open_envelope(
        envelope, recipient=bob, sender_dsa_public_key=alice.dsa_public_key
    )
    assert recovered == message


def test_wrong_recipient_cannot_decrypt(identities):
    alice, bob, mallory = identities
    envelope = SecureAgentChannel.seal(b"secret", recipient=bob, sender=alice)
    with pytest.raises(ValueError, match="different recipient"):
        SecureAgentChannel.open_envelope(
            envelope, recipient=mallory, sender_dsa_public_key=alice.dsa_public_key
        )


def test_tampered_ciphertext_is_rejected(identities):
    alice, bob, _mallory = identities
    envelope = SecureAgentChannel.seal(b"secret", recipient=bob, sender=alice)
    flipped_byte = bytes([envelope.ciphertext[0] ^ 1])
    tampered = dataclasses.replace(envelope, ciphertext=flipped_byte + envelope.ciphertext[1:])
    with pytest.raises(ValueError, match="signature verification failed"):
        SecureAgentChannel.open_envelope(
            tampered, recipient=bob, sender_dsa_public_key=alice.dsa_public_key
        )


def test_impersonation_is_rejected(identities):
    alice, bob, mallory = identities
    envelope = SecureAgentChannel.seal(b"secret", recipient=bob, sender=alice)
    with pytest.raises(ValueError, match="signature verification failed"):
        SecureAgentChannel.open_envelope(
            envelope, recipient=bob, sender_dsa_public_key=mallory.dsa_public_key
        )


def test_identities_are_unique(identities):
    alice, bob, mallory = identities
    keys = {alice.kem_public_key, bob.kem_public_key, mallory.kem_public_key}
    assert len(keys) == 3


def test_header_fields_are_authenticated(identities):
    alice, bob, _ = identities
    envelope = SecureAgentChannel.seal(b"x", recipient=bob, sender=alice, context=b"request")
    for field, value in (("sequence", 99), ("context", b"response"), ("timestamp_ms", 1)):
        forged = dataclasses.replace(envelope, **{field: value})
        with pytest.raises(ValueError, match="signature verification failed"):
            SecureAgentChannel.open_envelope(
                forged, recipient=bob, sender_dsa_public_key=alice.dsa_public_key
            )


def test_replay_is_rejected(identities):
    alice, bob, _ = identities
    guard = ReplayGuard()
    envelope = SecureAgentChannel.seal(b"x", recipient=bob, sender=alice, sequence=1)
    args = {"recipient": bob, "sender_dsa_public_key": alice.dsa_public_key, "replay_guard": guard}
    assert SecureAgentChannel.open_envelope(envelope, **args) == b"x"
    with pytest.raises(ValueError, match="replayed"):
        SecureAgentChannel.open_envelope(envelope, **args)


def test_stale_envelope_is_rejected(identities):
    alice, bob, _ = identities
    old = SecureAgentChannel.seal(b"x", recipient=bob, sender=alice, sequence=1, timestamp_ms=1)
    with pytest.raises(ValueError, match="freshness"):
        SecureAgentChannel.open_envelope(
            old, recipient=bob, sender_dsa_public_key=alice.dsa_public_key,
            replay_guard=ReplayGuard(),
        )


def test_response_cannot_be_replayed_as_request(identities):
    alice, bob, _ = identities
    response = SecureAgentChannel.seal(b"x", recipient=bob, sender=alice, context=b"response")
    with pytest.raises(ValueError, match="unexpected message context"):
        SecureAgentChannel.open_envelope(
            response, recipient=bob, sender_dsa_public_key=alice.dsa_public_key,
            expected_context=b"request",
        )


def test_wire_size_accounts_for_all_fields(identities):
    alice, bob, _ = identities
    envelope = SecureAgentChannel.seal(b"hello", recipient=bob, sender=alice)
    assert envelope.wire_size() > len(envelope.kem_ciphertext) + len(envelope.signature)
