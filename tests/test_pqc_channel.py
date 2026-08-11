import dataclasses

import pytest

from aquila.security.pqc_channel import AgentIdentity, SecureAgentChannel


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
    with pytest.raises(ValueError, match="AEAD authentication failed"):
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
