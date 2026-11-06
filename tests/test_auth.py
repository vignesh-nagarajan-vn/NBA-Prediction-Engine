import base64

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

from nba_engine.kalshi.auth import KalshiSigner, load_private_key, signing_path


def test_signing_path_strips_host_and_query():
    url = "https://external-api.kalshi.com/trade-api/v2/portfolio/orders?limit=5&status=resting"
    assert signing_path(url) == "/trade-api/v2/portfolio/orders"
    assert signing_path("/trade-api/ws/v2") == "/trade-api/ws/v2"


def test_ed25519_headers_verify():
    key = ed25519.Ed25519PrivateKey.generate()
    signer = KalshiSigner("key-123", key)
    h = signer.headers("get", "https://x/trade-api/v2/portfolio/balance?x=1", ts_ms=1700000000000)
    assert h["KALSHI-ACCESS-KEY"] == "key-123"
    assert h["KALSHI-ACCESS-TIMESTAMP"] == "1700000000000"
    msg = b"1700000000000GET/trade-api/v2/portfolio/balance"
    key.public_key().verify(base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]), msg)


def test_rsa_pss_headers_verify():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    signer = KalshiSigner("k", key)
    h = signer.headers("POST", "/trade-api/v2/portfolio/events/orders", ts_ms=1)
    key.public_key().verify(
        base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]),
        b"1POST/trade-api/v2/portfolio/events/orders",
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


def test_load_private_key_from_pem(tmp_path):
    key = ed25519.Ed25519PrivateKey.generate()
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption())
    p = tmp_path / "k.pem"
    p.write_bytes(pem)
    assert isinstance(load_private_key(p), ed25519.Ed25519PrivateKey)


def test_signer_requires_key_id():
    with pytest.raises(ValueError):
        KalshiSigner("", ed25519.Ed25519PrivateKey.generate())
