"""Kalshi API request signing.

Kalshi signs `timestamp_ms + METHOD + path` (path includes the `/trade-api/...` prefix and
excludes the query string). Ed25519 keys sign the bytes directly; RSA keys use RSA-PSS with
SHA-256, MGF1(SHA-256) and salt length equal to the digest length. Kalshi picks the verifier
from the registered public key, so the algorithm follows whatever key type is loaded.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from urllib.parse import urlsplit

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

PrivateKey = ed25519.Ed25519PrivateKey | rsa.RSAPrivateKey


def load_private_key(path: Path | str, password: bytes | None = None) -> PrivateKey:
    data = Path(path).read_bytes()
    key = serialization.load_pem_private_key(data, password=password)
    if not isinstance(key, ed25519.Ed25519PrivateKey | rsa.RSAPrivateKey):
        raise ValueError(f"Unsupported Kalshi key type: {type(key).__name__}")
    return key


def sign_message(key: PrivateKey, message: bytes) -> bytes:
    if isinstance(key, ed25519.Ed25519PrivateKey):
        return key.sign(message)
    return key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


def signing_path(url_or_path: str) -> str:
    """Path component to sign: no scheme/host, no query string."""
    return urlsplit(url_or_path).path


class KalshiSigner:
    def __init__(self, key_id: str, key: PrivateKey):
        if not key_id:
            raise ValueError("key_id is required")
        self.key_id = key_id
        self._key = key

    @classmethod
    def from_file(cls, key_id: str, path: Path | str) -> KalshiSigner:
        return cls(key_id, load_private_key(path))

    def headers(self, method: str, url_or_path: str, ts_ms: int | None = None) -> dict[str, str]:
        ts = str(ts_ms if ts_ms is not None else int(time.time() * 1000))
        msg = (ts + method.upper() + signing_path(url_or_path)).encode()
        sig = base64.b64encode(sign_message(self._key, msg)).decode()
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": sig,
        }
