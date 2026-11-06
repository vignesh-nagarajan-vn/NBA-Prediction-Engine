"""Kalshi trading fees.

Fee per order = ceil_to_cent(rate * multiplier * count * P * (1 - P)), where P is the
contract price in dollars. Rounding is per ORDER (per fill batch), so many tiny orders cost
more than one large order. Rates by series `fee_type` (read from GET /series/{ticker}):

- quadratic:                  taker 0.07, maker 0
- quadratic_with_maker_fees:  taker 0.07, maker 0.0175   (KXNBAGAME as of Nov 2026)

The rates are Kalshi's published schedule, not something the API returns, so re-check
them against the official fee schedule whenever Kalshi announces changes.
"""

from __future__ import annotations

from decimal import ROUND_CEILING, Decimal

TAKER_RATE = Decimal("0.07")
MAKER_RATE = Decimal("0.0175")
CENT = Decimal("0.01")

_RATES: dict[str, tuple[Decimal, Decimal]] = {
    # fee_type: (taker_rate, maker_rate)
    "quadratic": (TAKER_RATE, Decimal(0)),
    "quadratic_with_maker_fees": (TAKER_RATE, MAKER_RATE),
}


def _d(x: Decimal | str | float | int) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))


def order_fee(
    price: Decimal | str | float,
    count: Decimal | str | float | int,
    *,
    maker: bool,
    fee_type: str = "quadratic_with_maker_fees",
    multiplier: Decimal | str | float | int = 1,
) -> Decimal:
    """Fee in dollars for one order (or one fill batch) of `count` contracts at `price`."""
    p, c, m = _d(price), _d(count), _d(multiplier)
    if not Decimal(0) <= p <= Decimal(1):
        raise ValueError(f"price must be in [0, 1] dollars, got {p}")
    if c < 0:
        raise ValueError("count must be non-negative")
    if fee_type not in _RATES:
        raise NotImplementedError(f"fee_type {fee_type!r} not modeled; check Kalshi fee schedule")
    taker_rate, maker_rate = _RATES[fee_type]
    rate = maker_rate if maker else taker_rate
    raw = rate * m * c * p * (1 - p)
    return raw.quantize(CENT, rounding=ROUND_CEILING)


def fee_per_contract(price, count, **kw) -> Decimal:
    """Effective per-contract fee for an order of `count` contracts (includes rounding)."""
    c = _d(count)
    if c == 0:
        return Decimal(0)
    return order_fee(price, c, **kw) / c
