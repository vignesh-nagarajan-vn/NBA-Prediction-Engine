from decimal import Decimal

import pytest

from nba_engine.kalshi.fees import fee_per_contract, order_fee


def test_taker_100_at_50c():
    # 0.07 * 100 * 0.5 * 0.5 = 1.75
    assert order_fee("0.50", 100, maker=False) == Decimal("1.75")


def test_rounds_up_per_order():
    # 0.07 * 1 * 0.25 = 0.0175 -> rounds up to 2 cents
    assert order_fee("0.50", 1, maker=False) == Decimal("0.02")
    # so one-lot orders pay ~2c/contract vs 1.75c in a 100-lot
    assert fee_per_contract("0.50", 1, maker=False) > fee_per_contract("0.50", 100, maker=False)


def test_maker_fee():
    # 0.0175 * 100 * 0.25 = 0.4375 -> 0.44
    assert order_fee("0.50", 100, maker=True) == Decimal("0.44")
    assert order_fee("0.50", 100, maker=True, fee_type="quadratic") == Decimal("0.00")


def test_fee_symmetric_and_smaller_at_extremes():
    assert order_fee("0.10", 100, maker=False) == order_fee("0.90", 100, maker=False)
    assert order_fee("0.10", 100, maker=False) < order_fee("0.50", 100, maker=False)
    assert order_fee("1.00", 100, maker=False) == Decimal("0.00")


def test_fractional_counts_and_multiplier():
    assert order_fee("0.50", "10.50", maker=False) == Decimal("0.19")  # 0.18375 -> 0.19
    assert order_fee("0.50", 100, maker=False, multiplier=2) == Decimal("3.50")


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        order_fee("1.5", 1, maker=False)
    with pytest.raises(NotImplementedError):
        order_fee("0.5", 1, maker=False, fee_type="flat")
