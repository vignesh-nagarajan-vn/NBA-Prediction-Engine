"""Local Kalshi order book.

Kalshi books hold BIDS ONLY on each side: a YES bid at p is equivalent to a NO ask at 1 - p.
So:  best YES ask = 1 - best NO bid,  best NO ask = 1 - best YES bid.
Prices are dollars (Decimal, 1-cent ticks for KXNBAGAME); sizes are fixed-point contracts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal

ONE = Decimal(1)
ZERO = Decimal(0)


def _levels(raw: Iterable[Iterable[str]] | None) -> dict[Decimal, Decimal]:
    out: dict[Decimal, Decimal] = {}
    for price, size in raw or ():
        s = Decimal(size)
        if s > 0:
            out[Decimal(price)] = s
    return out


@dataclass
class OrderBook:
    ticker: str
    yes: dict[Decimal, Decimal] = field(default_factory=dict)  # YES bids: price -> size
    no: dict[Decimal, Decimal] = field(default_factory=dict)  # NO bids: price -> size
    clock: int = 0
    last_update_ms: int = 0

    # ---- updates -------------------------------------------------------------------------
    def apply_snapshot(self, yes_levels, no_levels, clock: int = 0, ts_ms: int = 0) -> None:
        self.yes = _levels(yes_levels)
        self.no = _levels(no_levels)
        self.clock = clock
        self.last_update_ms = ts_ms

    def apply_delta(self, side: str, price: str | Decimal, delta: str | Decimal,
                    clock: int = 0, ts_ms: int = 0) -> None:
        book = self.yes if side == "yes" else self.no if side == "no" else None
        if book is None:
            raise ValueError(f"unknown side {side!r}")
        p, d = Decimal(price), Decimal(delta)
        new = book.get(p, ZERO) + d
        if new > 0:
            book[p] = new
        else:
            # Negative would mean we missed a message; drop the level and let the caller's
            # sequence check trigger a resync.
            book.pop(p, None)
        self.clock = max(self.clock, clock)
        self.last_update_ms = ts_ms or self.last_update_ms

    @classmethod
    def from_rest(cls, ticker: str, orderbook_fp: dict) -> OrderBook:
        ob = cls(ticker)
        ob.apply_snapshot(orderbook_fp.get("yes_dollars"), orderbook_fp.get("no_dollars"))
        return ob

    # ---- queries -------------------------------------------------------------------------
    def best_yes_bid(self) -> Decimal | None:
        return max(self.yes) if self.yes else None

    def best_no_bid(self) -> Decimal | None:
        return max(self.no) if self.no else None

    def best_yes_ask(self) -> Decimal | None:
        b = self.best_no_bid()
        return ONE - b if b is not None else None

    def best_no_ask(self) -> Decimal | None:
        b = self.best_yes_bid()
        return ONE - b if b is not None else None

    def mid(self) -> Decimal | None:
        bid, ask = self.best_yes_bid(), self.best_yes_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    def spread(self) -> Decimal | None:
        bid, ask = self.best_yes_bid(), self.best_yes_ask()
        return None if bid is None or ask is None else ask - bid

    def asks(self, side: str) -> list[tuple[Decimal, Decimal]]:
        """Ask ladder (price, size) for buying `side`, best (cheapest) first."""
        opposite = self.no if side == "yes" else self.yes
        return sorted(((ONE - p, s) for p, s in opposite.items()), key=lambda x: x[0])

    def bids(self, side: str) -> list[tuple[Decimal, Decimal]]:
        book = self.yes if side == "yes" else self.no
        return sorted(book.items(), key=lambda x: x[0], reverse=True)

    def cost_to_buy(self, side: str, count: Decimal | int) -> tuple[Decimal, Decimal] | None:
        """Walk the ask ladder. Returns (filled_count, avg_price) or None if the book is empty."""
        want = Decimal(count)
        filled, cost = ZERO, ZERO
        for price, size in self.asks(side):
            take = min(size, want - filled)
            filled += take
            cost += take * price
            if filled >= want:
                break
        return (filled, cost / filled) if filled > 0 else None

    def is_crossed(self) -> bool:
        bid, ask = self.best_yes_bid(), self.best_yes_ask()
        return bid is not None and ask is not None and bid >= ask

    def top(self) -> dict:
        return {
            "ticker": self.ticker,
            "yes_bid": self.best_yes_bid(),
            "yes_ask": self.best_yes_ask(),
            "yes_bid_size": self.yes.get(self.best_yes_bid(), None) if self.yes else None,
            "yes_ask_size": self.no.get(self.best_no_bid(), None) if self.no else None,
            "clock": self.clock,
        }

    def to_levels(self) -> dict[str, list[list[str]]]:
        """Serializable form matching Kalshi's [price, size] string pairs, ascending."""
        def ser(book):
            return [[f"{p:.4f}", f"{s:.2f}"] for p, s in sorted(book.items())]
        return {"yes_dollars": ser(self.yes), "no_dollars": ser(self.no)}
