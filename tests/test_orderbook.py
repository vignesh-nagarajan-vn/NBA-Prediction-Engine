from decimal import Decimal as D

from nba_engine.kalshi.orderbook import OrderBook

# Shape of a real GET /markets/{ticker}/orderbook response (Nov 2026), trimmed.
REST_SAMPLE = {
    "no_dollars": [["0.4800", "287110.91"], ["0.4900", "406301.62"], ["0.5000", "731751.72"]],
    "yes_dollars": [["0.4700", "103791.52"], ["0.4800", "3343.35"], ["0.4900", "34215.00"]],
}


def test_from_rest_top_of_book():
    ob = OrderBook.from_rest("T", REST_SAMPLE)
    assert ob.best_yes_bid() == D("0.49")
    assert ob.best_yes_ask() == D("0.50")  # 1 - best NO bid (0.50)
    assert ob.best_no_ask() == D("0.51")
    assert ob.spread() == D("0.01")
    assert ob.mid() == D("0.495")
    assert not ob.is_crossed()


def test_deltas_add_and_remove_levels():
    ob = OrderBook("T")
    ob.apply_snapshot([["0.40", "10"]], [["0.55", "5"]], clock=1)
    ob.apply_delta("yes", "0.41", "7.5", clock=2)
    assert ob.best_yes_bid() == D("0.41")
    ob.apply_delta("yes", "0.41", "-7.5", clock=3)
    assert ob.best_yes_bid() == D("0.40")
    assert D("0.41") not in ob.yes
    ob.apply_delta("no", "0.55", "-5", clock=4)
    assert ob.best_yes_ask() is None
    assert ob.clock == 4


def test_cost_to_buy_walks_ladder():
    ob = OrderBook("T")
    # NO bids at 0.60 (10) and 0.58 (20) => YES asks at 0.40 x10, 0.42 x20
    ob.apply_snapshot([], [["0.60", "10"], ["0.58", "20"]])
    filled, avg = ob.cost_to_buy("yes", 20)
    assert filled == 20
    assert avg == (D("0.40") * 10 + D("0.42") * 10) / 20
    filled, _ = ob.cost_to_buy("yes", 100)
    assert filled == 30  # book only has 30
    assert OrderBook("E").cost_to_buy("yes", 1) is None


def test_to_levels_roundtrip():
    ob = OrderBook.from_rest("T", REST_SAMPLE)
    again = OrderBook.from_rest("T", ob.to_levels())
    assert again.yes == ob.yes and again.no == ob.no
