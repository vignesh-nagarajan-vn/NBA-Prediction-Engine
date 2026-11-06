import numpy as np
import pandas as pd
import pytest

from nba_engine.data.teams import TEAMS
from nba_engine.features.market import american_to_prob, closing_lines, kalshi_pretip
from nba_engine.features.players import PlayerRatings
from nba_engine.features.ratings import elo_ratings, margin_ratings
from nba_engine.features.schedule_context import haversine_km, schedule_features
from nba_engine.history.espn_backfill import parse_odds, parse_player_box
from nba_engine.models.metrics import calibration, log_loss, summarize

CODES = [t.code for t in TEAMS]


def synthetic_games(n_days=300, strengths=None, hca=3.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    strengths = strengths if strengths is not None else {c: 0.0 for c in CODES}
    rows, gid = [], 0
    start = pd.Timestamp("2020-10-20T23:00Z")
    for d in range(n_days):
        teams = rng.permutation(CODES)
        for h, a in zip(teams[:5], teams[5:10], strict=True):
            m = strengths[h] - strengths[a] + hca + rng.normal(0, noise)
            hs = 100 + int(round(m / 2)) if m != 0 else 101
            as_ = 100 - int(round(m / 2))
            if hs == as_:
                hs += 1
            rows.append({"game_id": str(gid), "season": 2021, "season_type": "regular",
                         "tip_utc": start + pd.Timedelta(days=d), "home": h, "away": a,
                         "home_score": hs, "away_score": as_, "completed": True,
                         "neutral": False})
            gid += 1
    return pd.DataFrame(rows)


def test_haversine_known_distance():
    # Boston to Los Angeles ~ 4,170 km
    assert 4100 < haversine_km((42.366, -71.062), (34.043, -118.267)) < 4250


def test_schedule_features_rest_b2b_travel():
    t = pd.Timestamp
    games = pd.DataFrame([
        {"game_id": "1", "tip_utc": t("2024-11-01T23:30Z"), "home": "BOS", "away": "NYK",
         "neutral": False},
        {"game_id": "2", "tip_utc": t("2024-11-03T02:30Z"), "home": "LAL", "away": "BOS",
         "neutral": False},  # Nov 2 ET: BOS on a back-to-back, flew to LA
    ])
    f = schedule_features(games).set_index("game_id")
    assert f.loc["1", "home_rest"] == 7  # first game
    assert f.loc["2", "away_rest"] == 1 and f.loc["2", "away_b2b"] == 1
    assert f.loc["2", "away_travel_km"] > 4000
    assert f.loc["2", "away_tz_shift"] == -3


def test_elo_is_pregame_and_zero_sum():
    g = synthetic_games(n_days=5, hca=0.0)
    e = elo_ratings(g)
    assert (e.iloc[:5][["elo_home", "elo_away"]] == 1505).all().all()  # no info yet
    last = e.iloc[-1]
    assert np.isfinite(last["elo_diff"])


def test_margin_ratings_recover_truth_without_leakage():
    rng = np.random.default_rng(1)
    truth = {c: float(rng.normal(0, 5)) for c in CODES}
    mean = np.mean(list(truth.values()))
    truth = {c: v - mean for c, v in truth.items()}
    g = synthetic_games(n_days=250, strengths=truth, hca=3.0)
    r = margin_ratings(g, half_life_days=1000, ridge=0.01)
    assert r.iloc[0]["rating_margin_pred"] == 0.0  # first game sees nothing
    last = r.iloc[-1]
    assert abs(last["rating_hca"] - 3.0) < 0.3
    row = g.iloc[-1]
    expect = truth[row.home] - truth[row.away] + 3.0
    assert abs(last["rating_margin_pred"] - expect) < 0.5


def test_player_ratings_recover_a_star():
    """One team has a +10 star who sits some games; the model should find him and the
    team's pregame strength should drop when he is out."""
    rng = np.random.default_rng(2)
    games, box = [], []
    start = pd.Timestamp("2021-01-01T23:00Z")
    for d in range(400):
        h, a = ("BOS", "NYK") if d % 2 == 0 else ("NYK", "BOS")
        star_plays = rng.random() > 0.3
        margin = 2.0 + (10 * 36 / 48 if star_plays else 0) * (1 if h == "BOS" else -1)
        hs, as_ = 100 + round(margin), 100
        if hs == as_:
            hs += 1
        gid = str(d)
        games.append({"game_id": gid, "season": 2021, "season_type": "regular",
                      "tip_utc": start + pd.Timedelta(days=d), "home": h, "away": a,
                      "home_score": hs, "away_score": as_, "completed": True,
                      "neutral": False})
        for team in ("BOS", "NYK"):
            roster = [f"{team}{i}" for i in range(8)]
            for p in roster:
                is_star = p == "BOS0"
                if is_star and not star_plays:
                    box.append({"game_id": gid, "team": team, "player_id": p, "player": p,
                                "did_not_play": True, "reason": "ANKLE", "minutes": 0.0})
                    continue
                mins = 36.0 if is_star else 30.0
                box.append({"game_id": gid, "team": team, "player_id": p, "player": p,
                            "did_not_play": False, "reason": "", "minutes": mins})
    games, box = pd.DataFrame(games), pd.DataFrame(box)
    pr = PlayerRatings(half_life_days=10_000, ridge=1.0, solve_every_days=1)
    feats = pr.run(games, box).set_index("game_id")
    top = pr.top_players(box, k=1)
    assert top.iloc[0]["player"] == "BOS0"
    late = games[games["tip_utc"] > start + pd.Timedelta(days=300)]
    bos_home = late[late["home"] == "BOS"]
    with_star = [g for g in bos_home["game_id"]
                 if not box[(box.game_id == g) & (box.player_id == "BOS0")].did_not_play.iloc[0]]
    without = [g for g in bos_home["game_id"] if g not in with_star]
    assert (feats.loc[with_star, "home_player_strength"].mean()
            > feats.loc[without, "home_player_strength"].mean() + 3)
    assert (feats.loc[without, "home_out_regulars"] == 1).all()


def test_closing_lines_and_devig():
    p = american_to_prob(pd.Series([-150, 130]))
    assert abs(p[0] - 0.6) < 1e-9 and abs(p[1] - 100 / 230) < 1e-9
    odds = pd.DataFrame([
        {"game_id": "1", "provider": "a", "spread": -5.0, "home_ml": -200, "away_ml": 170},
        {"game_id": "1", "provider": "b", "spread": -6.0, "home_ml": None, "away_ml": None},
    ])
    cl = closing_lines(odds).iloc[0]
    assert cl["home_spread"] == -5.5
    raw_h, raw_a = 200 / 300, 100 / 270
    assert abs(cl["ml_p_home"] - raw_h / (raw_h + raw_a)) < 1e-9


def test_kalshi_pretip_uses_last_quote_before_tip():
    games = pd.DataFrame([{"game_id": "g", "tip_utc": pd.Timestamp("2026-11-06T02:00Z"),
                           "home": "PHX", "away": "ORL"}])
    tip = int(games.tip_utc[0].timestamp())
    markets = pd.DataFrame([
        {"ticker": "KXNBAGAME-26NOV05ORLPHX-PHX", "event_ticker": "KXNBAGAME-26NOV05ORLPHX",
         "team": "PHX"},
        {"ticker": "KXNBAGAME-26NOV05ORLPHX-ORL", "event_ticker": "KXNBAGAME-26NOV05ORLPHX",
         "team": "ORL"},
    ])
    candles = pd.DataFrame([
        {"ticker": "KXNBAGAME-26NOV05ORLPHX-PHX", "end_ts": tip - 120, "yes_bid": 0.50,
         "yes_ask": 0.52, "last": 0.51},
        {"ticker": "KXNBAGAME-26NOV05ORLPHX-PHX", "end_ts": tip + 60, "yes_bid": 0.90,
         "yes_ask": 0.92, "last": 0.91},  # after tip: must be ignored
        {"ticker": "KXNBAGAME-26NOV05ORLPHX-ORL", "end_ts": tip - 60, "yes_bid": 0.47,
         "yes_ask": 0.49, "last": 0.48},
    ])
    k = kalshi_pretip(markets, candles, games).iloc[0]
    assert abs(k["kalshi_p_home"] - (0.51 + (1 - 0.48)) / 2) < 1e-9
    assert abs(k["kalshi_spread"] - 0.02) < 1e-9


def test_metrics_calibration():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.random(20000) < p).astype(int)
    a, b = calibration(y, p)
    assert abs(b - 1) < 0.08 and abs(a) < 0.05
    s = summarize(y, p)
    assert s["log_loss"] < log_loss(y, np.full_like(p, 0.5))


def test_espn_parsers():
    summary = {
        "boxscore": {"players": [{"team": {"abbreviation": "GS"}, "statistics": [{
            "keys": ["minutes", "points", "fieldGoalsMade-fieldGoalsAttempted",
                     "freeThrowsMade-freeThrowsAttempted", "plusMinus"],
            "athletes": [
                {"athlete": {"id": "1", "displayName": "A"}, "starter": True,
                 "stats": ["34", "30", "11-20", "4-5", "+7"]},
                {"athlete": {"id": "2", "displayName": "B"}, "didNotPlay": True,
                 "reason": "LEFT KNEE SORENESS", "stats": []},
            ]}]}]},
        "pickcenter": [{"provider": {"name": "DraftKings", "priority": 1}, "spread": -2.5,
                        "overUnder": 219.5, "homeTeamOdds": {"moneyLine": -142},
                        "awayTeamOdds": {"moneyLine": 120}}],
    }
    pb = parse_player_box("9", summary)
    assert pb[0]["team"] == "GSW" and pb[0]["minutes"] == 34 and pb[0]["plus_minus"] == 7
    assert pb[0]["fga"] == 20 and pb[1]["did_not_play"] and pb[1]["minutes"] == 0
    od = parse_odds("9", summary)[0]
    assert od["spread"] == -2.5 and od["home_ml"] == -142


@pytest.mark.parametrize("bad", [None, "", "--", "abc"])
def test_minutes_parser_tolerates_junk(bad):
    from nba_engine.history.espn_backfill import _minutes
    assert _minutes(bad) == 0.0
