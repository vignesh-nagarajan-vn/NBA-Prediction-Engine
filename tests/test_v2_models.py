import numpy as np
import pandas as pd

from nba_engine.data.teams import CONFERENCE, TEAMS
from nba_engine.features.efficiency import efficiency_ratings
from nba_engine.history.pbp import elapsed_s, game_stints
from nba_engine.models.gbm import GBMRegressor
from nba_engine.models.ingame import stern_prob
from nba_engine.predict.lineups import projected_box
from nba_engine.predict.season import SimConfig, simulate_season

CODES = [t.code for t in TEAMS]


def test_gbm_learns_interaction():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6000, 5))
    y = 2 * X[:, 0] + 3 * (X[:, 1] > 0) * X[:, 2] + rng.normal(0, 0.5, 6000)
    m = GBMRegressor(n_estimators=200, min_samples_leaf=40).fit(X[:5000], y[:5000])
    rmse = np.sqrt(np.mean((m.predict(X[5000:]) - y[5000:]) ** 2))
    linear_only = np.sqrt(np.mean((2 * X[5000:, 0] - y[5000:]) ** 2))
    assert rmse < 0.75 * linear_only
    assert m.feature_gain(5)[3:].sum() < 0.2  # noise features rarely used


def test_efficiency_ratings_find_the_good_offense():
    rows, box = [], []
    start = pd.Timestamp("2021-01-01T23:00Z")
    rng = np.random.default_rng(1)
    for d in range(120):
        teams = rng.permutation(CODES)
        for h, a in zip(teams[:5], teams[5:10], strict=True):
            gid = f"{d}-{h}"
            hs = 120 if h == "BOS" else 100
            as_ = 120 if a == "BOS" else 100
            if hs == as_:
                hs += 1
            rows.append({"game_id": gid, "tip_utc": start + pd.Timedelta(days=d), "home": h,
                         "away": a, "home_score": hs, "away_score": as_, "neutral": False,
                         "completed": True})
            for t in (h, a):  # exactly 100 possessions each
                box.append({"game_id": gid, "team": t, "fga": 85, "fta": 0, "oreb": 0,
                            "turnovers": 15, "points": 0, "minutes": 240})
    games, box = pd.DataFrame(rows), pd.DataFrame(box)
    e = efficiency_ratings(games, box, half_life_days=1000, ridge=0.1)
    last = e.iloc[-50:]
    bos_home = last.merge(games, on="game_id").query("home == 'BOS'")
    assert len(bos_home) == 0 or (bos_home["ortg_home"] > 112).all()
    # pace = possessions per team per game, 100 in this synthetic league
    assert 95 < e["pace_pred"].iloc[-1] < 105


def test_game_stints_tracks_subs_and_infers_period_starters():
    home = [f"h{i}" for i in range(7)]
    away = [f"a{i}" for i in range(5)]
    box = pd.DataFrame(
        [{"team": "BOS", "player_id": p, "starter": i < 5} for i, p in enumerate(home)]
        + [{"team": "NYK", "player_id": p, "starter": True} for p in away])

    def play(seq, period, clock, hs, as_, typ, team=None, p1=None, p2=None, scoring=False,
             val=0):
        return {"seq": seq, "period": period, "clock_s": clock,
                "elapsed_s": elapsed_s(period, clock), "home_score": hs, "away_score": as_,
                "type": typ, "team": team, "p1": p1, "p2": p2, "p3": None,
                "scoring": scoring, "score_value": val}

    plays = pd.DataFrame([
        play(1, 1, 700, 2, 0, "Jump Shot", "BOS", "h0", scoring=True, val=2),
        play(2, 1, 600, 2, 0, "Substitution", "BOS", "h5", "h0"),
        play(3, 1, 100, 5, 3, "Jump Shot", "BOS", "h5", scoring=True, val=3),
        # Period 2: h6 starts without a logged sub (h1 sat at the break)
        play(4, 2, 700, 7, 3, "Jump Shot", "BOS", "h6", scoring=True, val=2),
        play(5, 2, 690, 7, 3, "Defensive Rebound", "BOS", "h2"),
        play(6, 2, 680, 7, 3, "Defensive Rebound", "BOS", "h3"),
        play(7, 2, 670, 7, 3, "Defensive Rebound", "BOS", "h4"),
        play(8, 2, 660, 7, 3, "Defensive Rebound", "BOS", "h5"),
    ])
    st = pd.DataFrame(game_stints("g", plays, box, "BOS", "NYK"))
    assert len(st) == 3 and st["valid"].all()
    first, second, third = st.iloc[0], st.iloc[1], st.iloc[2]
    assert first["dur_s"] == 120 and first["home_pts"] == 2
    assert set(second[["h0", "h1", "h2", "h3", "h4"]]) == {"h5", "h1", "h2", "h3", "h4"}
    assert second["home_pts"] == 3 and second["away_pts"] == 3
    assert set(third[["h0", "h1", "h2", "h3", "h4"]]) == {"h6", "h2", "h3", "h4", "h5"}


def test_stern_prob_behaves():
    assert stern_prob(0, 2880, 0) == 0.5
    assert stern_prob(5, 60, 0) > stern_prob(5, 2000, 0) > 0.5
    assert stern_prob(0, 2880, 6) > 0.6
    assert stern_prob(3, 0, -10) == 1 - 1e-4
    assert stern_prob(-1, 0, 10) == 1e-4


def test_season_sim_probabilities_are_consistent():
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(40):
        teams = rng.permutation(CODES)
        for h, a in zip(teams[:15], teams[15:], strict=True):
            rows.append({"home": h, "away": a, "neutral": False,
                         "mu": 2.5 + (5 if h == "BOS" else 0) - (5 if a == "BOS" else 0)})
    rem = pd.DataFrame(rows)
    t = simulate_season(rem, {}, {}, SimConfig(n_sims=2000, seed=1)).set_index("team")
    assert abs(t["p_title"].sum() - 1) < 1e-9
    assert abs(t["p_finals"].sum() - 2) < 1e-9
    for conf in ("East", "West"):
        ct = t[t["conference"] == conf]
        assert abs(ct["p_playoffs"].sum() - 8) < 1e-9
        assert abs(ct["p_top6"].sum() - 6) < 1e-9
        assert abs(ct["p_seed1"].sum() - 1) < 1e-9
    assert t.loc["BOS", "p_title"] == t["p_title"].max()
    assert t.loc["BOS", "proj_wins"] > t.loc["NYK", "proj_wins"]
    assert CONFERENCE["BOS"] == "East"


def test_projected_box_marks_injured_only_near_term():
    as_of = pd.Timestamp("2026-11-12T00:00Z")
    up = pd.DataFrame([
        {"game_id": "soon", "tip_utc": as_of + pd.Timedelta(days=1), "home": "BOS",
         "away": "NYK"},
        {"game_id": "later", "tip_utc": as_of + pd.Timedelta(days=30), "home": "BOS",
         "away": "NYK"},
    ])
    rosters = {"BOS": ["b1", "b2"], "NYK": ["n1"]}
    inj = pd.DataFrame([{"player_id": "b1", "status": "Out"},
                        {"player_id": "n1", "status": "Day-To-Day"}])
    box = projected_box(up, rosters, inj, as_of).set_index(["game_id", "player_id"])
    assert box.loc[("soon", "b1"), "reason"].startswith("OUT")
    assert box.loc[("later", "b1"), "reason"] == "COACH'S DECISION"
    assert box.loc[("soon", "n1"), "reason"] == "COACH'S DECISION"
