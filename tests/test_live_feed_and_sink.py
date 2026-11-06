import pyarrow.parquet as pq

from nba_engine.data.live_feed import PlayTracker, is_dead_ball, is_long_stoppage, parse_plays
from nba_engine.storage.sink import ParquetSink

SUMMARY = {"plays": [
    {"id": "1", "sequenceNumber": "4", "type": {"text": "Jumpball"}, "text": "jump",
     "awayScore": 0, "homeScore": 0, "period": {"number": 1}, "clock": {"displayValue": "12:00"},
     "scoringPlay": False, "scoreValue": 0, "team": {"id": "20"},
     "wallclock": "2026-11-05T00:13:15Z"},
    {"id": "2", "sequenceNumber": "70", "type": {"text": "Jump Shot"}, "text": "makes 3",
     "awayScore": 11, "homeScore": 17, "period": {"number": 1}, "clock": {"displayValue": "6:15"},
     "scoringPlay": True, "scoreValue": 3, "team": {"id": "8"},
     "wallclock": "2026-11-05T00:25:57Z"},
]}


def test_dead_ball_classification():
    for t in ["Full Timeout", "End Period", "Free Throw - 1 of 2", "Coach's Challenge (Stands)",
              "Ref-Initiated Review (Stands)", "Technical Foul", "Jumpball"]:
        assert is_dead_ball(t), t
    for t in ["Jump Shot", "Defensive Rebound", "Driving Layup Shot", "Lost Ball Turnover"]:
        assert not is_dead_ball(t), t
    assert is_long_stoppage("Full Timeout")
    assert not is_long_stoppage("Free Throw - 1 of 2")


def test_parse_and_dedupe_plays():
    plays = parse_plays("g1", SUMMARY, recv_ms=123)
    assert [p.play_id for p in plays] == ["1", "2"]
    assert plays[1].scoring and plays[1].home_score == 17
    tr = PlayTracker()
    assert len(tr.new_plays(plays)) == 2
    assert tr.new_plays(plays) == []


def test_sink_roundtrip_and_fallback(tmp_path):
    sink = ParquetSink(tmp_path)
    sink.write("s", {"recv_ms": 1762300000000, "a": 1, "b": "x"})
    sink.write("s", {"recv_ms": 1762300000001, "a": 2, "b": "y"})
    sink.write("mixed", {"recv_ms": 1762300000000, "v": 1})
    sink.write("mixed", {"recv_ms": 1762300000000, "v": "not-an-int"})
    assert sink.flush() == 4
    files = list((tmp_path / "s").rglob("*.parquet"))
    assert len(files) == 1 and "date=2025-11-04" in str(files[0])
    assert pq.read_table(files[0]).column("a").to_pylist() == [1, 2]
    mixed = pq.read_table(next((tmp_path / "mixed").rglob("*.parquet")))
    assert "record_json" in mixed.column_names
    assert sink.flush() == 0
