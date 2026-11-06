"""Buffered, append-only Parquet sink for recorded streams.

Layout: {root}/{stream}/date=YYYY-MM-DD/part-{epoch_ms}-{n}.parquet (UTC date of recv time).
Records must be flat dicts (nested data should be JSON-encoded strings by the caller).
Each flush writes new part files, so a crash loses at most one flush interval of data and
never corrupts files already written. DuckDB reads the whole tree with
`read_parquet('data/raw/<stream>/**/*.parquet', hive_partitioning=true)`.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)


def to_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), default=str)


class ParquetSink:
    def __init__(self, root: Path | str, max_buffer: int = 5000):
        self.root = Path(root)
        self.max_buffer = max_buffer
        self._buf: dict[str, list[dict]] = defaultdict(list)
        self._lock = threading.Lock()
        self._n = 0
        self.rows_written: dict[str, int] = defaultdict(int)

    def write(self, stream: str, record: dict) -> None:
        if "recv_ms" not in record:
            record = {"recv_ms": int(time.time() * 1000), **record}
        with self._lock:
            self._buf[stream].append(record)
            full = len(self._buf[stream]) >= self.max_buffer
        if full:
            self.flush(stream)

    def flush(self, stream: str | None = None) -> int:
        with self._lock:
            streams = [stream] if stream else list(self._buf)
            batches = {s: self._buf.pop(s, []) for s in streams}
        total = 0
        for s, rows in batches.items():
            if not rows:
                continue
            by_day: dict[str, list[dict]] = defaultdict(list)
            for r in rows:
                day = datetime.fromtimestamp(r["recv_ms"] / 1000, UTC).strftime("%Y-%m-%d")
                by_day[day].append(r)
            for day, day_rows in by_day.items():
                self._n += 1
                out = self.root / s / f"date={day}"
                out.mkdir(parents=True, exist_ok=True)
                path = out / f"part-{int(time.time() * 1000)}-{self._n:06d}.parquet"
                try:
                    table = pa.Table.from_pylist(day_rows)
                except (pa.ArrowInvalid, pa.ArrowTypeError):
                    # Heterogeneous rows: fall back to a single JSON column, never drop data.
                    table = pa.table({
                        "recv_ms": [r["recv_ms"] for r in day_rows],
                        "record_json": [to_json(r) for r in day_rows],
                    })
                pq.write_table(table, path, compression="zstd")
                total += len(day_rows)
                self.rows_written[s] += len(day_rows)
        return total
