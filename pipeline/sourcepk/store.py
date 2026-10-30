"""Processed series with every vintage kept.

One CSV per series in <data>/series/<key>.csv, append-only:

    period,value,retrieved_at,raw_path,raw_sha256

A row is added when a period appears for the first time, or when the
publisher's value for a period changes (a revision). Old rows are never
edited or deleted, so the full revision history is always available.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from .adapters import Observation
from .archive import RawRef

COLUMNS = ["period", "value", "retrieved_at", "raw_path", "raw_sha256"]
TOLERANCE = 1e-9


@dataclass
class Row:
    period: str
    value: float
    retrieved_at: str
    raw_path: str
    raw_sha256: str


@dataclass
class Changes:
    new: list[str] = field(default_factory=list)
    revised: list[tuple[str, float, float]] = field(default_factory=list)


class Store:
    def __init__(self, root: Path):
        self.root = Path(root) / "series"

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.csv"

    def rows(self, key: str) -> list[Row]:
        path = self._path(key)
        if not path.exists():
            return []
        with open(path, newline="", encoding="utf-8") as f:
            return [Row(r["period"], float(r["value"]), r["retrieved_at"], r["raw_path"], r["raw_sha256"])
                    for r in csv.DictReader(f)]

    def latest(self, key: str) -> dict[str, Row]:
        """Latest vintage for each period, ordered by period."""
        out: dict[str, Row] = {}
        for r in self.rows(key):
            out[r.period] = r
        return dict(sorted(out.items()))

    def history(self, key: str, period: str) -> list[Row]:
        return [r for r in self.rows(key) if r.period == period]

    def ingest(self, key: str, observations: list[Observation], raw: RawRef) -> Changes:
        current = self.latest(key)
        changes, new_rows = Changes(), []
        for o in observations:
            if o.value is None:
                continue
            old = current.get(o.period)
            if old is None:
                changes.new.append(o.period)
            elif abs(old.value - o.value) > TOLERANCE:
                changes.revised.append((o.period, old.value, o.value))
            else:
                continue
            new_rows.append(Row(o.period, o.value, raw.retrieved_at, raw.path, raw.sha256))
        if new_rows:
            self.root.mkdir(parents=True, exist_ok=True)
            path = self._path(key)
            write_header = not path.exists()
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(COLUMNS)
                for r in new_rows:
                    w.writerow([r.period, repr(r.value), r.retrieved_at, r.raw_path, r.raw_sha256])
        return changes
