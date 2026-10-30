"""Shared test helpers. All data here is synthetic and only exercises the code paths;
none of it is an official figure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sourcepk.web import Response, redact  # noqa: E402

TEST_KEY = "test-key-9f8e7d6c5b4a"


def months(start: str, n: int) -> list[str]:
    y, m = int(start[:4]), int(start[5:])
    out = []
    for _ in range(n):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def easydata_csv(code: str, values: dict[str, float], name: str = "Synthetic series") -> bytes:
    lines = ["Dataset Name,Series Key,Series Name,Observation Date,Observation Value,Unit,Observation Status"]
    for period, v in values.items():
        date = f"{period}-28" if len(period) == 7 else period
        lines.append(f"Synthetic,{code},{name},{date},{v},Index,Normal")
    return ("\n".join(lines) + "\n").encode()


def worldbank_json(code: str, values: dict[str, float | None], pages: int = 1) -> bytes:
    rows = [{"indicator": {"id": code, "value": "x"}, "country": {"id": "PK", "value": "Pakistan"},
             "countryiso3code": "PAK", "date": y, "value": v, "unit": "", "obs_status": "", "decimal": 0}
            for y, v in sorted(values.items(), reverse=True)]
    return json.dumps([{"page": 1, "pages": pages, "per_page": 200, "total": len(rows),
                        "sourceid": "2", "lastupdated": "2026-07-01"}, rows]).encode()


def imf_json(code: str, values: dict[str, float]) -> bytes:
    return json.dumps({"values": {code: {"PAK": values}}, "api": {"version": "1", "output-method": "json"}}).encode()


class FakeFetcher:
    """Serves prepared bodies by matching part of the address; records every address asked for."""

    def __init__(self, routes: dict[str, bytes]):
        self.routes = routes
        self.requested: list[str] = []

    def get(self, url: str, timeout: int = 60) -> Response:
        self.requested.append(url)
        for part, body in self.routes.items():
            if part in url:
                return Response(url=redact(url), status=200, content_type="text/plain", body=body)
        return Response(url=redact(url), status=404, content_type="text/plain", body=b"not found")


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("SP_EASYDATA_KEY", TEST_KEY)
    return TEST_KEY
