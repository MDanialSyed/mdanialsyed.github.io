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


EASYDATA_META = {  # the published name and unit each registry series expects
    "TS_GP_PT_CPI_M.P00011516": ("National CPI; an Inflation Measure (Year-on-Year basis)", "Percent"),
    "TS_GP_RL_LSM1516_M.LSM000160000": ("Quantum Index - Overall", "Index"),
    "TS_GP_BOP_BPM6SUM_M.P00010": ("BOP-BPM6-Current account balance", "Million USD"),
    "TS_GP_IR_REPOMR_D.ORR": ("Weighted-average Overnight Repo Rate", "Percent"),
}


def easydata_csv(code: str, values: dict[str, float], name: str | None = None, unit: str | None = None) -> bytes:
    """Shaped like EasyData's CSV: newest first, lines ended by a carriage return only."""
    n, u = EASYDATA_META.get(code, ("Synthetic series", "Index"))
    lines = ["Dataset Name,Series Key,Series Name,Observation Date,Observation Value,Unit,Observation Status,Status Comments"]
    for period, v in sorted(values.items(), reverse=True):
        date = f"{period}-28" if len(period) == 7 else period
        lines.append(f"Synthetic,{code},{name or n},{date},{v},{unit or u},Normal,")
    return ("\r".join(lines) + "\r").encode()


def worldbank_json(code: str, values: dict[str, float | None], pages: int = 1) -> bytes:
    rows = [{"indicator": {"id": code, "value": "x"}, "country": {"id": "PK", "value": "Pakistan"},
             "countryiso3code": "PAK", "date": y, "value": v, "unit": "", "obs_status": "", "decimal": 0}
            for y, v in sorted(values.items(), reverse=True)]
    return json.dumps([{"page": 1, "pages": pages, "per_page": 200, "total": len(rows),
                        "sourceid": "2", "lastupdated": "2026-07-01"}, rows]).encode()


def imf_json(code: str, values: dict[str, float]) -> bytes:
    return json.dumps({"values": {code: {"PAK": values}}, "api": {"version": "1", "output-method": "json"}}).encode()


def synthetic_history(code: str) -> dict[str, float]:
    """A flat history shaped by the frequency in the series code (_Y. annual, _W. weekly, _D./_AH. by date)."""
    from datetime import date, timedelta
    if "_Y." in code:
        return {f"{y}-06-30": 10.0 for y in range(2016, 2027)}
    if "_W." in code or "_D." in code or "_AH." in code:
        step = 7 if "_W." in code else 1
        start = date(2026, 1, 2)
        return {(start + timedelta(days=step * i)).isoformat(): 10.0 for i in range(40)}
    return {p: 10.0 for p in months("2024-07", 27)}


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
        if "easydata.sbp.org.pk/api/v1/series/" in url:  # any other State Bank series: a flat synthetic history
            code = url.split("/series/")[1].split("/")[0]
            body = easydata_csv(code, synthetic_history(code))
            return Response(url=redact(url), status=200, content_type="text/plain", body=body)
        return Response(url=redact(url), status=404, content_type="text/plain", body=b"not found")


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("SP_EASYDATA_KEY", TEST_KEY)
    return TEST_KEY
