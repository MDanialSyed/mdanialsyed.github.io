"""State Bank of Pakistan EasyData API.

    GET https://easydata.sbp.org.pk/api/v1/series/<series code>/data
        ?api_key=<key>&start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&format=csv

The response is a CSV with one row per observation, including the columns
'Observation Date' and 'Observation Value'. The key is redacted before the
address is stored anywhere.
"""
from __future__ import annotations

import csv
import io
from datetime import date
from urllib.parse import quote

from .. import periods
from ..secrets import get_secret
from . import FetchResult, FormatError, Observation, parse_number

BASE = "https://easydata.sbp.org.pk/api/v1/series/{code}/data"
DATE_COL = "observation date"
VALUE_COL = "observation value"
CODE_COLS = ("series key", "series code", "series id")


def build_url(code: str, start: str, end: str, key: str) -> str:
    return (BASE.format(code=quote(code, safe=".")) +
            f"?api_key={key}&start_date={start}&end_date={end}&format=csv")


def parse(body: bytes, code: str, frequency: str) -> tuple[list[Observation], dict]:
    text = body.decode("utf-8-sig", errors="strict")
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if any(c.strip() for c in r)]
    if not rows:
        raise FormatError(f"{code}: empty response")
    header = [h.strip().lower() for h in rows[0]]
    if DATE_COL not in header or VALUE_COL not in header:
        raise FormatError(f"{code}: expected columns 'Observation Date' and 'Observation Value', got {rows[0]}")
    di, vi = header.index(DATE_COL), header.index(VALUE_COL)
    code_i = next((header.index(c) for c in CODE_COLS if c in header), None)
    obs, meta = [], {}
    for r in rows[1:]:
        if len(r) <= max(di, vi):
            raise FormatError(f"{code}: short row {r}")
        if code_i is not None and r[code_i].strip() and r[code_i].strip() != code:
            raise FormatError(f"{code}: response contains a different series ({r[code_i].strip()})")
        d = periods.parse_date(r[di])
        extra = {rows[0][i].strip(): r[i].strip() for i in range(len(r)) if i not in (di, vi) and i < len(rows[0])}
        if not meta:
            meta = extra
        obs.append(Observation(period=periods.to_period(d, frequency), value=parse_number(r[vi]),
                               meta={"date": d.isoformat()}))
    return obs, meta


def fetch(series, source, archive, fetcher, start: str | None = None, end: str | None = None) -> FetchResult:
    start = start or series.start
    end = end or date.today().isoformat()
    key = get_secret(source.secret or "easydata")
    resp = fetcher.get(build_url(series.code, start, end, key))
    if resp.status != 200:
        raise FormatError(f"{series.code}: HTTP {resp.status} from {resp.url}")
    raw = archive.store(source.key, f"{series.code}.csv", resp.body, resp.url,
                        extra={"series": series.code, "start_date": start, "end_date": end,
                               "content_type": resp.content_type})
    obs, meta = parse(resp.body, series.code, series.frequency)
    return FetchResult(raw=raw, observations=obs, meta=meta)
