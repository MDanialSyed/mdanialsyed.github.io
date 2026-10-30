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
from ..secrets import get_secret, scrub
from . import FetchResult, FormatError, Observation, parse_number

BASE = "https://easydata.sbp.org.pk/api/v1/series/{code}/data"
DATE_COL = "observation date"
VALUE_COL = "observation value"
CODE_COLS = ("series key", "series code", "series id")


def build_url(code: str, start: str, end: str, key: str) -> str:
    return (BASE.format(code=quote(code, safe=".")) +
            f"?api_key={key}&start_date={start}&end_date={end}&format=csv")


def parse(body: bytes, code: str, frequency: str, unit: str = "", name: str = "") -> tuple[list[Observation], dict]:
    """Rows of the CSV (EasyData ends lines with a carriage return only). If the registry states the
    series' unit or published name, every row must carry exactly that unit and name."""
    text = body.decode("utf-8-sig", errors="strict")
    reader = csv.reader(io.StringIO(text, newline=""))
    rows = [r for r in reader if any(c.strip() for c in r)]
    if not rows:
        raise FormatError(f"{code}: empty response")
    header = [h.strip().lower() for h in rows[0]]
    if DATE_COL not in header or VALUE_COL not in header:
        raise FormatError(f"{code}: expected columns 'Observation Date' and 'Observation Value', got {rows[0]}")
    di, vi = header.index(DATE_COL), header.index(VALUE_COL)
    code_i = next((header.index(c) for c in CODE_COLS if c in header), None)
    unit_i = header.index("unit") if "unit" in header else None
    name_i = header.index("series name") if "series name" in header else None
    if unit and unit_i is None:
        raise FormatError(f"{code}: the file states no unit; expected '{unit}'")
    if name and name_i is None:
        raise FormatError(f"{code}: the file states no series name; expected '{name}'")
    obs, meta = [], {}
    for r in rows[1:]:
        if len(r) <= max(di, vi):
            raise FormatError(f"{code}: short row {r}")
        if code_i is not None and r[code_i].strip() and r[code_i].strip() != code:
            raise FormatError(f"{code}: response contains a different series ({r[code_i].strip()})")
        if unit and r[unit_i].strip() != unit:
            raise FormatError(f"{code}: unit is '{r[unit_i].strip()}', expected '{unit}'")
        if name and r[name_i].strip() != name:
            raise FormatError(f"{code}: series is '{r[name_i].strip()}', expected '{name}'")
        d = periods.parse_date(r[di])
        extra = {rows[0][i].strip(): r[i].strip() for i in range(len(r)) if i not in (di, vi) and i < len(rows[0])}
        if not meta:
            meta = extra
        obs.append(Observation(period=periods.to_period(d, frequency), value=parse_number(r[vi]),
                               meta={"date": d.isoformat()}))
    seen = [o.period for o in obs]
    if len(seen) != len(set(seen)):
        raise FormatError(f"{code}: the file has more than one value for the same period")
    return sorted(obs, key=lambda o: o.period), meta


def _snippet(body: bytes, n: int = 240) -> str:
    """A short, single-line excerpt of an error response, with any key removed."""
    text = " ".join(body[:4000].decode("utf-8", errors="replace").split())
    return scrub(text[:n])


def probe(code: str = "TS_GP_PT_CPI_M.P00011516") -> dict:
    """Diagnose a refused request without revealing the key: the key's shape, and how the service
    answers with and without it, under the pipeline's own client name and the plain library one."""
    import requests
    from .. import __version__
    from ..web import USER_AGENT, redact
    from ..archive import now_utc
    out = {"checked_at": now_utc(),
           "pipeline_version": __version__, "attempts": []}
    try:
        key = get_secret("easydata")
        out["key_length"] = len(key)
        out["key_starts_with_letter"] = key[:1].isalpha()
        out["key_has_spaces"] = any(c.isspace() for c in key)
    except Exception as e:  # noqa: BLE001
        key = ""
        out["key_error"] = type(e).__name__
    end = date.today().isoformat()
    start = (date.today().replace(year=date.today().year - 1)).isoformat()
    tries = [("no key", build_url(code, start, end, "").replace("api_key=&", ""), USER_AGENT),
             ("key, pipeline client name", build_url(code, start, end, key), USER_AGENT),
             ("key, plain library client name", build_url(code, start, end, key), None),
             ("key, json", build_url(code, start, end, key).replace("format=csv", "format=json"), USER_AGENT)]
    for label, url, ua in tries:
        headers = {"User-Agent": ua} if ua else {}
        try:
            r = requests.get(url, headers=headers, timeout=60)
            out["attempts"].append({"try": label, "url": redact(url), "status": r.status_code,
                                    "content_type": r.headers.get("Content-Type", ""),
                                    "server": r.headers.get("Server", ""), "body": _snippet(r.content)})
        except requests.RequestException as e:
            out["attempts"].append({"try": label, "url": redact(url), "error": type(e).__name__})
    return out


def fetch(series, source, archive, fetcher, start: str | None = None, end: str | None = None) -> FetchResult:
    start = start or series.start
    end = end or date.today().isoformat()
    key = get_secret(source.secret or "easydata")
    resp = fetcher.get(build_url(series.code, start, end, key))
    if resp.status != 200:
        raise FormatError(f"{series.code}: HTTP {resp.status} from {resp.url}: {_snippet(resp.body)}")
    raw = archive.store(source.key, f"{series.code}.csv", resp.body, resp.url,
                        extra={"series": series.code, "start_date": start, "end_date": end,
                               "content_type": resp.content_type})
    obs, meta = parse(resp.body, series.code, series.frequency, series.unit, series.published_name)
    return FetchResult(raw=raw, observations=obs, meta=meta)
