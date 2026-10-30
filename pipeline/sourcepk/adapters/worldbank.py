"""World Bank Indicators API (World Development Indicators). No key needed.

    GET https://api.worldbank.org/v2/country/<ISO3>/indicator/<code>?format=json&per_page=200

The response is a JSON array: [page metadata, [observations]].
"""
from __future__ import annotations

import json

from . import FetchResult, FormatError, Observation, parse_number

BASE = "https://api.worldbank.org/v2/country/{country}/indicator/{code}?format=json&per_page=200"


def parse(body: bytes, code: str) -> tuple[list[Observation], dict]:
    data = json.loads(body)
    if not isinstance(data, list) or len(data) < 2 or not isinstance(data[1], list):
        raise FormatError(f"{code}: unexpected World Bank response")
    page = data[0] or {}
    if page.get("pages", 1) > 1:
        raise FormatError(f"{code}: response has {page['pages']} pages; raise per_page")
    obs = []
    for row in data[1]:
        if (row.get("indicator") or {}).get("id") != code:
            raise FormatError(f"{code}: response contains a different indicator")
        obs.append(Observation(period=str(row["date"]), value=parse_number(row.get("value")),
                               meta={"obs_status": row.get("obs_status", "")}))
    meta = {"lastupdated": page.get("lastupdated", ""), "source_id": page.get("sourceid", "")}
    return sorted(obs, key=lambda o: o.period), meta


def fetch(series, source, archive, fetcher, country: str = "PAK", **_) -> FetchResult:
    resp = fetcher.get(BASE.format(country=country, code=series.code))
    if resp.status != 200:
        raise FormatError(f"{series.code}: HTTP {resp.status} from {resp.url}")
    raw = archive.store(source.key, f"{series.code}.json", resp.body, resp.url,
                        extra={"series": series.code, "content_type": resp.content_type})
    obs, meta = parse(resp.body, series.code)
    return FetchResult(raw=raw, observations=obs, meta=meta)
