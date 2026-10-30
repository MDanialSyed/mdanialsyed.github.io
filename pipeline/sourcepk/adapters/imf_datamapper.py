"""IMF DataMapper API (World Economic Outlook series). No key needed.

    GET https://www.imf.org/external/datamapper/api/v1/<indicator>/<ISO3>

The response is JSON: {"values": {"<indicator>": {"<ISO3>": {"<year>": value}}}}.
Years after the current one are IMF projections and are marked as such.
"""
from __future__ import annotations

import json
from datetime import date

from . import FetchResult, FormatError, Observation, parse_number

BASE = "https://www.imf.org/external/datamapper/api/v1/{code}/{country}"


def parse(body: bytes, code: str, country: str, today: date | None = None) -> tuple[list[Observation], dict]:
    data = json.loads(body)
    try:
        values = data["values"][code][country]
    except (KeyError, TypeError):
        raise FormatError(f"{code}/{country}: unexpected IMF DataMapper response")
    this_year = (today or date.today()).year
    obs = [Observation(period=str(y), value=parse_number(v),
                       meta={"projection": int(y) >= this_year})
           for y, v in sorted(values.items())]
    return obs, {"api": data.get("api", {})}


def fetch(series, source, archive, fetcher, country: str = "PAK", **_) -> FetchResult:
    resp = fetcher.get(BASE.format(code=series.code, country=country))
    if resp.status != 200:
        raise FormatError(f"{series.code}: HTTP {resp.status} from {resp.url}")
    raw = archive.store(source.key, f"{series.code}_{country}.json", resp.body, resp.url,
                        extra={"series": series.code, "country": country, "content_type": resp.content_type})
    obs, meta = parse(resp.body, series.code, country)
    return FetchResult(raw=raw, observations=obs, meta=meta)
