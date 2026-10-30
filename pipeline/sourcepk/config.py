"""Settings and the source registry.

The registry (registry/<country>.toml) lists every publisher, every series the
pipeline downloads and every indicator it publishes. It is the single place
where a number's origin is defined, and it is published on the site as the
source registry.
"""
from __future__ import annotations

import os
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.9 and 3.10
    import tomli as tomllib
from dataclasses import dataclass, field
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PIPELINE_DIR.parent
REGISTRY_DIR = PIPELINE_DIR / "registry"
DEFAULT_DATA_DIR = Path.home() / "SourcePakistan" / "data"

# Statuses an indicator moves through. Only "confirmed" indicators are published.
STATUSES = ("unmapped", "needs-confirmation", "confirmed", "retired")


@dataclass(frozen=True)
class Source:
    key: str
    publisher: str
    dataset: str
    adapter: str
    homepage: str
    terms: str = "to confirm"
    secret: str | None = None
    status: str = "active"


@dataclass(frozen=True)
class Series:
    key: str
    source: str
    code: str
    frequency: str  # daily | weekly | monthly | quarterly | annual
    description: str
    start: str = "2015-07-01"
    status: str = "needs-confirmation"
    note: str = ""
    unit: str = ""            # the unit the publisher states; a different one stops the download
    published_name: str = ""  # the publisher's own series name; a different one stops the download


@dataclass(frozen=True)
class Indicator:
    key: str
    label: str
    unit: str
    series: tuple[str, ...]
    transform: str = ""
    display: dict | None = None
    status: str = "needs-confirmation"
    range: tuple[float, float] | None = None
    jump_abs: float | None = None
    jump_rel: float | None = None
    max_age_days: int | None = None
    confirm: str = ""
    confirmed: str = ""       # when and against what the values were checked
    section: str = ""
    period_format: str = ""   # "" (from the period), "fytd" or "fy_end_year"
    value_scale: float = 1.0  # converts the publisher's unit to the page's (0.001: US$ million to billion)
    context: tuple[str, ...] = ()  # comparisons shown beside the figure: "previous", "year_ago"


@dataclass(frozen=True)
class Chart:
    """A published history drawn as a chart: one series, as published, over a stated window."""
    key: str
    label: str
    series: str
    transform: str            # recent@1, fy_months@1 or fy_years@1
    status: str = "needs-confirmation"
    confirmed: str = ""
    n: int = 24               # recent@1: number of periods
    years_back: int = 9       # fy_years@1: fiscal years before the current one
    note: str = ""


@dataclass
class Registry:
    country_code: str
    country_name: str
    fy_start_month: int
    sources: dict[str, Source] = field(default_factory=dict)
    series: dict[str, Series] = field(default_factory=dict)
    indicators: dict[str, Indicator] = field(default_factory=dict)
    identities: list[dict] = field(default_factory=list)
    cross_checks: list[dict] = field(default_factory=list)
    charts: dict[str, Chart] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Return a list of problems; an empty list means the registry is consistent."""
        problems = []
        for s in self.series.values():
            if s.source not in self.sources:
                problems.append(f"series {s.key}: unknown source {s.source}")
            if s.status not in STATUSES:
                problems.append(f"series {s.key}: unknown status {s.status}")
        for i in self.indicators.values():
            if i.status not in STATUSES:
                problems.append(f"indicator {i.key}: unknown status {i.status}")
            for sk in i.series:
                if sk not in self.series:
                    problems.append(f"indicator {i.key}: unknown series {sk}")
            for c in i.context:
                if c not in ("previous", "year_ago"):
                    problems.append(f"indicator {i.key}: unknown context '{c}'")
            if i.status not in ("unmapped", "retired"):
                from .transforms import TRANSFORMS
                if i.transform not in TRANSFORMS:
                    problems.append(f"indicator {i.key}: unknown calculation '{i.transform}'")
                for sk in i.series:
                    if sk in self.series and (self.series[sk].status == "unmapped" or not self.series[sk].code):
                        problems.append(f"indicator {i.key} is '{i.status}' but series {sk} has no code")
            if i.status == "confirmed":
                for sk in i.series:
                    if sk in self.series and self.series[sk].status != "confirmed":
                        problems.append(f"indicator {i.key} is confirmed but series {sk} is not")
        for ch in self.charts.values():
            if ch.series not in self.series:
                problems.append(f"chart {ch.key}: unknown series {ch.series}")
            elif ch.status == "confirmed" and self.series[ch.series].status != "confirmed":
                problems.append(f"chart {ch.key} is confirmed but series {ch.series} is not")
            from .transforms import CHART_TRANSFORMS
            if ch.transform not in CHART_TRANSFORMS:
                problems.append(f"chart {ch.key}: unknown calculation '{ch.transform}'")
        for c in self.cross_checks:
            for k in (c.get("a"), c.get("b")):
                if k not in self.indicators:
                    problems.append(f"cross check {c.get('name')}: unknown indicator {k}")
        for ident in self.identities:
            for k in [ident.get("lhs")] + [t.lstrip("+-") for t in ident.get("rhs", [])]:
                if k not in self.indicators:
                    problems.append(f"identity {ident.get('name')}: unknown indicator {k}")
        return problems


def load_registry(country: str = "pakistan", path: Path | None = None) -> Registry:
    path = Path(path) if path else REGISTRY_DIR / f"{country}.toml"
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    c = raw["country"]
    reg = Registry(country_code=c["code"], country_name=c["name"], fy_start_month=c.get("fy_start_month", 1))
    for key, s in raw.get("sources", {}).items():
        reg.sources[key] = Source(key=key, **s)
    for s in raw.get("series", []):
        if s["key"] in reg.series:
            raise ValueError(f"duplicate series {s['key']}")
        reg.series[s["key"]] = Series(**s)
    for i in raw.get("indicators", []):
        if i["key"] in reg.indicators:
            raise ValueError(f"duplicate indicator {i['key']}")
        i = dict(i)
        i["series"] = tuple(i["series"]) if isinstance(i["series"], list) else (i["series"],)
        if "range" in i:
            i["range"] = tuple(i["range"])
        if "context" in i:
            i["context"] = tuple(i["context"])
        reg.indicators[i["key"]] = Indicator(**i)
    for ch in raw.get("charts", []):
        if ch["key"] in reg.charts:
            raise ValueError(f"duplicate chart {ch['key']}")
        reg.charts[ch["key"]] = Chart(**ch)
    reg.identities = raw.get("identities", [])
    reg.cross_checks = raw.get("cross_checks", [])
    return reg


def data_dir(arg: str | None = None) -> Path:
    """The archive folder: outside the repo, never committed."""
    path = Path(arg or os.environ.get("SP_DATA_DIR") or DEFAULT_DATA_DIR).expanduser().resolve()
    if path == REPO_DIR or REPO_DIR in path.parents:
        raise SystemExit(f"The data folder must be outside the website repo: {path}")
    return path
