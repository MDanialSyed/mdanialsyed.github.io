"""Fetch, compute, check and publish.

fetch()    downloads every mapped series, archives the files and updates the series store
compute()  turns series into indicator values with the registry's named calculations
publish()  writes confirmed values that passed every blocking check, with full lineage,
           to the website's _data/pakistan/pipeline.json, and the registry to registry.json
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from . import __version__
from .adapters import FormatError, get_adapter
from .archive import Archive
from .archive import sha256
from .checks import BLOCK, HOLD, Finding, check_cross, check_identities, check_indicator
from . import periods
from .periods import label as period_label
from .config import REPO_DIR, Registry
from .store import Changes, Store
from .transforms import CHART_TRANSFORMS, ChartResult, NotComputable, Result, compute, format_display
from .web import DownloadError, Fetcher

SITE_DATA = REPO_DIR / "_data" / "pakistan"


@dataclass
class RunLog:
    run_id: str
    fetched: dict[str, dict] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    changes: dict[str, Changes] = field(default_factory=dict)
    results: dict[str, Result] = field(default_factory=dict)
    context: dict[str, list[tuple[str, Result]]] = field(default_factory=dict)
    not_computed: dict[str, str] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    published: list[str] = field(default_factory=list)
    carried: list[str] = field(default_factory=list)
    archive_problems: list[str] = field(default_factory=list)
    fetch_run: str = ""
    charts: dict[str, ChartResult] = field(default_factory=dict)


def save_fetch_log(log: RunLog, data_root: Path) -> Path:
    """What a download-only run found, so a later build can include it in its report."""
    folder = Path(data_root) / "runs" / log.run_id
    folder.mkdir(parents=True, exist_ok=True)
    doc = {"run_id": log.run_id, "fetched": log.fetched, "errors": log.errors,
           "changes": {k: {"new": v.new, "revised": [list(r) for r in v.revised]} for k, v in log.changes.items()}}
    from .secrets import scrub
    (folder / "fetch.json").write_text(scrub(json.dumps(doc, indent=2, ensure_ascii=False)) + "\n", encoding="utf-8")
    (Path(data_root) / "runs" / "LATEST_FETCH").write_text(log.run_id + "\n", encoding="utf-8")
    return folder


def load_fetch_log(log: RunLog, data_root: Path) -> str | None:
    pointer = Path(data_root) / "runs" / "LATEST_FETCH"
    if not pointer.exists():
        return None
    run_id = pointer.read_text(encoding="utf-8").strip()
    path = Path(data_root) / "runs" / run_id / "fetch.json"
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    log.fetched, log.errors = doc["fetched"], doc["errors"]
    log.changes = {k: Changes(v["new"], [tuple(r) for r in v["revised"]]) for k, v in doc["changes"].items()}
    log.fetch_run = run_id
    return run_id


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def fetch(reg: Registry, data_root: Path, log: RunLog, only: list[str] | None = None,
          fetcher: Fetcher | None = None) -> None:
    archive, store, fetcher = Archive(data_root), Store(data_root), fetcher or Fetcher()
    from .adapters import sbp_files
    sbp_files._CACHE.clear()  # each run downloads each file afresh
    for s in reg.series.values():
        if only and s.key not in only:
            continue
        if s.status in ("unmapped", "retired") or not s.code:
            continue
        source = reg.sources[s.source]
        try:
            res = get_adapter(source.adapter).fetch(s, source, archive, fetcher)
        except (DownloadError, FormatError, NotImplementedError, ValueError) as e:
            log.errors.append(f"{s.key}: {e}")
            continue
        except Exception as e:  # a missing key or an unexpected failure must never stop the other series
            log.errors.append(f"{s.key}: {type(e).__name__}: {e}")
            continue
        log.fetched[s.key] = {"raw": res.raw.path, "sha256": res.raw.sha256, "url": res.raw.url,
                              "observations": len(res.observations), "meta": res.meta}
        log.changes[s.key] = store.ingest(s.key, res.observations, res.raw)


def compute_all(reg: Registry, data_root: Path, log: RunLog, today: date | None = None) -> None:
    store = Store(data_root)
    for ind in reg.indicators.values():
        if ind.status in ("unmapped", "retired") or not ind.transform:
            continue
        series_key = ind.series[0]
        series = store.latest(series_key)
        try:
            res = compute(ind.transform, series_key, series, fy_start_month=reg.fy_start_month, today=today)
        except NotComputable as e:
            log.not_computed[ind.key] = str(e)
            continue
        log.results[ind.key] = _scaled(ind, res)
        log.context[ind.key] = _context(reg, ind, series_key, series, res, today)


def _scaled(ind, res: Result) -> Result:
    if ind.value_scale == 1:
        return res
    return dataclasses.replace(res, value=res.value * ind.value_scale,
                               note="; ".join(x for x in (res.note, f"x {ind.value_scale:g} for the page's unit") if x))


def _context(reg, ind, series_key, series, res: Result, today) -> list[tuple[str, Result]]:
    """The same calculation for the previous period and a year earlier, from the same stored series."""
    out = []
    earlier = [p for p in series if p < res.period]
    for kind in ind.context:
        if kind == "previous":
            if not earlier:
                continue
            upto = earlier[-1] if len(res.period) == 10 else periods.shift(res.period, -1)
        else:
            if len(res.period) == 10:
                continue
            upto = periods.shift(res.period, -(4 if "-Q" in res.period else 12 if len(res.period) == 7 else 1))
        sub = {p: r for p, r in series.items() if p <= upto}
        try:
            r2 = compute(ind.transform, series_key, sub, fy_start_month=reg.fy_start_month, today=today)
        except NotComputable:
            continue
        if r2.period == upto:
            out.append((kind, _scaled(ind, r2)))
    return out


def context_text(reg, key: str, items: list[tuple[str, Result]], latest_period: str) -> str:
    ind = reg.indicators[key]
    parts = []
    for kind, r in items:
        shown = format_display(r.value, ind.display or {})
        if kind == "year_ago":
            parts.append(f"{shown} a year earlier")
        elif ind.period_format == "fy_end_year" and len(r.period) == 4:
            parts.append(f"{shown} in {period_label(r.period, 'fy_end_year')}")
        elif len(r.period) == 10:
            parts.append(f"{shown} on {period_label(r.period).rsplit(' ', 1)[0]}")
        elif len(r.period) == 7:
            lab = period_label(r.period)
            parts.append(f"{shown} in {lab.split()[0] if r.period[:4] == latest_period[:4] else lab}")
        else:
            parts.append(f"{shown} in {period_label(r.period)}")
    return "; ".join(parts)


def compute_charts(reg: Registry, data_root: Path, log: RunLog, today: date | None = None) -> None:
    store = Store(data_root)
    for ch in reg.charts.values():
        if ch.status == "retired":
            continue
        try:
            log.charts[ch.key] = CHART_TRANSFORMS[ch.transform](
                ch.series, store.latest(ch.series), n=ch.n, years_back=ch.years_back,
                fy_start_month=reg.fy_start_month, today=today)
        except NotComputable as e:
            log.not_computed[f"chart {ch.key}"] = str(e)


def check_chart_lineage(key: str, res: ChartResult, data_root: Path) -> list[Finding]:
    out, seen = [], set()
    for series_key, row in res.inputs:
        if (row.raw_path, row.raw_sha256) in seen:
            continue
        seen.add((row.raw_path, row.raw_sha256))
        path = data_root / row.raw_path
        if not (row.raw_path and row.raw_sha256) or not path.exists():
            out.append(Finding(BLOCK, f"chart {key}", "lineage", f"{series_key}: archived file missing ({row.raw_path})"))
        elif sha256(path.read_bytes()) != row.raw_sha256:
            out.append(Finding(BLOCK, f"chart {key}", "lineage", f"{series_key}: archived file changed ({row.raw_path})"))
    return out


def run_checks(reg: Registry, data_root: Path, log: RunLog, previous: dict, today: date) -> None:
    for key, res in log.charts.items():
        log.findings += check_chart_lineage(key, res, Path(data_root))
        if reg.charts[key].status != "confirmed":
            log.findings.append(Finding(HOLD, f"chart {key}", "status", f"not published: chart is '{reg.charts[key].status}'"))
    for key, res in log.results.items():
        log.findings += check_indicator(reg.indicators[key], res, data_root, previous.get(key), today)
    log.findings += check_identities(reg, log.results)
    log.findings += check_cross(reg, log.results)


def code_version() -> str:
    """The git commit of the pipeline code; copies made for the Mac carry it in CODE_VERSION."""
    stamp = REPO_DIR / "pipeline" / "CODE_VERSION"
    if not (REPO_DIR / ".git").exists():
        return stamp.read_text(encoding="utf-8").strip() if stamp.exists() else "unknown"
    try:
        sha = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO_DIR), "status", "--porcelain", "pipeline"],
                               capture_output=True, text=True).stdout.strip()
        return f"{sha}{'+local-changes' if dirty else ''}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _inputs(reg: Registry, res: Result) -> list[dict]:
    out = []
    for series_key, row in res.inputs:
        s = reg.series[series_key]
        src = reg.sources[s.source]
        out.append({"series": series_key, "code": s.code, "publisher": src.publisher,
                    "dataset": src.dataset, "period": row.period, "value": row.value,
                    "retrieved_at": row.retrieved_at, "file": row.raw_path, "sha256": row.raw_sha256})
    return out


def lineage_record(reg: Registry, key: str, res: Result, context: list | None = None) -> dict:
    ind = reg.indicators[key]
    inputs = _inputs(reg, res)
    context = context or []
    s0 = reg.series[ind.series[0]]
    src0 = reg.sources[s0.source]
    return {
        "key": key, "label": ind.label, "unit": ind.unit, "value": round(res.value, 6),
        "display": format_display(res.value, ind.display or {}), "period": res.period,
        "period_label": period_label(res.period, ind.period_format or
                                     ("fytd" if ind.transform.startswith("fytd") else ""), reg.fy_start_month),
        "status": "official", "frequency": s0.frequency, "publisher": src0.publisher, "dataset": src0.dataset,
        "series_code": s0.code, "source_url": src0.homepage, "calc": ind.transform,
        "calc_note": res.note, "retrieved_at": max(r.retrieved_at for _, r in res.inputs),
        "context": context_text(reg, key, context, res.period),
        "inputs": inputs,
        "context_items": [{"kind": kind, "value": round(r.value, 6),
                           "display": format_display(r.value, ind.display or {}), "period": r.period,
                           "inputs": _inputs(reg, r)} for kind, r in context],
    }


def load_interim(site_data: Path = SITE_DATA) -> dict:
    """The interim figures currently on the dashboard, for side-by-side comparison."""
    path = site_data / "indicators.json"
    if not path.exists():
        return {}
    return {r["key"]: r for r in json.loads(path.read_text(encoding="utf-8"))}


def chart_record(reg: Registry, key: str, res: ChartResult) -> dict:
    ch = reg.charts[key]
    s = reg.series[ch.series]
    src = reg.sources[s.source]
    files = {}
    for _, row in res.inputs:
        files.setdefault((row.raw_path, row.raw_sha256), row.retrieved_at)
    return {"key": key, "label": ch.label, "status": "official", "publisher": src.publisher,
            "dataset": src.dataset, "series_code": s.code, "series_name": s.published_name or s.description,
            "unit": s.unit, "source_url": src.homepage, "calc": ch.transform, "calc_note": res.note,
            "retrieved_at": max(r.retrieved_at for _, r in res.inputs), "points": res.points,
            "files": [{"file": f, "sha256": h, "retrieved_at": t} for (f, h), t in sorted(files.items())]}


def previous_charts(site_data: Path = SITE_DATA) -> dict:
    path = site_data / "pipeline.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("charts", {})


def load_previous(site_data: Path = SITE_DATA) -> dict:
    path = site_data / "pipeline.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("indicators", {})


def publish(reg: Registry, log: RunLog, site_data: Path = SITE_DATA) -> dict:
    """Write confirmed values that passed every blocking check. A confirmed indicator that could
    not be refreshed this run (download failure or a blocking check) keeps its last published
    record, marked as carried forward, so the page never shows a value that failed a check."""
    blocked = {f.indicator for f in log.findings if f.level == BLOCK}
    previous = load_previous(site_data)
    indicators = {}
    for key, ind in reg.indicators.items():
        if ind.status != "confirmed":
            continue
        if key in log.results and key not in blocked:
            indicators[key] = lineage_record(reg, key, log.results[key], log.context.get(key))
            log.published.append(key)
        elif key in previous:
            indicators[key] = {**previous[key], "carried_forward": True}
            log.carried.append(key)
    charts = {}
    for key, ch in reg.charts.items():
        if ch.status != "confirmed":
            continue
        if key in log.charts and f"chart {key}" not in blocked:
            charts[key] = chart_record(reg, key, log.charts[key])
            log.published.append(f"chart {key}")
        elif key in previous_charts(site_data):
            charts[key] = {**previous_charts(site_data)[key], "carried_forward": True}
            log.carried.append(f"chart {key}")
    doc = {"generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
           "run_id": log.run_id, "pipeline_version": __version__, "code_version": code_version(),
           "indicators": indicators, "charts": charts}
    site_data.mkdir(parents=True, exist_ok=True)
    (site_data / "pipeline.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    export_registry(reg, site_data)
    return doc


def export_registry(reg: Registry, site_data: Path = SITE_DATA) -> None:
    """The source registry, published on the site's Sources page."""
    doc = {
        "country": reg.country_name,
        "sources": [{"key": s.key, "publisher": s.publisher, "dataset": s.dataset,
                     "access": "Published files" if s.adapter == "files" else "Data service (API)",
                     "homepage": s.homepage, "licence": "" if s.terms == "to confirm" else s.terms,
                     "status": "Planned" if s.status == "planned" else "Live"}
                    for s in reg.sources.values()],
        "indicators": [{"key": i.key, "label": i.label, "unit": i.unit, "status": i.status,
                        "section": i.section, "calc": i.transform,
                        "series": [{"key": sk, "code": reg.series[sk].code,
                                    "source": reg.sources[reg.series[sk].source].publisher,
                                    "dataset": reg.sources[reg.series[sk].source].dataset,
                                    "frequency": reg.series[sk].frequency}
                                   for sk in i.series]}
                       for i in reg.indicators.values()],
    }
    site_data.mkdir(parents=True, exist_ok=True)
    (site_data / "registry.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
