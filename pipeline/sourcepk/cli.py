"""Command line.

    python -m sourcepk doctor              check the setup (never prints the key)
    python -m sourcepk run                 download, compute, check and write a report (dry run)
    python -m sourcepk run --publish       ...and write confirmed values to the site data
    python -m sourcepk fetch [--series K]  download and archive only
    python -m sourcepk build [--publish]   compute and check from the stored series, no downloads
    python -m sourcepk status              every indicator and its confirmation status
    python -m sourcepk inspect SERIES      stored values and revision history for one series
    python -m sourcepk verify-archive      re-check every archived file's fingerprint
    python -m sourcepk export-registry     write the source registry for the site's Sources page
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from . import __version__, build, report
from .archive import Archive
from .config import data_dir, load_registry
from .secrets import has_secret, scrub
from .store import Store


def _say(text: str = "") -> None:
    print(scrub(text))


def cmd_doctor(args, reg) -> int:
    ok = True
    _say(f"Source: Pakistan pipeline {__version__}, Python {sys.version.split()[0]}")
    problems = reg.validate()
    _say(f"Registry: {len(reg.sources)} sources, {len(reg.series)} series, {len(reg.indicators)} indicators"
         + (f"; {len(problems)} problems" if problems else "; consistent"))
    for p in problems:
        _say(f"  - {p}")
    ok &= not problems
    try:
        root = data_dir(args.data)
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".write-test"
        probe.write_text("ok")
        probe.unlink()
        _say(f"Data folder: {root} (writable, outside the repo)")
    except (OSError, SystemExit) as e:
        _say(f"Data folder: problem: {e}")
        ok = False
    needs_key = any(reg.sources[s.source].secret for s in reg.series.values() if s.code)
    if needs_key:
        present = has_secret("easydata")
        _say(f"EasyData key: {'found' if present else 'not found (store it in the Keychain; see README)'}")
        ok &= present
    if args.online:
        import requests
        for name, url in [("State Bank EasyData", "https://easydata.sbp.org.pk/"),
                          ("World Bank API", "https://api.worldbank.org/v2/country/PAK?format=json"),
                          ("IMF DataMapper", "https://www.imf.org/external/datamapper/api/v1/countries")]:
            try:
                r = requests.get(url, timeout=20, headers={"User-Agent": "SourcePakistan-pipeline"})
                _say(f"{name}: HTTP {r.status_code}")
            except requests.RequestException as e:
                _say(f"{name}: unreachable ({type(e).__name__})")
                ok = False
    _say("Ready." if ok else "Fix the items above, then run doctor again.")
    return 0 if ok else 1


def _site(args) -> Path:
    return Path(args.site).expanduser().resolve() if args.site else build.SITE_DATA


def _build(args, reg, log, root) -> int:
    today = date.today()
    site = _site(args)
    build.compute_all(reg, root, log, today=today)
    build.run_checks(reg, root, log, build.load_previous(site), today)
    log.archive_problems = Archive(root).verify()
    if args.publish:
        build.publish(reg, log, site_data=site)
    folder = report.write(reg, log, root, build.load_interim(site), args.publish)
    blocks = sum(f.level == "block" for f in log.findings)
    flags = sum(f.level == "flag" for f in log.findings)
    _say(f"Computed {len(log.results)} indicators; {blocks} blocked, {flags} flagged, "
         f"{len(log.published)} published{'' if args.publish else ' (dry run)'}.")
    _say(f"Report: {folder / 'report.md'}")
    return 2 if blocks or log.archive_problems else 0


def cmd_run(args, reg) -> int:
    root = data_dir(args.data)
    log = build.RunLog(build.new_run_id())
    build.fetch(reg, root, log, only=args.series)
    _say(f"Downloaded {len(log.fetched)} series; {len(log.errors)} errors.")
    for e in log.errors:
        _say(f"  - {e}")
    return _build(args, reg, log, root)


def cmd_fetch(args, reg) -> int:
    root = data_dir(args.data)
    log = build.RunLog(build.new_run_id())
    build.fetch(reg, root, log, only=args.series)
    build.save_fetch_log(log, root)
    for key, ch in log.changes.items():
        _say(f"{key}: {len(ch.new)} new periods, {len(ch.revised)} revised")
    for e in log.errors:
        _say(f"error: {e}")
    return 1 if log.errors else 0


def cmd_build(args, reg) -> int:
    """Compute, check and report from the stored series, including the latest download run."""
    root = data_dir(args.data)
    log = build.RunLog(build.new_run_id())
    fetch_run = build.load_fetch_log(log, root)
    if fetch_run:
        _say(f"Using downloads from run {fetch_run}.")
    return _build(args, reg, log, root)


def cmd_status(args, reg) -> int:
    width = max(len(k) for k in reg.indicators)
    for i in reg.indicators.values():
        codes = ", ".join(reg.series[s].code or "(no code yet)" for s in i.series)
        _say(f"{i.key:<{width}}  {i.status:<18}  {i.transform or '-':<22}  {codes}")
    return 0


def cmd_inspect(args, reg) -> int:
    store = Store(data_dir(args.data))
    rows = store.rows(args.series_key)
    if not rows:
        _say(f"No stored values for {args.series_key}.")
        return 1
    latest = store.latest(args.series_key)
    for period in list(latest)[-args.last:]:
        hist = store.history(args.series_key, period)
        line = f"{period}  {latest[period].value:g}"
        if len(hist) > 1:
            line += "   revisions: " + " -> ".join(f"{r.value:g} ({r.retrieved_at[:10]})" for r in hist)
        _say(line)
    return 0


def cmd_verify(args, reg) -> int:
    bad = Archive(data_dir(args.data)).verify()
    _say("Every archived file matches its fingerprint." if not bad else "\n".join(["Changed or missing:"] + bad))
    return 1 if bad else 0


def cmd_export(args, reg) -> int:
    build.export_registry(reg, _site(args))
    _say(f"Wrote {_site(args) / 'registry.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sourcepk", description="Source: Pakistan data pipeline")
    p.add_argument("--data", help="data folder (default: $SP_DATA_DIR or ~/SourcePakistan/data)")
    p.add_argument("--country", default="pakistan")
    p.add_argument("--site", help="site data folder to read and write (default: the repo's _data/pakistan)")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("doctor")
    d.add_argument("--online", action="store_true", help="also check that the official sites respond")
    for name in ("run", "build"):
        s = sub.add_parser(name)
        s.add_argument("--publish", action="store_true", help="write confirmed values to the site data")
        if name == "run":
            s.add_argument("--series", nargs="*", help="only these series")
    f = sub.add_parser("fetch")
    f.add_argument("--series", nargs="*")
    sub.add_parser("status")
    i = sub.add_parser("inspect")
    i.add_argument("series_key")
    i.add_argument("--last", type=int, default=24)
    sub.add_parser("verify-archive")
    sub.add_parser("export-registry")
    args = p.parse_args(argv)
    if getattr(args, "series", None) is None and hasattr(args, "series"):
        args.series = None
    reg = load_registry(args.country)
    handlers = {"doctor": cmd_doctor, "run": cmd_run, "fetch": cmd_fetch, "build": cmd_build,
                "status": cmd_status, "inspect": cmd_inspect, "verify-archive": cmd_verify,
                "export-registry": cmd_export}
    return handlers[args.cmd](args, reg)


if __name__ == "__main__":
    raise SystemExit(main())
