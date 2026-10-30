#!/usr/bin/env python3
"""Check the Source: Pakistan data and the latest note before they go live.

Fails (exit 1) on anything that would put a wrong or unsourced number on the
site; prints warnings for things a person should look at in review.

Checks:
  1. Every data file exists, is a non-empty list (or object for meta/derived)
     and has the columns the pages read.
  2. Every indicator the pages and derive.py use is present; keys are unique;
     every shown row has a source name and an https link.
  3. derived.json matches a fresh run of derive.py.
  4. With --baseline DIR: large moves against last week's data are flagged
     for review (warnings, not failures).
  5. Every number in the newest weekly note appears somewhere in the data, so
     the note cannot quote a figure the dashboard does not hold.

Usage:
    python3 scripts/pakistan/check_data.py
    python3 scripts/pakistan/check_data.py --baseline /path/to/old/_data/pakistan --report report.md
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "_data" / "pakistan"
NOTES = ROOT / "_pakistan_notes"

COLUMNS = {
    "indicators": ["key", "section", "label", "value", "display", "period", "context", "source", "url"],
    "cpi_policy": ["month", "cpi_yoy", "policy_rate"],
    "gdp_quarters": ["quarter", "months", "growth"],
    "yield_curve": ["tenor", "latest", "previous", "latest_display", "previous_display"],
    "reserves_path": ["label", "sbp_bn", "kind", "note", "source", "url"],
    "fiscal_path": ["fy", "overall_balance", "primary_balance"],
    "external": ["item", "total", "m1", "m2", "change", "source", "url"],
    "peers": ["country", "cpi", "cpi_display", "measure", "policy_rate", "policy_display", "last_move",
              "source", "url", "rate_source", "rate_url"],
    "scorecard": ["item", "latest", "target", "status", "tone", "note", "source", "url"],
    "calendar": ["when", "event", "note", "source", "url"],
    "views": ["who", "view", "source", "url"],
}

# Indicators the dashboard page or derive.py reads by key.
REQUIRED_KEYS = [
    "gdp_fy", "gdp_q", "gdp_proj", "lsm_fytd", "pvt_credit",
    "cpi_yoy", "core_urban", "spi", "cpi_target", "cpi_target_lo", "cpi_target_hi",
    "policy_rate", "policy_prev", "tbill_12m", "pib_10y",
    "kse100", "kse100_base", "usdpkr", "usdpkr_base", "brent",
    "sbp_res", "res_goal", "ca_fytd", "rem_m", "rem_fytd", "trade_fytd",
    "fbr_fytd", "fbr_fytd_target", "fbr_fy_target", "deficit_fy", "primary_fy", "imf_status",
]

# Week-on-week moves big enough that a person should confirm them.
JUMPS = {
    "cpi_yoy": ("abs", 3.0), "core_urban": ("abs", 2.0), "policy_rate": ("abs", 1.5),
    "sbp_res": ("rel", 0.15), "usdpkr": ("rel", 0.05), "kse100": ("rel", 0.10),
    "brent": ("rel", 0.20), "tbill_12m": ("abs", 1.5), "pib_10y": ("abs", 1.5),
}

MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def numbers_in(text):
    """Numbers as they would be quoted: 10.3, 21.52, 167,193 -> 167193."""
    text = re.sub(r"\((?:https?://)[^)]*\)", " ", text)       # markdown link targets
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\b\d{1,2}\s+" + MONTHS + r"\b", " ", text)  # 30 Oct
    text = re.sub(r"\b" + MONTHS + r"\s+\d{1,2}\b", " ", text)  # Oct 30
    text = re.sub(r"\b(?:19|20)\d{2}\b", " ", text)            # years
    text = re.sub(r"\d{4}-\d{2}-\d{2}", " ", text)             # ISO dates
    text = re.sub(r"Rs(?=\d)", "Rs ", text)                     # Rs3,083bn
    found = set()
    for m in re.finditer(r"(?<![A-Za-z0-9.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?", text):
        whole = m.group(1).replace(",", "")
        frac = m.group(2) or ""
        found.add(round(float(whole + frac), 4))
    return found


def collect_numbers(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("url", "rate_url"):
                continue
            collect_numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            collect_numbers(v, out)
    elif isinstance(obj, bool) or obj is None:
        return
    elif isinstance(obj, (int, float)):
        out.add(round(float(obj), 4))
        out.add(round(abs(float(obj)), 4))
    elif isinstance(obj, str):
        out.update(numbers_in(obj))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", help="folder holding last week's _data/pakistan files")
    ap.add_argument("--report", help="append a markdown summary to this file")
    args = ap.parse_args()

    errors, warnings, notes = [], [], []
    data = {}

    # 1. Files and columns
    for name, cols in COLUMNS.items():
        path = DATA / f"{name}.json"
        if not path.exists():
            errors.append(f"{name}.json is missing")
            continue
        try:
            rows = load(path)
        except json.JSONDecodeError as e:
            errors.append(f"{name}.json is not valid JSON: {e}")
            continue
        if not isinstance(rows, list) or not rows:
            errors.append(f"{name}.json must be a non-empty list of rows")
            continue
        for i, r in enumerate(rows):
            missing = [c for c in cols if c not in r]
            if missing:
                errors.append(f"{name}.json row {i} is missing {', '.join(missing)}")
            for c in ("url", "rate_url"):
                if c in r and r.get(c) and not str(r[c]).startswith("https://"):
                    errors.append(f"{name}.json row {i}: {c} must be an https link")
        data[name] = rows

    for name in ("meta", "derived"):
        path = DATA / f"{name}.json"
        if not path.exists():
            errors.append(f"{name}.json is missing")
        else:
            data[name] = load(path)

    # 2. Indicators
    ind = data.get("indicators", [])
    keys = [r.get("key") for r in ind]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        errors.append(f"indicators.json has duplicate keys: {', '.join(dupes)}")
    for k in REQUIRED_KEYS:
        if k not in keys:
            errors.append(f"indicators.json is missing required key '{k}'")
    for r in ind:
        if r.get("section") != "ref":
            if not r.get("source") or not str(r.get("url", "")).startswith("https://"):
                errors.append(f"indicator '{r.get('key')}' needs a source name and an https link")
        v = r.get("value")
        if v is not None and not isinstance(v, (int, float)):
            errors.append(f"indicator '{r.get('key')}' value must be a number or null")
        if not str(r.get("display", "")).strip():
            errors.append(f"indicator '{r.get('key')}' has an empty display")

    meta = data.get("meta", {})
    for field in ("asof", "asof_iso", "takeaway"):
        if not meta.get(field):
            errors.append(f"meta.json needs '{field}'")
    for group, fields in (("curve", ("latest", "previous", "latest_short", "previous_short", "note")),
                          ("external", ("title", "note", "total", "m1", "m2", "change"))):
        missing = [f for f in fields if not (meta.get(group) or {}).get(f)]
        if missing:
            errors.append(f"meta.json '{group}' needs {', '.join(missing)}")
    try:
        asof = date.fromisoformat(meta.get("asof_iso", ""))
        age = (date.today() - asof).days
        if age > 8:
            warnings.append(f"meta.json asof_iso is {asof} ({age} days old)")
    except ValueError:
        errors.append("meta.json needs asof_iso as YYYY-MM-DD")

    # 2b. Official figures from the data pipeline: complete lineage, confirmed in the registry
    pipeline, registry = {}, {}
    if (DATA / "pipeline.json").exists():
        pipeline = load(DATA / "pipeline.json").get("indicators", {})
    if (DATA / "registry.json").exists():
        registry = {i["key"]: i for i in load(DATA / "registry.json").get("indicators", [])}
    else:
        errors.append("registry.json is missing; run python -m sourcepk export-registry in pipeline/")
    for k, r in pipeline.items():
        if registry.get(k, {}).get("status") != "confirmed":
            errors.append(f"pipeline figure '{k}' is not a confirmed indicator in registry.json")
        for f in ("value", "display", "period", "period_label", "publisher", "series_code", "calc", "retrieved_at"):
            if r.get(f) in (None, ""):
                errors.append(f"pipeline figure '{k}' has no {f}")
        inputs = r.get("inputs") or []
        if not inputs:
            errors.append(f"pipeline figure '{k}' has no lineage")
        for x in inputs:
            if not re.fullmatch(r"[0-9a-f]{64}", str(x.get("sha256", ""))) or not x.get("file"):
                errors.append(f"pipeline figure '{k}' has an input without a file fingerprint")
        if r.get("carried_forward"):
            warnings.append(f"pipeline figure '{k}' was not refreshed in the last run")
    data["pipeline"] = [{"value": r.get("value"), "display": r.get("display")} for r in pipeline.values()]
    charts = load(DATA / "pipeline.json").get("charts", {}) if (DATA / "pipeline.json").exists() else {}
    for k, c in charts.items():
        if not c.get("points") or not c.get("files") or not all(re.fullmatch(r"[0-9a-f]{64}", f.get("sha256", "")) for f in c["files"]):
            errors.append(f"pipeline chart '{k}' has no points or incomplete lineage")
    data["pipeline_charts"] = [[p.get("value") for p in c.get("points", [])] for c in charts.values()]

    # 2c. Cross-source: the hand-entered inflation history must match the State Bank's series
    official_cpi = {p["period"]: p["value"] for p in charts.get("cpi_yoy_recent", {}).get("points", [])}
    for r in data.get("cpi_policy", []):
        m, v = str(r.get("month"))[:7], r.get("cpi_yoy")
        if v is not None and m in official_cpi and abs(float(v) - float(official_cpi[m])) > 1e-9:
            errors.append(f"cpi_policy.json {m}: CPI {v} differs from the State Bank's {official_cpi[m]}")
    if official_cpi:
        notes.append("### Cross-source\n- The inflation chart's monthly CPI matches the State Bank's series.")

    # 2d. Moves beside the headline tiles: interim rows that carry a change need a basis
    for r in ind:
        if r.get("change") is not None and not str(r.get("change_basis", "")).strip():
            errors.append(f"indicator '{r.get('key')}' has a change but no change_basis")

    # 3. Derived figures
    res = subprocess.run([sys.executable, str(ROOT / "scripts/pakistan/derive.py"), "--check"],
                         capture_output=True, text=True)
    if res.returncode != 0:
        errors.append((res.stdout + res.stderr).strip() or "derived.json check failed")

    # 4. Moves against last week
    if args.baseline:
        bpath = Path(args.baseline) / "indicators.json"
        if bpath.exists():
            old = {r["key"]: r for r in load(bpath)}
            new = {r["key"]: r for r in ind}
            changed = []
            for k, r in new.items():
                o = old.get(k)
                if o is None:
                    changed.append(f"`{k}` added: {r.get('display')}")
                    continue
                if (o.get("display"), o.get("period")) != (r.get("display"), r.get("period")):
                    changed.append(f"`{k}`: {o.get('display')} ({o.get('period')}) → {r.get('display')} ({r.get('period')})")
                if k in JUMPS and o.get("value") is not None and r.get("value") is not None:
                    kind, limit = JUMPS[k]
                    a, b = float(o["value"]), float(r["value"])
                    move = abs(b - a) if kind == "abs" else (abs(b - a) / abs(a) if a else 0)
                    if move > limit:
                        warnings.append(f"large move in `{k}`: {o.get('display')} → {r.get('display')}; confirm against the source")
            for k in old:
                if k not in new:
                    warnings.append(f"`{k}` was removed")
            notes.append("### What changed\n" + ("\n".join(f"- {c}" for c in changed) if changed else "- No indicator changed."))
        else:
            warnings.append(f"no baseline found at {bpath}")

    # 5. The headline takeaway and the newest note may only quote numbers that are in the data
    # The page's written analysis (takeaway, "what it means", chart commentary) may only quote
    # numbers that are in the data; so it is left out of the numbers it is checked against.
    written = {"takeaway": str(meta.get("takeaway", "")), "so_what": str(meta.get("so_what", ""))}
    for k, v in (meta.get("commentary") or {}).items():
        written[f"commentary.{k}"] = str(v)
    known = set()
    for name, rows in data.items():
        if name == "meta":
            rows = {k: v for k, v in rows.items() if k not in ("takeaway", "so_what", "commentary")}
        collect_numbers(rows, known)
    if not written["takeaway"].strip():
        errors.append("meta.json needs a one-sentence takeaway")
    if not written["so_what"].strip():
        errors.append("meta.json needs a one- or two-sentence so_what")
    for field, text in written.items():
        unknown = sorted(n for n in numbers_in(text) if n not in known)
        if unknown:
            errors.append(f"meta.json {field} quotes numbers that are not in the data: " + ", ".join(f"{n:g}" for n in unknown))
    if NOTES.exists():
        latest = sorted(NOTES.glob("*.md"))
        if latest:
            note = latest[-1]
            text = note.read_text(encoding="utf-8")
            body = text.split("---", 2)[2] if text.startswith("---") else text
            title = re.search(r"^title:\s*(.+)$", text, re.M)
            subtitle = re.search(r"^subtitle:\s*(.+)$", text, re.M)
            quoted = numbers_in(body + " " + (title.group(1) if title else "") + " " + (subtitle.group(1) if subtitle else ""))
            unknown = sorted(n for n in quoted if n not in known)
            if unknown:
                shown = ", ".join(f"{n:g}" for n in unknown)
                errors.append(f"{note.name} quotes numbers that are not in the data: {shown}")
            else:
                notes.append(f"### Weekly note\n- `{note.name}`: every number it quotes is in the data.")

    # Output
    lines = []
    if errors:
        lines.append("### Problems (must fix)\n" + "\n".join(f"- {e}" for e in errors))
    if warnings:
        lines.append("### Check before merging\n" + "\n".join(f"- {w}" for w in warnings))
    lines.extend(notes)
    summary = "\n\n".join(lines) if lines else "All checks passed."
    print(summary)
    if args.report:
        with open(args.report, "a", encoding="utf-8") as f:
            f.write("\n\n## Checks\n\n" + summary + "\n")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
