"""The run report: what was downloaded, what changed, what every check found, and what was
published. Written to <data>/runs/<run id>/ as report.md (to read) and report.json (to audit).
The weekly review starts from this report."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .build import RunLog, code_version
from .checks import BLOCK, FLAG, HOLD, INFO
from .config import Registry
from .periods import label as period_label
from .secrets import scrub
from .transforms import format_display

LEVEL_TITLES = {BLOCK: "Blocked (not published)", FLAG: "Flags to confirm",
                HOLD: "Held (indicator not yet confirmed)", INFO: "For the record"}


def _value_rows(reg: Registry, log: RunLog, interim: dict) -> list[list[str]]:
    rows = []
    blocked = {f.indicator for f in log.findings if f.level == BLOCK}
    for key, res in log.results.items():
        ind = reg.indicators[key]
        style = ind.period_format or ("fytd" if ind.transform.startswith("fytd") else "")
        shown = format_display(res.value, ind.display or {})
        hand = interim.get(key, {})
        if hand:
            same = "same" if hand.get("display") == shown else "differs"
            hand_text = f"{hand.get('display', '')} ({hand.get('period', '')}) {same}"
        else:
            hand_text = "not on the page"
        state = ("blocked" if key in blocked else "published" if key in log.published
                 else "held" if ind.status != "confirmed" else "not published")
        rows.append([key, shown, period_label(res.period, style, reg.fy_start_month), ind.transform,
                     hand_text, state])
    return rows


def render_markdown(reg: Registry, log: RunLog, interim: dict, published: bool) -> str:
    out = [f"# Pipeline run {log.run_id}", "",
           f"Code version {code_version()}. {'Published to the site data.' if published else 'Dry run: nothing written to the site.'}",
           ""]
    out += ["## Downloads", "", "| Series | Observations | New periods | Revised | Archived file |",
            "|---|---:|---:|---:|---|"]
    for key, f in log.fetched.items():
        ch = log.changes.get(key)
        out.append(f"| {key} | {f['observations']} | {len(ch.new) if ch else 0} | "
                   f"{len(ch.revised) if ch else 0} | {f['raw']} |")
    if log.errors:
        out += ["", "Download or parsing errors (these series were not updated):", ""]
        out += [f"- {e}" for e in log.errors]
    revisions = [(k, r) for k, ch in log.changes.items() for r in ch.revised]
    if revisions:
        out += ["", "## Revisions by the publisher", "", "| Series | Period | Was | Now |", "|---|---|---:|---:|"]
        out += [f"| {k} | {p} | {a:g} | {b:g} |" for k, (p, a, b) in revisions]
    out += ["", "## Values", "",
            "Compared with the hand-collected figure currently on the page.", "",
            "| Indicator | Value | Period | Calculation | Hand-collected | State |",
            "|---|---:|---|---|---|---|"]
    out += ["| " + " | ".join(r) + " |" for r in _value_rows(reg, log, interim)]
    if log.not_computed:
        out += ["", "Not computed:", ""] + [f"- {k}: {v}" for k, v in log.not_computed.items()]
    if log.carried:
        out += ["", "Carried forward from the last run (not refreshed this time): " + ", ".join(log.carried)]
    for level in (BLOCK, FLAG, HOLD, INFO):
        items = [f for f in log.findings if f.level == level]
        if items:
            out += ["", f"## {LEVEL_TITLES[level]}", ""]
            out += [f"- {f.indicator} ({f.check}): {f.message}" for f in items]
    out += ["", "## Archive", ""]
    out.append("Every archived file matches its fingerprint." if not log.archive_problems else
               "Files that no longer match their fingerprint:\n\n" + "\n".join(f"- {p}" for p in log.archive_problems))
    return scrub("\n".join(out) + "\n")


def write(reg: Registry, log: RunLog, data_root: Path, interim: dict, published: bool) -> Path:
    folder = Path(data_root) / "runs" / log.run_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.md").write_text(render_markdown(reg, log, interim, published), encoding="utf-8")
    doc = {
        "run_id": log.run_id, "code_version": code_version(), "published": published,
        "fetched": log.fetched, "errors": log.errors,
        "changes": {k: asdict(v) for k, v in log.changes.items()},
        "results": {k: {"value": r.value, "period": r.period, "note": r.note,
                        "inputs": [{"series": s, **asdict(row)} for s, row in r.inputs]}
                    for k, r in log.results.items()},
        "not_computed": log.not_computed, "findings": [asdict(f) for f in log.findings],
        "published_keys": log.published, "carried_forward": log.carried,
        "archive_problems": log.archive_problems,
    }
    (folder / "report.json").write_text(scrub(json.dumps(doc, indent=2, ensure_ascii=False)) + "\n", encoding="utf-8")
    return folder
