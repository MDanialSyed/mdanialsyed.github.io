"""Integrity checks on every computed value before anything is published.

Levels:
  block  the value is not published this run
  flag   published, but listed in the report for the owner to confirm
  hold   not published because the indicator is not yet confirmed
  info   for the record
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .archive import sha256
from .config import Indicator, Registry
from .periods import period_end
from .transforms import Result

BLOCK, FLAG, HOLD, INFO = "block", "flag", "hold", "info"


@dataclass
class Finding:
    level: str
    indicator: str
    check: str
    message: str


def check_indicator(ind: Indicator, res: Result, data_root: Path, previous: dict | None,
                    today: date) -> list[Finding]:
    out = []
    # Lineage: every input row points to an archived file whose fingerprint still matches.
    for series_key, row in res.inputs:
        if not (row.raw_path and row.raw_sha256 and row.retrieved_at):
            out.append(Finding(BLOCK, ind.key, "lineage", f"{series_key} {row.period}: incomplete lineage"))
            continue
        path = data_root / row.raw_path
        if not path.exists():
            out.append(Finding(BLOCK, ind.key, "lineage", f"{series_key}: archived file missing ({row.raw_path})"))
        elif sha256(path.read_bytes()) != row.raw_sha256:
            out.append(Finding(BLOCK, ind.key, "lineage", f"{series_key}: archived file changed ({row.raw_path})"))
    # Range: plausible bounds from the registry.
    if ind.range and not (ind.range[0] <= res.value <= ind.range[1]):
        out.append(Finding(BLOCK, ind.key, "range", f"{res.value:g} outside {ind.range[0]:g} to {ind.range[1]:g}"))
    # Jump: compared with the last published value of a different period.
    if previous and previous.get("period") != res.period and previous.get("value") is not None:
        old = float(previous["value"])
        if ind.jump_abs is not None and abs(res.value - old) > ind.jump_abs:
            out.append(Finding(FLAG, ind.key, "jump", f"moved {res.value - old:+g} since {previous['period']} (limit {ind.jump_abs:g})"))
        if ind.jump_rel is not None and old and abs(res.value / old - 1) > ind.jump_rel:
            out.append(Finding(FLAG, ind.key, "jump", f"moved {100 * (res.value / old - 1):+.1f}% since {previous['period']}"))
    # Staleness: the latest period is older than the indicator's release cycle allows.
    if ind.max_age_days is not None:
        age = (today - period_end(res.period)).days
        if age > ind.max_age_days:
            out.append(Finding(FLAG, ind.key, "staleness", f"latest period {res.period} is {age} days old (limit {ind.max_age_days})"))
    if ind.status != "confirmed":
        out.append(Finding(HOLD, ind.key, "status", f"not published: indicator is '{ind.status}'. {ind.confirm}".strip()))
    return out


def check_identities(reg: Registry, results: dict[str, Result]) -> list[Finding]:
    """Totals that must add up, e.g. trade balance = exports - imports."""
    out = []
    for ident in reg.identities:
        terms = ident["rhs"]
        needed = [ident["lhs"]] + [t.lstrip("+-") for t in terms]
        missing = [k for k in needed if k not in results]
        if missing:
            out.append(Finding(INFO, ident["lhs"], "identity", f"{ident['name']}: skipped, missing {', '.join(missing)}"))
            continue
        periods_used = {results[k].period for k in needed}
        if len(periods_used) > 1:
            out.append(Finding(BLOCK, ident["lhs"], "identity", f"{ident['name']}: inputs cover different periods {sorted(periods_used)}"))
            continue
        rhs = sum(-results[t[1:]].value if t.startswith("-") else results[t.lstrip('+')].value for t in terms)
        lhs = results[ident["lhs"]].value
        if abs(lhs - rhs) > ident.get("tolerance", 1e-6):
            out.append(Finding(BLOCK, ident["lhs"], "identity", f"{ident['name']}: {lhs:g} vs {rhs:g}"))
    return out


def check_cross(reg: Registry, results: dict[str, Result]) -> list[Finding]:
    """Two sources for the same quantity and period must agree within a tolerance."""
    out = []
    for c in reg.cross_checks:
        a, b = results.get(c["a"]), results.get(c["b"])
        if not a or not b or a.period != b.period:
            continue
        if abs(a.value - b.value) > c.get("tolerance", 0):
            out.append(Finding(FLAG, c["a"], "cross-source",
                               f"{c['name']}: {a.value:g} vs {b.value:g} for {a.period}"))
    return out
