"""Named, versioned calculations from series to published values.

Every published value records which calculation produced it (for example
'yoy_pct@1'). A calculation is never changed in place: a new behaviour gets a
new version number, so old lineage records keep their meaning.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from . import periods
from .store import Row

MINUS = "−"


@dataclass
class Result:
    value: float
    period: str
    inputs: list[tuple[str, Row]]   # (series key, the exact stored row used)
    note: str = ""


class NotComputable(ValueError):
    pass


def _latest(series: dict[str, Row]) -> tuple[str, Row]:
    if not series:
        raise NotComputable("series is empty")
    period = next(reversed(series))
    return period, series[period]


def as_published(key: str, series: dict[str, Row], **_) -> Result:
    period, row = _latest(series)
    return Result(row.value, period, [(key, row)])


def yoy_pct(key: str, series: dict[str, Row], **_) -> Result:
    period, row = _latest(series)
    steps = 4 if "-Q" in period else 12 if len(period) == 7 else 1
    prev = series.get(periods.shift(period, -steps))
    if prev is None or prev.value == 0:
        raise NotComputable(f"no value a year before {period}")
    return Result((row.value / prev.value - 1) * 100, period, [(key, row), (key, prev)])


def mom_pct(key: str, series: dict[str, Row], **_) -> Result:
    period, row = _latest(series)
    prev = series.get(periods.shift(period, -1))
    if prev is None or prev.value == 0:
        raise NotComputable(f"no value before {period}")
    return Result((row.value / prev.value - 1) * 100, period, [(key, row), (key, prev)])


def fytd_sum(key: str, series: dict[str, Row], fy_start_month: int = 7, **_) -> Result:
    """Sum of monthly flows from the start of the fiscal year to the latest month; every month must exist."""
    period, _row = _latest(series)
    p = periods.fiscal_year_start(period, fy_start_month)
    used = []
    while p <= period:
        if p not in series:
            raise NotComputable(f"missing month {p} in the fiscal year to {period}")
        used.append((key, series[p]))
        p = periods.shift(p, 1)
    return Result(sum(r.value for _, r in used), period, used,
                  note=f"{used[0][1].period} to {period}")


def fytd_mean_yoy_pct(key: str, series: dict[str, Row], fy_start_month: int = 7, **_) -> Result:
    """Average of an index over the fiscal year to date, compared with the same months a year
    earlier (how average inflation and manufacturing growth to date are reported). Every month of
    both spans must exist."""
    period, _row = _latest(series)
    p = periods.fiscal_year_start(period, fy_start_month)
    cur, prev = [], []
    while p <= period:
        q = periods.shift(p, -12)
        if p not in series or q not in series:
            raise NotComputable(f"missing {p if p not in series else q} for the fiscal year to {period}")
        cur.append((key, series[p]))
        prev.append((key, series[q]))
        p = periods.shift(p, 1)
    mean_cur = sum(r.value for _, r in cur) / len(cur)
    mean_prev = sum(r.value for _, r in prev) / len(prev)
    if mean_prev == 0:
        raise NotComputable("average a year earlier is zero")
    return Result((mean_cur / mean_prev - 1) * 100, period, cur + prev,
                  note=f"average {cur[0][1].period} to {period} vs {prev[0][1].period} to {prev[-1][1].period}")


def calendar_year(key: str, series: dict[str, Row], offset: int = 0, today: date | None = None, **_) -> Result:
    """The value for the current calendar year (+offset), e.g. an IMF projection."""
    year = str((today or date.today()).year + offset)
    if year not in series:
        raise NotComputable(f"no value for {year}")
    return Result(series[year].value, year, [(key, series[year])])


TRANSFORMS = {
    "as_published@1": as_published,
    "yoy_pct@1": yoy_pct,
    "mom_pct@1": mom_pct,
    "fytd_sum@1": fytd_sum,
    "fytd_mean_yoy_pct@1": fytd_mean_yoy_pct,
    "calendar_year@1": calendar_year,
    "next_calendar_year@1": lambda k, s, **kw: calendar_year(k, s, offset=1, **kw),
}


def compute(transform: str, key: str, series: dict[str, Row], **kw) -> Result:
    if transform not in TRANSFORMS:
        raise NotComputable(f"unknown calculation {transform}")
    return TRANSFORMS[transform](key, series, **kw)


def round_half_up(value: float, decimals: int) -> Decimal:
    """Round as publishers do (10.25 -> 10.3), working from the shortest decimal form of the value
    rather than its binary approximation."""
    return Decimal(repr(value)).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


def format_display(value: float, spec: dict) -> str:
    """Format a value for the page: scale, decimals, prefix/suffix, true minus sign, optional plus sign."""
    v = value * spec.get("scale", 1)
    decimals = spec.get("decimals", 1)
    r = round_half_up(v, decimals)
    body = f"{abs(r):,.{decimals}f}" if spec.get("thousands", True) else f"{abs(r):.{decimals}f}"
    sign = MINUS if r < 0 else ("+" if spec.get("plus") and r > 0 else "")
    return f"{sign}{spec.get('prefix', '')}{body}{spec.get('suffix', '')}"
