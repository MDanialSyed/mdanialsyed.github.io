"""Dates and periods.

Every observation is keyed by a period string whose shape depends on the
series frequency: daily and weekly 'YYYY-MM-DD', monthly 'YYYY-MM',
quarterly 'YYYY-Qn', annual 'YYYY'.
"""
from __future__ import annotations

import calendar
from datetime import date, datetime

DATE_FORMATS = ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d-%b-%Y", "%d-%b-%y",
                "%d/%m/%Y", "%d-%m-%Y", "%b-%Y", "%b %Y", "%Y-%m", "%Y")


def parse_date(text: str) -> date:
    text = text.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date: {text!r}")


def to_period(d: date, frequency: str) -> str:
    if frequency in ("daily", "weekly"):
        return d.isoformat()
    if frequency == "monthly":
        return f"{d.year:04d}-{d.month:02d}"
    if frequency == "quarterly":
        return f"{d.year:04d}-Q{(d.month - 1) // 3 + 1}"
    if frequency == "annual":
        return f"{d.year:04d}"
    raise ValueError(f"Unknown frequency: {frequency}")


def period_end(period: str) -> date:
    """The last day a period covers, used for staleness checks."""
    if len(period) == 10:
        return date.fromisoformat(period)
    if "-Q" in period:
        y, q = period.split("-Q")
        m = int(q) * 3
        return date(int(y), m, calendar.monthrange(int(y), m)[1])
    if len(period) == 7:
        y, m = int(period[:4]), int(period[5:])
        return date(y, m, calendar.monthrange(y, m)[1])
    if len(period) == 4:
        return date(int(period), 12, 31)
    raise ValueError(f"Unknown period: {period}")


def shift(period: str, steps: int) -> str:
    """Move a monthly, quarterly or annual period by a number of its own units."""
    if "-Q" in period:
        y, q = period.split("-Q")
        n = int(y) * 4 + int(q) - 1 + steps
        return f"{n // 4:04d}-Q{n % 4 + 1}"
    if len(period) == 7:
        n = int(period[:4]) * 12 + int(period[5:]) - 1 + steps
        return f"{n // 12:04d}-{n % 12 + 1:02d}"
    if len(period) == 4:
        return f"{int(period) + steps:04d}"
    raise ValueError(f"Cannot shift a daily period: {period}")


def fiscal_year_start(period: str, fy_start_month: int) -> str:
    """First month of the fiscal year that a monthly period belongs to."""
    y, m = int(period[:4]), int(period[5:])
    start_year = y if m >= fy_start_month else y - 1
    return f"{start_year:04d}-{fy_start_month:02d}"


def fiscal_year_label(period: str, fy_start_month: int) -> str:
    """Pakistan-style label: July 2026 to June 2027 is FY27."""
    start = fiscal_year_start(period, fy_start_month)
    end_year = int(start[:4]) + (1 if fy_start_month > 1 else 0)
    return f"FY{end_year % 100:02d}"


MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def label(period: str, style: str = "", fy_start_month: int = 7) -> str:
    """How a period is shown on the page: '28 Oct 2026', 'Sep 2026', 'Q3 2026', '2026',
    'Jul–Sep FY27' (fiscal year to date) or 'FY27' (an annual figure labelled by the year the
    fiscal year ends, as the IMF does for Pakistan)."""
    if style == "fytd":
        start = fiscal_year_start(period, fy_start_month)
        first, last = MONTHS[int(start[5:]) - 1], MONTHS[int(period[5:]) - 1]
        span = first if first == last else f"{first}\u2013{last}"
        return f"{span} {fiscal_year_label(period, fy_start_month)}"
    if style == "fy_end_year":
        return f"FY{int(period) % 100:02d}"
    if len(period) == 10:
        d = date.fromisoformat(period)
        return f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    if len(period) == 7:
        return f"{MONTHS[int(period[5:]) - 1]} {period[:4]}"
    if "-Q" in period:
        y, q = period.split("-Q")
        return f"Q{q} {y}"
    return period
