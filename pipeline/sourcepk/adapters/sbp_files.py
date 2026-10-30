"""State Bank of Pakistan releases published as PDF files.

    forex   https://www.sbp.org.pk/assets/document/forex.pdf
            Liquid foreign exchange reserves (million US$): month-end and week-end levels of
            net reserves with SBP, net reserves with banks, and the total.
    mtb     https://www.sbp.org.pk/ecodata/auction-tbills.pdf
            Market treasury bill auction results: cut-off yields by tenor.

A series code names the file and the column, for example "forex#sbp" or "mtb#12M". Each file
is downloaded once per run, archived untouched, turned into text with pypdf, and read only if
its title and column headings are exactly what the reader expects. Anything else raises a
FormatError and nothing from the file is used.
"""
from __future__ import annotations

import io
import re
from datetime import date, datetime

from ..secrets import scrub
from . import FetchResult, FormatError, Observation, parse_number

FILES = {
    "forex": "https://www.sbp.org.pk/assets/document/forex.pdf",
    "mtb": "https://www.sbp.org.pk/ecodata/auction-tbills.pdf",
}

_CACHE: dict[str, object] = {}   # one download per file per run


def pdf_text(body: bytes) -> str:
    """Text of every page, in order."""
    try:
        from pypdf import PdfReader
    except ImportError as e:  # the Mac installs it from requirements.txt
        raise FormatError("pypdf is not installed") from e
    if not body.startswith(b"%PDF"):
        raise FormatError("the download is not a PDF")
    reader = PdfReader(io.BytesIO(body))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().upper()


# ---------------------------------------------------------------- weekly reserves

FOREX_TITLE = "LIQUID FOREIGN EXCHANGE RESERVES"
FOREX_HEADINGS = ("NET RESERVES WITH SBP", "NET RESERVES WITH BANKS", "TOTAL LIQUID")
FOREX_COLUMNS = {"sbp": 0, "banks": 1, "total": 2}
_NUM = r"(-?[\d,]+\.\d+)"
WEEK_ROW = re.compile(r"(\d{1,2}-[A-Za-z]{3}-\d{2,4})\s*(?:[PR]\b\s*)?" + _NUM + r"\s+" + _NUM + r"\s+" + _NUM)


def parse_forex(text: str, column: str) -> list[Observation]:
    """Week-end levels (million US$) for one column. Month-end and annual rows are left out:
    the weekly rows are the series the State Bank announces every Thursday."""
    flat = _squash(text)
    if FOREX_TITLE not in flat:
        raise FormatError("forex.pdf: title 'Liquid foreign exchange reserves' not found")
    for h in FOREX_HEADINGS:
        if h not in flat:
            raise FormatError(f"forex.pdf: column heading '{h}' not found")
    if "MILLION US" not in flat:
        raise FormatError("forex.pdf: unit 'million US$' not found")
    if column not in FOREX_COLUMNS:
        raise FormatError(f"forex.pdf: unknown column {column}")
    obs, seen = [], set()
    for m in WEEK_ROW.finditer(text):
        d = _date(m.group(1))
        sbp, banks, total = (parse_number(m.group(i)) for i in (2, 3, 4))
        if abs(sbp + banks - total) > 0.15:   # the row must add up: SBP + banks = total
            raise FormatError(f"forex.pdf: {m.group(1)} does not add up ({sbp} + {banks} != {total})")
        if d.isoformat() in seen:
            raise FormatError(f"forex.pdf: {m.group(1)} appears twice")
        seen.add(d.isoformat())
        value = (sbp, banks, total)[FOREX_COLUMNS[column]]
        obs.append(Observation(period=d.isoformat(), value=value,
                               meta={"provisional": bool(re.search(r"\bP\b", m.group(0)))}))
    if not obs:
        raise FormatError("forex.pdf: no week-end rows found")
    return sorted(obs, key=lambda o: o.period)


# ---------------------------------------------------------------- treasury bill auctions

MTB_TITLE = "MARKET TREASURY BILLS"
TENORS = {"1M": ("1-M", "1 M", "1-MONTH", "01-M"), "3M": ("3-M", "3 M", "3-MONTH", "03-M"),
          "6M": ("6-M", "6 M", "6-MONTH", "06-M"), "12M": ("12-M", "12 M", "12-MONTH")}
AS_ON = re.compile(r"(?:AS ON|HELD ON|AUCTION DATE)[:\s]*([A-Z][a-z]{2,8}\.? \d{1,2},? \d{4}|\d{1,2}[-/ ][A-Za-z]{3,9}[-/ ]\d{2,4})", re.I)


def parse_mtb(text: str, tenor: str) -> list[Observation]:
    """Cut-off yield (percent) of one tenor at the latest auction in the file."""
    flat = _squash(text)
    if MTB_TITLE not in flat:
        raise FormatError("auction-tbills.pdf: title 'Market Treasury Bills' not found")
    if "CUT" not in flat or "YIELD" not in flat:
        raise FormatError("auction-tbills.pdf: 'cut-off yield' not found")
    if tenor not in TENORS:
        raise FormatError(f"auction-tbills.pdf: unknown tenor {tenor}")
    m = AS_ON.search(text)
    if not m:
        raise FormatError("auction-tbills.pdf: auction date not found")
    d = _date(m.group(1))
    value = None
    for label in TENORS[tenor]:
        hit = re.search(re.escape(label) + r"[A-Z ]*?\s+(\d{1,2}\.\d{2,4})\s*%?", flat)
        if hit:
            value = float(hit.group(1))
            break
    if value is None:
        raise FormatError(f"auction-tbills.pdf: no cut-off yield for {tenor}")
    return [Observation(period=d.isoformat(), value=value)]


# ---------------------------------------------------------------- shared

def _date(text: str) -> date:
    t = text.strip().replace(",", "").replace(".", "")
    for fmt in ("%d-%b-%y", "%d-%b-%Y", "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(t, fmt).date()
        except ValueError:
            continue
    raise FormatError(f"unrecognised date {text!r}")


PARSERS = {"forex": parse_forex, "mtb": parse_mtb}


def fetch(series, source, archive, fetcher, **_) -> FetchResult:
    try:
        file_key, column = series.code.split("#", 1)
    except ValueError:
        raise FormatError(f"{series.key}: code must be '<file>#<column>', got {series.code!r}")
    if file_key not in FILES:
        raise FormatError(f"{series.key}: unknown State Bank file {file_key}")
    url = FILES[file_key]
    resp = _CACHE.get(url)
    if resp is None:
        resp = fetcher.get(url)
        _CACHE[url] = resp
    if resp.status != 200:
        raise FormatError(f"{series.code}: HTTP {resp.status} from {resp.url}")
    raw = archive.store(source.key, url.rsplit("/", 1)[1], resp.body, resp.url,
                        extra={"file": file_key, "content_type": resp.content_type})
    try:
        text = pdf_text(resp.body)
    except FormatError:
        raise
    except Exception as e:  # a malformed PDF must not stop the other series
        raise FormatError(f"{series.code}: could not read the PDF ({type(e).__name__}: {scrub(str(e))[:120]})")
    obs = PARSERS[file_key](text, column)
    return FetchResult(raw=raw, observations=obs, meta={"file": file_key, "column": column})
