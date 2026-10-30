"""State Bank of Pakistan releases published on its website.

    forex   https://www.sbp.org.pk/assets/document/forex.pdf (PDF)
            Liquid foreign exchange reserves (million US$): month-end and week-end levels of
            net reserves with SBP, net reserves with banks, and the total.
    rates   https://www.sbp.org.pk/ (web page)
            The box "Cut-off Rates in the Latest Auctions": market treasury bill cut-off yields
            by tenor at the latest auction, with its date.

A series code names the file and the column, for example "forex#sbp" or "rates#12M". Each file
is downloaded once per run, archived untouched, turned into text (pypdf for PDFs), and read only
if its title and headings are exactly what the reader expects. Anything else raises a
FormatError and nothing from the file is used.
"""
from __future__ import annotations

import io
import re
from datetime import date, datetime

from ..secrets import scrub
from . import FetchResult, FormatError, Observation, parse_number

FILES = {
    "forex": ("https://www.sbp.org.pk/assets/document/forex.pdf", "forex.pdf", "pdf"),
    "rates": ("https://www.sbp.org.pk/", "sbp-home.html", "html"),
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

RATES_HEADING = "CUT-OFF RATES IN THE LATEST AUCTIONS"
MTB_SEGMENT = re.compile(r"MTBS\s+TENOR\s+CUT-OFF YIELD(.*?)\(AS ON ([A-Z]{3,9}\.? \d{1,2},? \d{4})\)")


def html_text(body: bytes) -> str:
    h = body.decode("utf-8", errors="replace")
    if "<html" not in h[:2000].lower():
        raise FormatError("the download is not a web page")
    h = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", h)
    h = re.sub(r"<[^>]+>", " ", h)
    for a, b in (("&amp;", "&"), ("&nbsp;", " "), ("&#8211;", "-")):
        h = h.replace(a, b)
    return re.sub(r"\s+", " ", h)


def parse_rates(text: str, tenor: str) -> list[Observation]:
    """Cut-off yield (percent) of one treasury bill tenor at the latest auction, from the State
    Bank's 'Cut-off Rates in the Latest Auctions' box, dated by the box's 'as on' date."""
    flat = _squash(text)
    if RATES_HEADING not in flat:
        raise FormatError("sbp.org.pk: box 'Cut-off Rates in the Latest Auctions' not found")
    m = MTB_SEGMENT.search(flat)
    if not m:
        raise FormatError("sbp.org.pk: treasury bill table or its 'as on' date not found")
    rows = dict(re.findall(r"(\d{1,2})-M\s+(\d{1,2}\.\d{2,4})%", m.group(1)))
    if not rows:
        raise FormatError("sbp.org.pk: no treasury bill cut-off yields in the box")
    months = tenor.rstrip("M")
    if tenor not in ("1M", "3M", "6M", "12M"):
        raise FormatError(f"sbp.org.pk: unknown tenor {tenor}")
    if months not in rows:
        raise FormatError(f"sbp.org.pk: no {tenor} cut-off yield in the box")
    return [Observation(period=_date(m.group(2).title()).isoformat(), value=float(rows[months]))]


# ---------------------------------------------------------------- shared

def _date(text: str) -> date:
    t = text.strip().replace(",", "").replace(".", "")
    for fmt in ("%d-%b-%y", "%d-%b-%Y", "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(t, fmt).date()
        except ValueError:
            continue
    raise FormatError(f"unrecognised date {text!r}")


PARSERS = {"forex": parse_forex, "rates": parse_rates}


def fetch(series, source, archive, fetcher, **_) -> FetchResult:
    try:
        file_key, column = series.code.split("#", 1)
    except ValueError:
        raise FormatError(f"{series.key}: code must be '<file>#<column>', got {series.code!r}")
    if file_key not in FILES:
        raise FormatError(f"{series.key}: unknown State Bank file {file_key}")
    url, name, kind = FILES[file_key]
    resp = _CACHE.get(url)
    if resp is None:
        resp = fetcher.get(url)
        _CACHE[url] = resp
    if resp.status != 200:
        raise FormatError(f"{series.code}: HTTP {resp.status} from {resp.url}")
    raw = archive.store(source.key, name, resp.body, resp.url,
                        extra={"file": file_key, "content_type": resp.content_type})
    try:
        text = pdf_text(resp.body) if kind == "pdf" else html_text(resp.body)
    except FormatError:
        raise
    except Exception as e:  # a malformed file must not stop the other series
        raise FormatError(f"{series.code}: could not read the file ({type(e).__name__}: {scrub(str(e))[:120]})")
    obs = PARSERS[file_key](text, column)
    return FetchResult(raw=raw, observations=obs, meta={"file": file_key, "column": column})
