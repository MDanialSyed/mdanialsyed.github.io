"""Source adapters: one per kind of official source.

Each adapter downloads one series, stores the file in the archive untouched,
and parses it into observations. Parsing never guesses: if the file does not
have the expected shape, the adapter raises an error and nothing is ingested.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..archive import RawRef


@dataclass
class Observation:
    period: str
    value: float | None
    meta: dict = field(default_factory=dict)


@dataclass
class FetchResult:
    raw: RawRef
    observations: list[Observation]
    meta: dict = field(default_factory=dict)


class FormatError(ValueError):
    """The downloaded file does not have the shape the parser expects."""


def get_adapter(name: str):
    from . import easydata, files, imf_datamapper, sbp_files, worldbank
    adapters = {"easydata": easydata, "worldbank": worldbank, "imf_datamapper": imf_datamapper,
                "files": files, "sbp_files": sbp_files}
    if name not in adapters:
        raise KeyError(f"No adapter called {name}")
    return adapters[name]


def parse_number(text) -> float | None:
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    t = str(text).strip().replace(",", "").replace("−", "-")
    if t in ("", "-", "..", "NA", "N/A", "n.a.", "null", "None"):
        return None
    return float(t)
