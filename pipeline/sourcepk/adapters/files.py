"""Official files (PDF and Excel releases from PBS, the Finance Division, FBR, PSX and SBP).

Phase 3: each release gets its own parser here, with tests against stored
past files, before any of its indicators can be confirmed.
"""
from __future__ import annotations


def fetch(series, source, archive, fetcher, **_):
    raise NotImplementedError(
        f"{series.key}: parsers for {source.publisher} files are built in phase 3")
