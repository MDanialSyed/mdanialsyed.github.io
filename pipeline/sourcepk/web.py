"""Polite downloads from official sites.

The pipeline identifies itself, waits between requests, retries once, and
never records a secret: every address is redacted before it is stored.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

from . import __version__

USER_AGENT = f"SourcePakistan-pipeline/{__version__} (+https://www.danialsyed.com/pakistan/)"
SECRET_PARAMS = {"api_key", "apikey", "key", "token", "access_token"}
PAUSE_SECONDS = 1.0


@dataclass
class Response:
    url: str          # redacted: safe to store and show
    status: int
    content_type: str
    body: bytes


def redact(url: str) -> str:
    """Replace secret query parameters with REDACTED."""
    parts = urlsplit(url)
    query = [(k, "REDACTED" if k.lower() in SECRET_PARAMS else v)
             for k, v in parse_qsl(parts.query, keep_blank_values=True)]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


class Fetcher:
    def __init__(self, session: requests.Session | None = None, pause: float = PAUSE_SECONDS):
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.pause = pause
        self._last = 0.0

    def get(self, url: str, timeout: int = 60) -> Response:
        last_error: Exception | None = None
        for attempt in range(2):
            wait = self.pause - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.session.get(url, timeout=timeout)
                self._last = time.monotonic()
                if r.status_code >= 500 and attempt == 0:
                    continue
                return Response(url=redact(url), status=r.status_code,
                                content_type=r.headers.get("Content-Type", ""), body=r.content)
            except requests.RequestException as e:  # network error: one retry, then give up
                self._last = time.monotonic()
                last_error = e
        raise DownloadError(f"Download failed for {redact(url)}: {type(last_error).__name__}")


class DownloadError(RuntimeError):
    pass
