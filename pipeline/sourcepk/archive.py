"""The raw archive: every downloaded file, stored exactly as received.

Layout (inside the data folder, outside the repo):

    raw/<source>/<YYYY-MM-DD>/<name>                 the file, byte for byte
    raw/<source>/<YYYY-MM-DD>/<name>.manifest.json   address, time, size, SHA-256

Files are never overwritten. If a file with the same name but different
content is stored on the same day, the new one gets a fingerprint suffix.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RawRef:
    path: str           # relative to the data folder
    sha256: str
    retrieved_at: str   # ISO 8601, UTC
    url: str            # redacted address
    bytes: int


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class Archive:
    def __init__(self, root: Path):
        self.root = Path(root)

    def store(self, source: str, name: str, body: bytes, url: str, extra: dict | None = None,
              retrieved_at: str | None = None) -> RawRef:
        if "REDACTED" not in url and "api_key=" in url.lower():
            raise ValueError("Refusing to store an unredacted address")
        retrieved_at = retrieved_at or now_utc()
        digest = sha256(body)
        folder = self.root / "raw" / source / retrieved_at[:10]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        if path.exists():
            if sha256(path.read_bytes()) == digest:
                return self._ref(path)  # identical file already stored today
            path = folder / f"{path.stem}.{digest[:12]}{path.suffix}"
            if path.exists():
                return self._ref(path)
        path.write_bytes(body)
        manifest = {"url": url, "retrieved_at": retrieved_at, "sha256": digest, "bytes": len(body),
                    "source": source, **(extra or {})}
        Path(f"{path}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        path.chmod(0o444)  # read-only: the archive is append-only
        return RawRef(path=str(path.relative_to(self.root)), sha256=digest, retrieved_at=retrieved_at,
                      url=url, bytes=len(body))

    def _ref(self, path: Path) -> RawRef:
        m = json.loads(Path(f"{path}.manifest.json").read_text(encoding="utf-8"))
        return RawRef(path=str(path.relative_to(self.root)), sha256=m["sha256"],
                      retrieved_at=m["retrieved_at"], url=m["url"], bytes=m["bytes"])

    def verify(self) -> list[str]:
        """Re-hash every archived file; return the ones that no longer match their manifest."""
        bad = []
        for manifest in sorted((self.root / "raw").rglob("*.manifest.json")):
            path = Path(str(manifest)[: -len(".manifest.json")])
            m = json.loads(manifest.read_text(encoding="utf-8"))
            if not path.exists() or sha256(path.read_bytes()) != m["sha256"]:
                bad.append(str(path.relative_to(self.root)))
        return bad

    @staticmethod
    def as_dict(ref: RawRef) -> dict:
        return asdict(ref)
