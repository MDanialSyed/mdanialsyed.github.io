"""API keys: read from the environment or the macOS Keychain, never from files in the repo.

To store the EasyData key in the Keychain (run once in Terminal; it asks for the key):

    security add-generic-password -s source-pakistan-easydata -a "$USER" -w

The value is never printed, logged or written to any file by this package.
"""
from __future__ import annotations

import os
import shutil
import subprocess

KEYCHAIN_SERVICES = {"easydata": "source-pakistan-easydata"}
ENV_VARS = {"easydata": "SP_EASYDATA_KEY"}


_LOADED: set[str] = set()


class MissingSecret(RuntimeError):
    pass


def scrub(text: str) -> str:
    """Remove any key loaded in this process from text before it is printed or written."""
    for value in _LOADED:
        text = text.replace(value, "REDACTED")
    return text


def _remember(value: str) -> str:
    if len(value) >= 6:
        _LOADED.add(value)
    return value


def get_secret(name: str) -> str:
    env = ENV_VARS.get(name)
    if env and os.environ.get(env):
        return _remember(os.environ[env].strip())
    service = KEYCHAIN_SERVICES.get(name)
    if service and shutil.which("security"):
        res = subprocess.run(["security", "find-generic-password", "-s", service, "-w"],
                             capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            return _remember(res.stdout.strip())
    raise MissingSecret(
        f"No '{name}' key found. Set {env} or add it to the macOS Keychain as service '{service}'.")


def has_secret(name: str) -> bool:
    try:
        get_secret(name)
        return True
    except MissingSecret:
        return False
