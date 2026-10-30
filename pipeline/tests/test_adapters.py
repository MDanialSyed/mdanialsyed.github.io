from datetime import date

import pytest

from conftest import easydata_csv, imf_json, worldbank_json
from sourcepk.adapters import FormatError, easydata, imf_datamapper, worldbank


def test_easydata_parse():
    body = easydata_csv("A.B", {"2026-08": 1.5, "2026-09": 2.5})
    obs, meta = easydata.parse(body, "A.B", "monthly")
    assert [(o.period, o.value) for o in obs] == [("2026-08", 1.5), ("2026-09", 2.5)]
    assert meta["Series Name"] == "Synthetic series"


def test_easydata_checks_unit_and_name():
    body = easydata_csv("A.B", {"2026-09": 1.0}, name="Quantum Index - Overall", unit="Index")
    easydata.parse(body, "A.B", "monthly", unit="Index", name="Quantum Index - Overall")
    with pytest.raises(FormatError):
        easydata.parse(body, "A.B", "monthly", unit="Percent")
    with pytest.raises(FormatError):
        easydata.parse(body, "A.B", "monthly", name="Something else")


def test_easydata_rejects_other_series():
    with pytest.raises(FormatError):
        easydata.parse(easydata_csv("OTHER", {"2026-09": 1.0}), "A.B", "monthly")


def test_easydata_rejects_unexpected_columns():
    with pytest.raises(FormatError):
        easydata.parse(b"Date,Value\n2026-09-30,1\n", "A.B", "monthly")


def test_easydata_url_carries_key_only_in_request():
    url = easydata.build_url("TS_X.Y", "2020-01-01", "2026-10-30", "K")
    assert url.startswith("https://easydata.sbp.org.pk/api/v1/series/TS_X.Y/data?api_key=K")


def test_worldbank_parse():
    obs, meta = worldbank.parse(worldbank_json("NY.GDP.MKTP.CD", {"2024": 3.7e11, "2025": None}), "NY.GDP.MKTP.CD")
    assert [(o.period, o.value) for o in obs] == [("2024", 3.7e11), ("2025", None)]
    with pytest.raises(FormatError):
        worldbank.parse(worldbank_json("NY.GDP.MKTP.CD", {"2024": 1.0}, pages=2), "NY.GDP.MKTP.CD")
    with pytest.raises(FormatError):
        worldbank.parse(worldbank_json("OTHER", {"2024": 1.0}), "NY.GDP.MKTP.CD")


def test_imf_parse_marks_projections():
    obs, _ = imf_datamapper.parse(imf_json("NGDP_RPCH", {"2025": 2.5, "2027": 4.0}), "NGDP_RPCH", "PAK",
                                  today=date(2026, 10, 30))
    assert [(o.period, o.meta["projection"]) for o in obs] == [("2025", False), ("2027", True)]
    with pytest.raises(FormatError):
        imf_datamapper.parse(imf_json("NGDP_RPCH", {}), "PCPIPCH", "PAK")


def test_easydata_probe_never_reveals_the_key(monkeypatch, key):
    import json as _json
    import requests

    class R:
        status_code, headers = 403, {"Content-Type": "text/html", "Server": "x"}
        content = f"<html>Forbidden for {key}</html>".encode()

    monkeypatch.setattr(requests, "get", lambda *a, **k: R())
    out = easydata.probe()
    assert out["key_length"] == len(key) and len(out["attempts"]) == 4
    assert key not in _json.dumps(out) and "Forbidden for REDACTED" in out["attempts"][1]["body"]
