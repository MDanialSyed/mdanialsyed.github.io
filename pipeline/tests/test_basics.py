import os
import stat
from datetime import date

import pytest

from sourcepk import periods
from sourcepk.archive import Archive, RawRef
from sourcepk.adapters import Observation
from sourcepk.store import Store
from sourcepk.transforms import NotComputable, compute, format_display
from sourcepk.web import redact


def test_redact_removes_keys():
    url = "https://easydata.sbp.org.pk/api/v1/series/X/data?api_key=SECRET&start_date=2020-01-01&token=T"
    out = redact(url)
    assert "SECRET" not in out and "=T" not in out
    assert "api_key=REDACTED" in out and "start_date=2020-01-01" in out


def test_archive_is_append_only(tmp_path):
    a = Archive(tmp_path)
    url = "https://example.org/x?api_key=REDACTED"
    r1 = a.store("src", "x.csv", b"one", url, retrieved_at="2026-10-30T10:00:00Z")
    assert (tmp_path / r1.path).read_bytes() == b"one"
    assert not os.stat(tmp_path / r1.path).st_mode & stat.S_IWUSR
    assert a.store("src", "x.csv", b"one", url, retrieved_at="2026-10-30T11:00:00Z").path == r1.path
    r2 = a.store("src", "x.csv", b"two", url, retrieved_at="2026-10-30T12:00:00Z")
    assert r2.path != r1.path and (tmp_path / r1.path).read_bytes() == b"one"
    assert a.verify() == []


def test_archive_refuses_unredacted_address(tmp_path):
    with pytest.raises(ValueError):
        Archive(tmp_path).store("src", "x.csv", b"1", "https://example.org/x?api_key=SECRET")


def test_archive_detects_tampering(tmp_path):
    a = Archive(tmp_path)
    r = a.store("src", "x.csv", b"one", "https://example.org/x")
    p = tmp_path / r.path
    p.chmod(0o644)
    p.write_bytes(b"changed")
    assert a.verify() == [r.path]


def _raw(tag):
    return RawRef(path=f"raw/s/{tag}.csv", sha256=tag * 8, retrieved_at=f"2026-10-{tag}T00:00:00Z", url="u", bytes=1)


def test_store_keeps_every_vintage(tmp_path):
    s = Store(tmp_path)
    ch = s.ingest("k", [Observation("2026-08", 1.0), Observation("2026-09", 2.0)], _raw("01"))
    assert ch.new == ["2026-08", "2026-09"] and ch.revised == []
    ch = s.ingest("k", [Observation("2026-08", 1.0), Observation("2026-09", 2.5), Observation("2026-10", None)], _raw("02"))
    assert ch.new == [] and ch.revised == [("2026-09", 2.0, 2.5)]
    assert s.latest("k")["2026-09"].value == 2.5
    assert [r.value for r in s.history("k", "2026-09")] == [2.0, 2.5]
    assert len(s.rows("k")) == 3


def _series(values):
    from sourcepk.store import Row
    return {p: Row(p, v, "2026-10-30T00:00:00Z", f"raw/{p}", "h") for p, v in values.items()}


def test_transforms():
    s = _series({"2025-09": 100.0, "2026-08": 109.0, "2026-09": 110.3})
    r = compute("yoy_pct@1", "k", s)
    assert round(r.value, 6) == 10.3 and r.period == "2026-09" and len(r.inputs) == 2
    assert round(compute("mom_pct@1", "k", s).value, 4) == round((110.3 / 109 - 1) * 100, 4)
    with pytest.raises(NotComputable):
        compute("yoy_pct@1", "k", _series({"2026-09": 1.0}))


def test_fiscal_year_sum_requires_every_month():
    s = _series({"2026-06": 999.0, "2026-07": -100.0, "2026-08": -200.0, "2026-09": 50.0})
    r = compute("fytd_sum@1", "k", s, fy_start_month=7)
    assert r.value == -250.0 and len(r.inputs) == 3
    with pytest.raises(NotComputable):
        compute("fytd_sum@1", "k", _series({"2026-07": 1.0, "2026-09": 1.0}), fy_start_month=7)


def test_fiscal_year_average_growth():
    s = _series({"2025-07": 100.0, "2025-08": 100.0, "2026-07": 110.0, "2026-08": 112.0})
    r = compute("fytd_mean_yoy_pct@1", "k", s, fy_start_month=7)
    assert round(r.value, 6) == 11.0 and len(r.inputs) == 4


def test_projection_years():
    s = _series({"2026": 3.0, "2027": 4.0})
    assert compute("calendar_year@1", "k", s, today=date(2026, 10, 30)).value == 3.0
    assert compute("next_calendar_year@1", "k", s, today=date(2026, 10, 30)).period == "2027"


def test_display():
    assert format_display(-874.4, {"decimals": 0, "prefix": "$", "suffix": "m"}) == "−$874m"
    assert format_display(0.67, {"decimals": 2, "suffix": "%", "plus": True}) == "+0.67%"
    assert format_display(450.4e9, {"decimals": 1, "prefix": "$", "suffix": "bn", "scale": 1e-9}) == "$450.4bn"
    assert format_display(-0.04, {"decimals": 1, "suffix": "%"}) == "0.0%"
    assert format_display(10.25, {"decimals": 1, "suffix": "%"}) == "10.3%"   # half up, as publishers round
    assert format_display(-0.05, {"decimals": 1, "suffix": "%"}) == "\u22120.1%"
    assert format_display(167193.4, {"decimals": 0}) == "167,193"


def test_periods():
    assert periods.to_period(date(2026, 9, 30), "monthly") == "2026-09"
    assert periods.shift("2026-01", -1) == "2025-12"
    assert periods.shift("2026-Q1", -4) == "2025-Q1"
    assert periods.fiscal_year_start("2026-03", 7) == "2025-07"
    assert periods.fiscal_year_label("2026-09", 7) == "FY27"
    assert periods.label("2026-09") == "Sep 2026"
    assert periods.label("2026-10-28") == "28 Oct 2026"
    assert periods.label("2026-09", "fytd", 7) == "Jul–Sep FY27"
    assert periods.label("2026-07", "fytd", 7) == "Jul FY27"
    assert periods.label("2027", "fy_end_year") == "FY27"


def test_fetcher_identifies_itself():
    from sourcepk.web import USER_AGENT, Fetcher
    assert Fetcher().session.headers["User-Agent"] == USER_AGENT


def test_chart_calculations():
    from sourcepk.transforms import fy_months, fy_years, recent
    s = _series({f"2025-{m:02d}": float(m) for m in range(7, 13)} | {f"2026-{m:02d}": 100.0 + m for m in range(1, 10)})
    r = recent("k", s, n=3)
    assert [p["period"] for p in r.points] == ["2026-07", "2026-08", "2026-09"]
    f = fy_months("k", s, fy_start_month=7)
    fy26 = [p for p in f.points if p["fy"] == "FY26"]; fy27 = [p for p in f.points if p["fy"] == "FY27"]
    assert len(fy26) == 12 and [p["month"] for p in fy27] == ["Jul", "Aug", "Sep"] and fy27[0]["index"] == 0
    a = _series({str(y): float(y - 2000) for y in range(2015, 2030)})
    g = fy_years("k", a, years_back=9, today=date(2026, 10, 30), fy_start_month=7)
    assert g.points[0]["label"] == "FY18" and g.points[-1]["label"] == "FY27"
    assert g.points[-1]["projection"] and not g.points[-2]["projection"]
