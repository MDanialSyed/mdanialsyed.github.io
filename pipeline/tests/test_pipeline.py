"""End to end: download (from a fake server), archive, store, compute, check, publish, report."""
import dataclasses
import json
from datetime import date
from pathlib import Path

import pytest

from conftest import TEST_KEY, FakeFetcher, easydata_csv, imf_json, months, worldbank_json
from sourcepk import build, report
from sourcepk.archive import Archive
from sourcepk.checks import BLOCK, FLAG, HOLD, check_identities, check_indicator
from sourcepk.config import REPO_DIR, load_registry
from sourcepk.transforms import Result
from sourcepk.store import Row

TODAY = date(2026, 10, 30)


def cpi_values(bump: float = 0.0) -> dict[str, float]:
    """Synthetic year-on-year inflation rates, 4.0% in Jul 2024 rising 0.25 a month to 10.5% in Sep 2026."""
    ps = months("2024-07", 27)
    return {p: round(4.0 + 0.25 * i, 2) + (bump if p == "2026-09" else 0.0) for i, p in enumerate(ps)}


def routes(cpi=None, ca_last=-356.0):
    ms = months("2024-07", 27)
    ca = {p: -100.0 for p in ms}
    ca["2026-09"] = ca_last
    return {
        "TS_GP_PT_CPI_M.P00011516": easydata_csv("TS_GP_PT_CPI_M.P00011516", cpi or cpi_values()),
        "TS_GP_RL_LSM1516_M.LSM000160000": easydata_csv("TS_GP_RL_LSM1516_M.LSM000160000",
                                                         {p: 100 + i for i, p in enumerate(ms)}),
        "TS_GP_BOP_BPM6SUM_M.P00010": easydata_csv("TS_GP_BOP_BPM6SUM_M.P00010", ca),
        "TS_GP_IR_REPOMR_D.ORR": easydata_csv("TS_GP_IR_REPOMR_D.ORR", {"2026-10-27": 11.2, "2026-10-28": 11.49}),
        "NY.GDP.MKTP.CD": worldbank_json("NY.GDP.MKTP.CD", {"2024": 3.7e11, "2025": 4.1e11}),
        "NGDP_RPCH": imf_json("NGDP_RPCH", {"2025": 2.6, "2026": 3.0, "2027": 3.6}),
        "PCPIPCH": imf_json("PCPIPCH", {"2025": 4.5, "2026": 7.0, "2027": 8.0}),
    }


def unconfirmed(reg):
    """The real registry with every confirmation undone, so tests choose what is confirmed."""
    for k, ind in reg.indicators.items():
        if ind.status == "confirmed":
            reg.indicators[k] = dataclasses.replace(ind, status="needs-confirmation")
    for k, s in reg.series.items():
        if s.status == "confirmed":
            reg.series[k] = dataclasses.replace(s, status="needs-confirmation")
    return reg


def confirmed(reg, *keys):
    reg = unconfirmed(reg)
    for k in keys:
        ind = reg.indicators[k]
        reg.indicators[k] = dataclasses.replace(ind, status="confirmed")
        for s in ind.series:
            reg.series[s] = dataclasses.replace(reg.series[s], status="confirmed")
    return reg


def run(reg, data, site, fetcher):
    log = build.RunLog(build.new_run_id())
    build.fetch(reg, data, log, fetcher=fetcher)
    build.compute_all(reg, data, log, today=TODAY)
    build.run_checks(reg, data, log, build.load_previous(site), TODAY)
    log.archive_problems = Archive(data).verify()
    build.publish(reg, log, site_data=site)
    folder = report.write(reg, log, data, build.load_interim(site), True)
    return log, folder


def all_text(root: Path) -> str:
    return "\n".join(p.read_bytes().decode("utf-8", "replace") for p in root.rglob("*") if p.is_file())


def test_real_registry_is_consistent_with_the_dashboard():
    reg = load_registry()
    assert reg.validate() == []
    page = {r["key"] for r in json.loads((REPO_DIR / "_data/pakistan/indicators.json").read_text())}
    missing = [k for k, i in reg.indicators.items() if i.section != "reference" and k not in page]
    assert missing == []


def test_display_specs_reproduce_the_page():
    """For every mapped indicator, the registry's unit and format reproduce the figure now on the page."""
    reg = load_registry()
    page = {r["key"]: r for r in json.loads((REPO_DIR / "_data/pakistan/indicators.json").read_text())}
    from sourcepk.transforms import format_display
    for k, ind in reg.indicators.items():
        if ind.status != "unmapped" and k in page and page[k]["value"] is not None:
            assert format_display(page[k]["value"], ind.display or {}) == page[k]["display"], k


def test_end_to_end(tmp_path, key):
    data, site = tmp_path / "data", tmp_path / "site"
    reg = confirmed(load_registry(), "cpi_yoy", "lsm_fytd", "ca_m", "ca_fytd")
    fetcher = FakeFetcher(routes())
    log, folder = run(reg, data, site, fetcher)

    assert log.errors == []
    assert any(TEST_KEY in u for u in fetcher.requested)  # the key is sent to the publisher...
    assert TEST_KEY not in all_text(data) and TEST_KEY not in all_text(site)  # ...and never stored

    doc = json.loads((site / "pipeline.json").read_text())
    assert set(doc["indicators"]) == {"cpi_yoy", "lsm_fytd", "ca_m", "ca_fytd"}
    cpi = doc["indicators"]["cpi_yoy"]
    assert cpi["period"] == "2026-09" and cpi["period_label"] == "Sep 2026"
    assert cpi["display"] == "10.5%"
    assert cpi["calc"] == "as_published@1" and cpi["series_code"] == "TS_GP_PT_CPI_M.P00011516"
    assert cpi["context"] == "10.3% in Aug; 7.5% a year earlier"
    assert len(cpi["context_items"]) == 2 and cpi["context_items"][1]["period"] == "2025-09"
    assert doc["indicators"]["ca_m"]["context"] == "\u2212$100m in Aug"
    assert doc["indicators"]["ca_fytd"]["context"] == "\u2212$300m a year earlier"
    assert doc["indicators"]["lsm_fytd"]["period_label"] == "Jul\u2013Sep FY27"
    for inp in cpi["inputs"]:
        assert (data / inp["file"]).exists()
    assert doc["indicators"]["ca_fytd"]["display"] == "−$556m"
    assert doc["indicators"]["ca_fytd"]["period_label"] == "Jul–Sep FY27"
    assert len(doc["indicators"]["ca_fytd"]["inputs"]) == 3

    held = {f.indicator for f in log.findings if f.level == HOLD}
    assert {"lsm_m", "repo_on", "imf_gdp_proj"} <= held
    assert "lsm_m" in log.results and "lsm_m" not in doc["indicators"]
    assert "cpi_mom" not in log.results  # unmapped: no series code yet
    reg_doc = json.loads((site / "registry.json").read_text())
    assert any(s["publisher"] == "State Bank of Pakistan" for s in reg_doc["sources"])
    md = (folder / "report.md").read_text()
    assert "cpi_yoy" in md and "Every archived file matches" in md


def test_revisions_and_blocks(tmp_path, key):
    data, site = tmp_path / "data", tmp_path / "site"
    reg = confirmed(load_registry(), "cpi_yoy", "ca_m")
    run(reg, data, site, FakeFetcher(routes()))
    first = json.loads((site / "pipeline.json").read_text())["indicators"]

    # The publisher revises September CPI, and a broken current-account file arrives.
    log, folder = run(reg, data, site, FakeFetcher(routes(cpi=cpi_values(bump=1.0), ca_last=99999.0)))
    assert log.changes["sbp_cpi_yoy_national"].revised[0][0] == "2026-09"
    assert any(f.level == BLOCK and f.indicator == "ca_m" and f.check == "range" for f in log.findings)
    doc = json.loads((site / "pipeline.json").read_text())["indicators"]
    assert doc["ca_m"]["value"] == first["ca_m"]["value"] and doc["ca_m"]["carried_forward"]
    assert doc["cpi_yoy"]["value"] != first["cpi_yoy"]["value"]
    assert "Revisions by the publisher" in (folder / "report.md").read_text()


def test_tampered_archive_blocks(tmp_path, key):
    data, site = tmp_path / "data", tmp_path / "site"
    reg = confirmed(load_registry(), "cpi_yoy")
    log, _ = run(reg, data, site, FakeFetcher(routes()))
    f = data / log.results["cpi_yoy"].inputs[0][1].raw_path
    f.chmod(0o644)
    f.write_bytes(b"edited")
    log2 = build.RunLog("t")
    build.compute_all(reg, data, log2, today=TODAY)
    build.run_checks(reg, data, log2, {}, TODAY)
    assert any(x.level == BLOCK and x.check == "lineage" for x in log2.findings)


def test_missing_key_is_reported_not_fatal(tmp_path, monkeypatch):
    monkeypatch.delenv("SP_EASYDATA_KEY", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: None)
    log = build.RunLog("t")
    build.fetch(load_registry(), tmp_path, log, fetcher=FakeFetcher(routes()))
    assert any("MissingSecret" in e for e in log.errors)
    assert "wb_gdp_usd" in log.fetched  # sources without a key still download


def test_jump_and_staleness_flags():
    reg = load_registry()
    ind = reg.indicators["cpi_yoy"]
    row = Row("2026-03", 20.0, "2026-04-01T00:00:00Z", "", "")
    res = Result(20.0, "2026-03", [("sbp_cpi_yoy_national", row)])
    found = check_indicator(ind, res, Path("/nonexistent"), {"period": "2026-02", "value": 10.0}, TODAY)
    checks = {(f.level, f.check) for f in found}
    assert (FLAG, "jump") in checks and (FLAG, "staleness") in checks and (BLOCK, "lineage") in checks


def test_identities():
    reg = load_registry()
    reg.identities = [{"name": "a = b - c", "lhs": "a", "rhs": ["b", "-c"], "tolerance": 0.5}]
    r = lambda v: Result(v, "2026-09", [])
    assert check_identities(reg, {"a": r(5.0), "b": r(8.0), "c": r(3.0)}) == []
    bad = check_identities(reg, {"a": r(9.0), "b": r(8.0), "c": r(3.0)})
    assert bad and bad[0].level == BLOCK


def test_download_then_build_split(tmp_path, key):
    """The Mac downloads; the build runs later elsewhere and still reports what was downloaded."""
    data = tmp_path / "data"
    reg = load_registry()
    log = build.RunLog("20261030T060000Z")
    build.fetch(reg, data, log, fetcher=FakeFetcher(routes()))
    build.save_fetch_log(log, data)
    later = build.RunLog("20261030T100000Z")
    assert build.load_fetch_log(later, data) == "20261030T060000Z"
    assert later.fetched.keys() == log.fetched.keys()
    assert later.changes["sbp_cpi_yoy_national"].new == log.changes["sbp_cpi_yoy_national"].new
    assert TEST_KEY not in all_text(data)
