#!/usr/bin/env python3
"""Work out the derived figures for Source: Pakistan from the headline indicators.

Official figures from the data pipeline (_data/pakistan/pipeline.json) take the
place of hand-collected ones wherever both exist.

Writes _data/pakistan/derived.json (year-to-date changes, real interest
rates, gaps to targets, peers' real rates). The page shows these figures but
never computes them, so every number on the site traces to a data file.

Usage:
    python3 scripts/pakistan/derive.py          # write derived.json
    python3 scripts/pakistan/derive.py --check  # fail if derived.json is stale
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "_data" / "pakistan"
MINUS = "−"


def load(name):
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def sign(x):
    return "+" if x > 0 else (MINUS if x < 0 else "")


def pct(x):
    return f"{sign(round(x * 100, 1))}{abs(x * 100):.1f}%"


def pp(x):
    r = round(x, 1)
    return f"{sign(r)}{abs(r):.1f} pp"


def tone(good):
    return "good" if good else "bad"


def indicators():
    """Hand-collected figures, with any official pipeline figure taking their place."""
    ind = {r["key"]: r for r in load("indicators")}
    path = DATA / "pipeline.json"
    if path.exists():
        for key, r in json.loads(path.read_text(encoding="utf-8")).get("indicators", {}).items():
            ind[key] = {**ind.get(key, {}), "value": r["value"], "display": r["display"]}
    return ind


def derive():
    ind = indicators()

    def v(key):
        r = ind.get(key)
        if r is None or r.get("value") is None:
            raise SystemExit(f"derive.py: indicator '{key}' is missing or has no numeric value")
        return float(r["value"])

    kse = v("kse100") / v("kse100_base") - 1
    pkr = v("usdpkr_base") / v("usdpkr") - 1  # positive = rupee stronger
    real_headline = v("policy_rate") - v("cpi_yoy")
    real_core = v("policy_rate") - v("core_urban")
    fbr_gap = v("fbr_fytd") - v("fbr_fytd_target")
    res_gap = v("res_goal") - v("sbp_res")
    fbr_share = v("fbr_fytd") / v("fbr_fy_target")

    rows = [
        {"key": "kse_ytd", "label": "KSE-100, change since the end of last year", "value": round(kse, 6),
         "display": f"{pct(kse)} YTD", "tone": tone(kse >= 0)},
        {"key": "pkr_ytd", "label": "Rupee against the dollar since the end of last year", "value": round(pkr, 6),
         "display": f"Rupee {pct(pkr)} YTD", "tone": tone(pkr >= 0)},
        {"key": "real_headline", "label": "Real policy rate vs headline CPI", "value": round(real_headline, 4),
         "display": pp(real_headline), "tone": tone(real_headline > 0)},
        {"key": "real_core", "label": "Real policy rate vs urban core CPI", "value": round(real_core, 4),
         "display": pp(real_core), "tone": tone(real_core > 0)},
        {"key": "fbr_gap", "label": "FBR collection vs target, fiscal year to date", "value": round(fbr_gap, 2),
         "display": f"{sign(fbr_gap)}Rs{abs(fbr_gap):,.1f}bn", "tone": tone(fbr_gap >= 0)},
        {"key": "res_gap", "label": "SBP reserves, gap to end-December goal", "value": round(res_gap, 4),
         "display": f"${res_gap:.2f}bn to go" if res_gap > 0 else "Reached", "tone": "neutral" if res_gap > 0 else "good"},
        {"key": "fbr_share", "label": "Share of this fiscal year's FBR target collected", "value": round(fbr_share, 6),
         "display": f"{fbr_share * 100:.1f}%", "tone": "neutral"},
    ]

    peers = []
    for p in load("peers"):
        r = float(p["policy_rate"]) - float(p["cpi"])
        r1 = round(r, 1)
        peers.append({"country": p["country"], "value": round(r, 4), "display": pp(r),
                      "tone": "good" if r1 > 0 else ("bad" if r1 < 0 else "neutral")})
    return {"figures": rows, "peer_real_rates": peers}


def main():
    out = DATA / "derived.json"
    data = derive()
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if "--check" in sys.argv:
        current = out.read_text(encoding="utf-8") if out.exists() else ""
        if current != text:
            print("derive.py: derived.json is out of date; run python3 scripts/pakistan/derive.py")
            sys.exit(1)
        print("derive.py: derived.json is up to date")
        return
    out.write_text(text, encoding="utf-8")
    for r in data["figures"]:
        print(f"{r['key']:>14}  {r['display']}")


if __name__ == "__main__":
    main()
