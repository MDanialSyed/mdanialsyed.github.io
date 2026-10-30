# Source: Pakistan data pipeline

Downloads official statistics from the publishers' own data services, keeps every file
exactly as received, and turns them into the figures on the dashboard with named,
versioned calculations. Every published figure carries a lineage record back to the
archived file it came from.

## How a figure is made

1. **Download.** Each series in `registry/pakistan.toml` is requested from its publisher
   (State Bank of Pakistan EasyData, World Bank, IMF). Addresses are stored with any key
   replaced by `REDACTED`.
2. **Archive.** The file is saved untouched under `raw/<source>/<date>/` in the data folder,
   with a manifest giving its address, download time, size and SHA-256 fingerprint. Archived
   files are read-only and never overwritten.
3. **Store.** Values are added to `series/<series>.csv`. When a publisher revises a past
   value, the new value is added as a new row; the old one is kept.
4. **Calculate.** Each indicator names its calculation (for example `yoy_pct@1`). A
   calculation is never changed in place; a new behaviour gets a new version.
5. **Check.** Range, size of move since the last run, staleness, lineage (the archived
   file still matches its fingerprint), identities and cross-source checks. A failed
   blocking check keeps the value off the page.
6. **Publish.** Only indicators marked `confirmed` are written to
   `_data/pakistan/pipeline.json`. An indicator becomes confirmed after a run's values have
   been checked against the publisher's own release.
7. **Report.** Every run writes `runs/<run id>/report.md` and `report.json` in the data folder.

## Where it runs

The official sites are reached from the owner's Mac, which downloads and archives the files;
everything after that (calculations, checks, the report and the site update) runs from the
archive and needs no network.

- **One-time setup:** double-click `mac/setup.command` (placed next to a copy of this folder in
  `Documents/SourcePakistan`). It copies the pipeline to `~/SourcePakistan`, makes a private
  Python environment, asks once for the EasyData key and keeps it in the macOS Keychain, adds a
  weekly download job (Fridays 07:30, or on next wake) and runs the first download.
- **Weekly download:** `mac/fetch.sh`, run by that job, logs to `~/SourcePakistan/logs/fetch.log`.
- **Build and publish:** `python -m sourcepk --data ~/SourcePakistan/data --site <site data> build --publish`.

The key can instead be given for one session with `export SP_EASYDATA_KEY=...`. It is never
written to a file, a report or the site.

Commands: `doctor [--online]`, `fetch`, `build [--publish]`, `run [--publish]`, `status`,
`inspect <series>`, `verify-archive`, `export-registry`. Tests: `python -m pytest -q tests`.

## Adding or confirming an indicator

1. Find the series code in the publisher's catalogue and add it under `[[series]]`.
2. Add the indicator with its calculation, display format and check limits, status
   `needs-confirmation`.
3. Run the pipeline and compare the report's value with the publisher's release.
4. When they match, set both the series and the indicator to `confirmed`.
