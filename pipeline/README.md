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

## Setup (macOS)

```sh
# 1. Python environment
conda create -n sourcepk python=3.12 -y
conda activate sourcepk
pip install -r pipeline/requirements.txt

# 2. Data folder, outside the repo
mkdir -p ~/SourcePakistan/data

# 3. EasyData key, stored in the Keychain (you are prompted for it; it is never echoed)
security add-generic-password -s source-pakistan-easydata -a "$USER" -w

# 4. Check the setup
cd pipeline
python -m sourcepk doctor --online
```

The key can instead be given for one session with `export SP_EASYDATA_KEY=...`. It is never
written to a file, a report or the site.

## Weekly run

```sh
cd pipeline
python -m sourcepk run              # dry run: download, check, report
python -m sourcepk run --publish    # also update the site data
python -m pytest -q tests           # the test suite
```

Other commands: `status`, `inspect <series>`, `verify-archive`, `export-registry`, `build`.

## Adding or confirming an indicator

1. Find the series code in the publisher's catalogue and add it under `[[series]]`.
2. Add the indicator with its calculation, display format and check limits, status
   `needs-confirmation`.
3. Run the pipeline and compare the report's value with the publisher's release.
4. When they match, set both the series and the indicator to `confirmed`.
