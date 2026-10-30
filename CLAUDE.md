# danialsyed.com

Personal site and blog built with Jekyll (Clean Blog theme, Bootstrap 4) and
published by GitHub Pages straight from `master` at www.danialsyed.com.

- GitHub Pages builds the site itself, so only its whitelisted plugins work
  (jekyll-feed, jekyll-paginate, jekyll-sitemap). Don't add other plugins or
  a Gemfile change that needs them.
- Blog posts are in `_posts/`. Leave the theme's layouts and `_sass/` alone
  unless asked.

## Pakistan Vitals (`/pakistan/`)

A dashboard on Pakistan's economy plus a weekly note, refreshed every Friday
by `.github/workflows/pakistan-weekly.yml`, which follows the playbook in
`.claude/skills/pakistan-weekly-update/SKILL.md` and opens a pull request.
`PAKISTAN_VITALS.md` explains the setup.

| Where | What |
|---|---|
| `_data/pakistan/*.json` | All figures, each with its source link |
| `_pakistan_notes/` | Weekly notes (a Jekyll collection) |
| `pakistan/index.html` | The dashboard page |
| `pakistan/data.json`, `pakistan/feed.xml`, `pakistan/notes/` | Data download, notes feed, notes list |
| `_layouts/pakistan*.html`, `_includes/pakistan/` | Templates |
| `assets/pakistan/` | CSS and the d3 charts |
| `scripts/pakistan/` | `derive.py` (computed figures) and `check_data.py` (checks) |

Rules:

- Every number on the dashboard or in a note must come from
  `_data/pakistan/`, with a source and an https link. Never type a figure into
  a template or a note that isn't in the data.
- Never edit `derived.json` by hand. After changing data, run
  `python3 scripts/pakistan/derive.py`, then
  `python3 scripts/pakistan/check_data.py`, and fix anything it reports.
- Indicator keys are permanent names (`cpi_yoy`, `sbp_res`). Don't rename them
  or put a period in a key; the period goes in the row.
