#!/bin/bash
# Weekly download on this Mac: fetch every mapped series from the official sources and archive it
# in ~/SourcePakistan/data. Calculations, checks and publishing run afterwards from the archive.
ROOT="$HOME/SourcePakistan"
cd "$ROOT/code/pipeline" || exit 1
# Several triggers can fire close together (schedule, wake, run-now); skip if one ran in the last 10 minutes.
STAMP="$ROOT/data/runs/LAST_ATTEMPT"
if [ -f "$STAMP" ] && [ $(( $(date +%s) - $(stat -f %m "$STAMP") )) -lt 600 ] && [ "${1:-}" != "--force" ]; then
  exit 0
fi
echo "== $(date '+%Y-%m-%d %H:%M %Z') =="
"$ROOT/.venv/bin/python" -m sourcepk --data "$ROOT/data" fetch
status=$?
date -u +%Y-%m-%dT%H:%M:%SZ > "$STAMP" 2>/dev/null || true
exit $status
