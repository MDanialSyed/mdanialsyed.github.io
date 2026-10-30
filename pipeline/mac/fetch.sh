#!/bin/bash
# Weekly download on this Mac: fetch every mapped series from the official sources and archive it
# in ~/SourcePakistan/data. Calculations, checks and publishing run afterwards from the archive.
ROOT="$HOME/SourcePakistan"
cd "$ROOT/code/pipeline" || exit 1
echo "== $(date '+%Y-%m-%d %H:%M %Z') =="
"$ROOT/.venv/bin/python" -m sourcepk --data "$ROOT/data" fetch
status=$?
date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/data/runs/LAST_ATTEMPT" 2>/dev/null || true
exit $status
