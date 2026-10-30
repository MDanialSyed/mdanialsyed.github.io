#!/bin/bash
# Source: Pakistan: one-time setup on this Mac. Double-click to run.
#
#  1. Copies the pipeline to ~/SourcePakistan and makes a private Python environment for it.
#  2. Asks once for your EasyData key and saves it in your macOS Keychain (never in a file).
#  3. Adds a weekly job that downloads the official data every Friday at 07:30
#     (or when the Mac next wakes, if it was asleep).
#  4. Runs the first download now.
#
# To undo: launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/pk.sourcepakistan.fetch.plist
#          then delete that file and the ~/SourcePakistan folder.
set -euo pipefail

ROOT="$HOME/SourcePakistan"
HERE="$(cd "$(dirname "$0")" && pwd)"
LABEL="pk.sourcepakistan.fetch"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

echo
echo "Source: Pakistan setup"
echo "======================"

mkdir -p "$ROOT/data" "$ROOT/logs" "$ROOT/code"
rsync -a --delete --exclude "__pycache__" "$HERE/code/" "$ROOT/code/"
echo "1/4  Pipeline copied to $ROOT/code"

# Python 3.9 or newer: Anaconda first, then Homebrew, then the system's.
PY=""
for c in "$HOME/anaconda3/bin/python3" "/opt/anaconda3/bin/python3" "$HOME/opt/anaconda3/bin/python3" \
         "$HOME/miniconda3/bin/python3" "/opt/miniconda3/bin/python3" "/opt/homebrew/bin/python3" \
         "/usr/local/bin/python3" "/usr/bin/python3"; do
  if [ -x "$c" ] && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "No Python 3.9+ found. Install Anaconda or Python from python.org, then run this again."
  read -r -p "Press Enter to close." _; exit 1
fi
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  "$PY" -m venv "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -m pip install -q --upgrade pip >/dev/null 2>&1 || true
"$ROOT/.venv/bin/python" -m pip install -q -r "$ROOT/code/pipeline/requirements.txt"
echo "2/4  Python environment ready ($("$ROOT/.venv/bin/python" --version))"

if security find-generic-password -s source-pakistan-easydata >/dev/null 2>&1; then
  echo "3/4  EasyData key already saved in your Keychain"
else
  echo
  echo "Paste your new EasyData API key and press Enter (nothing will show as you paste):"
  read -r -s KEY
  echo
  KEY="$(printf '%s' "$KEY" | tr -d '[:space:]')"
  if [ -z "$KEY" ]; then echo "No key entered; run this again to add it."; read -r -p "Press Enter to close." _; exit 1; fi
  security add-generic-password -U -s source-pakistan-easydata -a "$USER" -T /usr/bin/security -w "$KEY"
  unset KEY
  echo "3/4  EasyData key saved in your Keychain"
fi

mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$ROOT/code/pipeline/mac/fetch.sh</string></array>
  <key>StartCalendarInterval</key>
  <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>30</integer></dict>
  <key>StandardOutPath</key><string>$ROOT/logs/fetch.log</string>
  <key>StandardErrorPath</key><string>$ROOT/logs/fetch.log</string>
</dict>
</plist>
PL
launchctl bootout "gui/$(id -u)" "$PLIST" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
TZNAME="$(readlink /etc/localtime | sed 's#.*/zoneinfo/##')"
printf '{"timezone": "%s", "set_up": "%s", "python": "%s"}\n' "$TZNAME" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PY" > "$ROOT/setup.json"
echo "4/4  Weekly download scheduled: Fridays 07:30 ($TZNAME)"

echo
echo "Running the first download now..."
/bin/bash "$ROOT/code/pipeline/mac/fetch.sh" | tee -a "$ROOT/logs/fetch.log"
echo
echo "All set. You can close this window."
