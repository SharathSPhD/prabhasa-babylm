#!/usr/bin/env bash
# PRAJÑĀ OS-level keep-alive: fires one autonomous research cycle, lock-guarded,
# expiry-bounded (1-week mandate). Install via user crontab (hourly).
# Removal: crontab -l | grep -v prajna_keepalive | crontab -
set -uo pipefail

PROJ=/home/sharaths/projects/PSALM-integration
LOG="$PROJ/research/keepalive/keepalive.log"
LOCK="$PROJ/research/memory/.cycle.lock"
PROMPT_FILE="$PROJ/research/keepalive/cycle_prompt.txt"
EXPIRY=20260615   # stop after this date (YYYYMMDD) — the 1-week autonomous mandate
CLAUDE=/home/sharaths/.local/bin/claude

mkdir -p "$PROJ/research/keepalive"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

# 1) Expiry guard — self-remove from crontab after the mandate window.
today=$(date +%Y%m%d)
if [ "$today" -ge "$EXPIRY" ]; then
  echo "$(ts) EXPIRED (>= $EXPIRY) — removing keepalive from crontab" >> "$LOG"
  crontab -l 2>/dev/null | grep -v 'prajna_keepalive' | crontab - 2>/dev/null
  exit 0
fi

# 2) Lock — never run two cycles at once (serialises in-session + cron + manual).
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(ts) cycle already running (lock held) — skip" >> "$LOG"
  exit 0
fi

# 3) Run one cycle headless, capped at 55 min so the next hourly fire is clean.
echo "$(ts) ===== PRAJÑĀ cycle start =====" >> "$LOG"
cd "$PROJ" || { echo "$(ts) cd failed" >> "$LOG"; exit 1; }
timeout 3300 "$CLAUDE" -p "$(cat "$PROMPT_FILE")" \
  --dangerously-skip-permissions >> "$LOG" 2>&1
rc=$?
echo "$(ts) ===== cycle end (exit $rc) =====" >> "$LOG"
# flock auto-released when fd 9 closes on exit.
