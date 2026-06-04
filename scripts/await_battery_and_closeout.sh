#!/usr/bin/env bash
# Poll until the strict-small battery finishes and the GPU frees, then run close-out.
# Safe to launch in the background; writes a heartbeat log and never touches a live run.
set -euo pipefail

ROOT="/home/sharaths/projects/PSALM-integration"
cd "$ROOT"
LOG="logs/closeout/await.log"
mkdir -p logs/closeout
POLL="${POLL:-600}"   # seconds between checks

battery_running () { pgrep -f "run_strict_small_battery.py" >/dev/null 2>&1; }
gpu_busy () { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; }
battery_done () { [ -f "data/checkpoints/strict_small/battery_summary.json" ]; }

echo "[await] watcher start $(date -Is) poll=${POLL}s" | tee -a "$LOG"
while true; do
  if battery_done && ! battery_running; then
    echo "[await] battery_summary.json present and no battery process -> proceeding" | tee -a "$LOG"
    break
  fi
  if ! battery_running && ! gpu_busy; then
    # Battery process gone and GPU idle even without summary (e.g. crash): still proceed
    echo "[await] no battery process and GPU idle -> proceeding (no summary)" | tee -a "$LOG"
    break
  fi
  echo "[await] $(date -Is) battery_running=$(battery_running && echo yes || echo no) gpu_busy=$(gpu_busy && echo yes || echo no)" >> "$LOG"
  sleep "$POLL"
done

# Wait a short grace period for the GPU to fully free before heavy eval.
for _ in $(seq 1 30); do gpu_busy || break; sleep 20; done

echo "[await] launching closeout $(date -Is)" | tee -a "$LOG"
bash scripts/closeout_battery.sh >> logs/closeout/closeout.log 2>&1 &
echo "[await] closeout pid $! ; watcher exiting" | tee -a "$LOG"
