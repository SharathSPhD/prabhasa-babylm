# PRAJÑĀ keep-alive — honest mechanism + limitation

## What works
- **In-app cron** (CronCreate, hourly at :26): fires one research cycle per
  `research/cycles/run.md` while the Claude Code session is running and idle.
  This is the primary 24/7 driver AS LONG AS Claude is open. Authenticated
  (uses the live session's credentials). Auto-expires after 7 days (the mandate).
- Each cycle reads `research/memory/state.json` and resumes deterministically, so
  progress survives context compaction within the session.

## Known limitation (verified 2026-06-08)
- **Headless `claude -p` fails with 401** (no API credentials in an unattended
  subprocess — the interactive session uses OAuth/subscription auth the child can't
  read). So a true OS-cron keep-alive that survives Claude *exiting* is blocked
  unless an API key is provided.

## To enable true cross-restart 24/7 (optional, user action)
1. Export an Anthropic API key so headless runs authenticate:
   `export ANTHROPIC_API_KEY=sk-ant-...`  (add to ~/.bashrc)
2. Install the OS cron:
   `(crontab -l 2>/dev/null; echo "41 * * * * ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY bash /home/sharaths/projects/PSALM-integration/research/keepalive/prajna_keepalive.sh") | crontab -`
3. The script is lock-guarded (no double cycles) and self-removes after 2026-06-15.
   Remove manually: `crontab -l | grep -v prajna_keepalive | crontab -`

## Re-arm after a Claude restart (manual, 1 line)
Tell Claude: "resume the PRAJÑĀ loop" — it reads `research/memory/state.json` and
continues; re-creates the in-app cron if needed.
