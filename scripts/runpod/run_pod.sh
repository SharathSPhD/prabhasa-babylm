#!/usr/bin/env bash
# Single, reviewable entrypoint for ALL RunPod pod interaction, so the user can grant a
# tight permission allowlist for exactly this script (not raw rsync/ssh globally).
#
# The agent routes every pod transfer/exec through this wrapper. Pod lifecycle (create/
# stop/delete) is done via the RunPod MCP, not here.
#
# Permission rule to add (project .claude/settings.local.json):
#   "permissions": { "allow": ["Bash(bash scripts/runpod/run_pod.sh:*)"] }
#
# Subcommands:
#   sync   HOST PORT              rsync code + panini-data-toolkit + corpora + tokenizer -> pod
#   bootstrap HOST PORT           ssh: build x86 env (uv sync) + clone eval pipeline
#   exec   HOST PORT "CMD"        ssh: run a command on the pod (training/eval jobs)
#   fetch  HOST PORT REMOTE LOCAL rsync: pull a path back from the pod (checkpoints/scores)
set -euo pipefail
SUB="${1:?subcommand: sync|bootstrap|exec|fetch}"; shift
HOST="${1:?pod publicIp}"; PORT="${2:?ssh port}"; shift 2
KEY="$HOME/.ssh/id_ed25519_sharathsphd"
SSHOPTS="-p $PORT -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30"
REPO=/home/sharaths/projects/PSALM-integration
TK=/home/sharaths/projects/panini-data-toolkit

case "$SUB" in
  sync)
    ssh $SSHOPTS "root@$HOST" "mkdir -p /workspace/psalm /workspace/panini-data-toolkit /workspace/psalm/data/corpora /workspace/psalm/data/tokenizer"
    rsync -rltz --no-owner --no-group --no-perms --omit-dir-times --info=progress2 -e "ssh $SSHOPTS" --delete \
      --exclude='.git' --exclude='.venv' --exclude='vendor' --exclude='data/checkpoints' \
      --exclude='data/hf_export' --exclude='data/official_scores' --exclude='__pycache__' \
      --exclude='.mypy_cache' --exclude='.ruff_cache' --exclude='.pytest_cache' \
      "$REPO/" "root@$HOST:/workspace/psalm/"
    rsync -rltz --no-owner --no-group --no-perms --omit-dir-times --info=progress2 -e "ssh $SSHOPTS" --delete --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
      "$TK/" "root@$HOST:/workspace/panini-data-toolkit/"
    rsync -rltz --no-owner --no-group --no-perms --omit-dir-times --info=progress2 -e "ssh $SSHOPTS" "$REPO/data/corpora/strict_small/" "root@$HOST:/workspace/psalm/data/corpora/strict_small/"
    rsync -rltz --no-owner --no-group --no-perms --omit-dir-times --info=progress2 -e "ssh $SSHOPTS" "$REPO/data/corpora/strict/"       "root@$HOST:/workspace/psalm/data/corpora/strict/"
    rsync -rltz --no-owner --no-group --no-perms --omit-dir-times --info=progress2 -e "ssh $SSHOPTS" "$REPO/data/tokenizer/"            "root@$HOST:/workspace/psalm/data/tokenizer/"
    echo "RUN_POD sync DONE" ;;
  bootstrap)
    ssh $SSHOPTS "root@$HOST" "bash -s" < "$REPO/scripts/runpod/bootstrap_pod.sh" ;;
  exec)
    ssh $SSHOPTS "root@$HOST" "${1:?command}" ;;
  fetch)
    REMOTE="${1:?remote path}"; LOCAL="${2:?local path}"
    mkdir -p "$LOCAL"
    rsync -rltz --no-owner --no-group --no-perms --omit-dir-times --info=progress2 -e "ssh $SSHOPTS" "root@$HOST:$REMOTE" "$LOCAL" ;;
  *) echo "unknown subcommand: $SUB" >&2; exit 2 ;;
esac
