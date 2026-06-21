#!/usr/bin/env bash
# Rsync the training essentials from GB10 to a RunPod pod (working branch must NOT be
# pushed to the public origin, so we rsync the working tree). Eval pipeline + its data are
# set up on the pod by bootstrap_pod.sh (clone + download_evals.py), not rsynced.
#
# Usage: bash scripts/runpod/sync_to_pod.sh <HOST> <PORT> [KEY]
set -euo pipefail
HOST="${1:?pod host (publicIp)}"; PORT="${2:?ssh port}"; KEY="${3:-$HOME/.ssh/id_ed25519_sharathsphd}"
REPO=/home/sharaths/projects/PSALM-integration
TK=/home/sharaths/projects/panini-data-toolkit
SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
RS="rsync -az --info=progress2 -e \"$SSH\""

# Sibling layout on the pod: /workspace/psalm + /workspace/panini-data-toolkit so the
# pyproject path dep ../panini-data-toolkit resolves.
$SSH "root@$HOST" "mkdir -p /workspace/psalm /workspace/panini-data-toolkit"

# 1) repo code (exclude venv/git/big data/vendor/checkpoints; keep src/scripts/configs/tests/pyproject)
eval $RS --delete \
  --exclude='.git' --exclude='.venv' --exclude='vendor' --exclude='data/checkpoints' \
  --exclude='data/hf_export' --exclude='data/official_scores' --exclude='__pycache__' \
  --exclude='.mypy_cache' --exclude='.ruff_cache' --exclude='.pytest_cache' \
  "$REPO/" "root@$HOST:/workspace/psalm/"

# 2) panini-data-toolkit (editable path dep)
eval $RS --delete --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
  "$TK/" "root@$HOST:/workspace/panini-data-toolkit/"

# 3) corpora + tokenizer (the training inputs)
$SSH "root@$HOST" "mkdir -p /workspace/psalm/data/corpora /workspace/psalm/data/tokenizer"
eval $RS "$REPO/data/corpora/strict_small/" "root@$HOST:/workspace/psalm/data/corpora/strict_small/"
eval $RS "$REPO/data/corpora/strict/"       "root@$HOST:/workspace/psalm/data/corpora/strict/"
eval $RS "$REPO/data/tokenizer/"            "root@$HOST:/workspace/psalm/data/tokenizer/"
echo "SYNC_TO_POD DONE"
