#!/usr/bin/env bash
# Pin and clone the official BabyLM 2025 evaluation pipeline (not vendored in git).
# Usage: bash scripts/setup_babylm_eval_pipeline.sh [DEST_DIR]
set -euo pipefail

REPO="https://github.com/babylm/evaluation-pipeline-2025.git"
COMMIT="bf55c1131e53654c2a87418f5629c26959acd710"
DEST="${1:-vendor/babylm-evaluation-pipeline-2025}"

if [[ -d "${DEST}/.git" ]]; then
  echo "Updating existing clone at ${DEST}"
  git -C "${DEST}" fetch --depth 1 origin "${COMMIT}" 2>/dev/null || git -C "${DEST}" fetch origin
  git -C "${DEST}" checkout "${COMMIT}"
else
  echo "Cloning ${REPO} @ ${COMMIT} -> ${DEST}"
  git clone --depth 1 "${REPO}" "${DEST}"
  git -C "${DEST}" fetch --depth 1 origin "${COMMIT}"
  git -C "${DEST}" checkout "${COMMIT}"
fi

echo "Installed. Export: export BABYLM_EVAL_ROOT=$(cd "${DEST}" && pwd)"
echo "Install pipeline deps: pip install -r ${DEST}/requirements.txt (requires psalm[ml] stack)"
