#!/usr/bin/env bash
# GB10 full-stack validation driver (ADR-0022).
# Prefer host smoke when Docker is slow/unavailable; optional container verify.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
LOG_DIR="${LOG_DIR:-$ROOT/docs/data}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/gb10-validation-log-2026-06.md}"
DOCKER_TIMEOUT_SEC="${DOCKER_TIMEOUT_SEC:-1800}"
IMAGE_TAG="${IMAGE_TAG:-psalm-gb10-verified:local}"
RUN_DOCKER="${RUN_DOCKER:-0}"

exec 1> >(tee -a "${LOG_FILE}.tmp")
exec 2>&1

section() { echo -e "\n## $1\n"; echo "_$(date -Iseconds)_"; }

section "Host inventory"
uname -m
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv 2>/dev/null || nvidia-smi | head -20

section "uv sync (dev + ml)"
uv sync --extra dev --extra ml --extra stats --extra verification

section "Optional stack wheels (host)"
uv pip install "vidyut==0.4.0" || true
# Unsloth: never let pip replace NGC/uv torch 2.12+cu130
uv pip install \
  "unsloth==2026.5.10" "unsloth-zoo==2026.5.5" "bitsandbytes==0.49.2" \
  "cut-cross-entropy" tyro docstring-parser msgspec hf-transfer \
  --no-deps 2>/dev/null || true

section "flash-attn build attempt (host, 600s cap)"
FLASH_LOG="/tmp/gb10-flash-attn-build.log"
echo "Log: $FLASH_LOG"
if timeout 600 uv pip install "flash-attn==2.8.3" --no-build-isolation 2>&1 | tee "$FLASH_LOG"; then
  echo "flash-attn: PASS (host build/install)"
else
  echo "flash-attn: FALLBACK — see tail of $FLASH_LOG"
  tail -30 "$FLASH_LOG" || true
fi

section "smoke.py (host)"
uv run python infra/dgx_spark/smoke.py "$@"
SMOKE_RC=$?

section "make gate (dev only, no GPU)"
uv sync --extra dev
uv run pytest -q
GATE_RC=$?

if [[ "$RUN_DOCKER" == "1" ]]; then
  section "Docker build ($IMAGE_TAG, timeout ${DOCKER_TIMEOUT_SEC}s)"
  if timeout "$DOCKER_TIMEOUT_SEC" docker build \
    -f infra/dgx_spark/Dockerfile.verified \
    -t "$IMAGE_TAG" \
    "$ROOT"; then
    section "Container smoke"
    docker run --rm --gpus all "$IMAGE_TAG"
  else
    echo "Docker build timed out or failed; host smoke remains authoritative."
  fi
else
  echo "Skipping Docker build (RUN_DOCKER=0). Set RUN_DOCKER=1 to verify Dockerfile.verified."
fi

section "Summary"
echo "- smoke.py exit: $SMOKE_RC"
echo "- pytest exit: $GATE_RC"
mkdir -p "$LOG_DIR"
if [[ -f "${LOG_FILE}.tmp" ]]; then
  {
    echo "# GB10 validation log (auto-generated)"
    echo ""
    echo "Host: $(uname -n) $(uname -m)"
    echo ""
    cat "${LOG_FILE}.tmp"
  } > "$LOG_FILE"
  rm -f "${LOG_FILE}.tmp"
  echo "Wrote $LOG_FILE"
fi

if [[ "$SMOKE_RC" -ne 0 ]]; then
  exit "$SMOKE_RC"
fi
exit "$GATE_RC"
