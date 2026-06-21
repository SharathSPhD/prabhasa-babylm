#!/usr/bin/env bash
# Runs ON the RunPod pod (after sync_to_pod.sh). Builds the x86/cu12 env + eval pipeline.
# Usage (from GB10):  ssh root@HOST -p PORT 'bash -s' < scripts/runpod/bootstrap_pod.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/psalm

echo "=== [bootstrap] env ==="; uname -m; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
command -v uv >/dev/null 2>&1 || { echo "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
uv --version

echo "=== [bootstrap] panini-data-toolkit (editable path dep; clone if missing) ==="
if [ ! -d /workspace/panini-data-toolkit/panini_data_toolkit ] && [ ! -d /workspace/panini-data-toolkit/src ]; then
  git clone --depth 1 https://github.com/SharathSPhD/panini-data-toolkit.git /workspace/panini-data-toolkit 2>&1 | tail -2
fi

echo "=== [bootstrap] uv sync --extra ml --extra stats (x86 cu12 env; torch is in the ml extra) ==="
uv sync --extra ml --extra stats 2>&1 | tail -6
uv run --no-sync python -c "import torch;print('torch',torch.__version__,'cuda_ok',torch.cuda.is_available(),torch.version.cuda)"
uv run --no-sync python -c "import psalm;print('psalm import OK')"

echo "=== [bootstrap] eval pipeline ==="
mkdir -p vendor
if [ ! -d vendor/babylm-evaluation-pipeline-2026 ]; then
  git clone --depth 1 https://github.com/babylm-org/babylm-eval.git vendor/babylm-evaluation-pipeline-2026 2>&1 | tail -3
fi
ls vendor/babylm-evaluation-pipeline-2026 | head
echo "BOOTSTRAP_BASE_DONE  (eval-data download + 192 fix handled interactively after layout check)"
