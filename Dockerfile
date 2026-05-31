# PSALM container for the DGX Spark (GB10 Grace-Blackwell, aarch64, CUDA 13).
#
# NOTE: pramana used nvcr.io/nvidia/pytorch:24.09-py3, which predates Blackwell
# (sm_121) + arm64 wheels. PSALM upgrades to a 25.x NGC PyTorch image that ships
# Blackwell-capable CUDA 13 and aarch64 builds. Verify the exact tag against the
# host driver (`nvidia-smi`) and NGC's Blackwell/arm64 support matrix before a
# long run; this is the Phase 0/1 de-risking item flagged in the plan.
ARG NGC_PYTORCH_TAG=25.04-py3
FROM nvcr.io/nvidia/pytorch:${NGC_PYTORCH_TAG}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, reproducible installs.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace/psalm

# Dependency layer (cached unless project metadata changes).
COPY pyproject.toml README.md ./
RUN mkdir -p src/psalm && \
    printf '"""PSALM package."""\n__version__ = "0.1.0"\n' > src/psalm/__init__.py && \
    uv sync --extra dev --extra stats --extra verification

# Full source.
COPY . .
RUN uv sync --extra dev --extra stats --extra verification

CMD ["/bin/bash"]
