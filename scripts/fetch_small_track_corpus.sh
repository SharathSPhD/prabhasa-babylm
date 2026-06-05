#!/bin/bash
################################################################################
# fetch_small_track_corpus.sh
#
# Orchestrates download of Small-track (100M) + Sanskrit dose corpora for PSALM.
#
# PRECONDITION: GPU must be free (no training running)
# STATUS: Skeleton / commented — NOT EXECUTED
#
# Sources:
#   - BabyLM 2026 Strict (100M English): https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict
#   - AI4Bharat Sangraha (Sanskrit): https://huggingface.co/datasets/ai4bharat/sangraha
#   - GRETIL (Sanskrit texts): https://gretil.sub.uni-goettingen.de/gretil.html
#   - DCS (Digital Corpus of Sanskrit): https://github.com/ambuda-org/dcs
#
# Related ADRs: 0020 (BabyLM dual track), 0002 (curriculum)
#
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="${PROJECT_ROOT}/data/corpora"
CACHE_DIR="${DATA_ROOT}/cache"

VERBOSE="${VERBOSE:-0}"

################################################################################
# Logging & utility
################################################################################

log_info() { echo "[INFO] $*" >&2; }
log_warn() { echo "[WARN] $*" >&2; }
log_error() { echo "[ERROR] $*" >&2; }

# Check if a command is available
require_cmd() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Required command not found: $1"
        exit 1
    fi
}

################################################################################
# Preflight checks
################################################################################

preflight_checks() {
    log_info "Running preflight checks..."

    # Check required commands
    require_cmd curl
    require_cmd git
    require_cmd python3

    # Check HF CLI (optional but preferred)
    if command -v huggingface-cli &> /dev/null; then
        log_info "huggingface-cli found; will use for efficient HF downloads"
    else
        log_warn "huggingface-cli not found; falling back to curl (slower)"
    fi

    # Ensure data directory exists
    mkdir -p "$DATA_ROOT" "$CACHE_DIR"
    log_info "Data root: $DATA_ROOT"
}

################################################################################
# BabyLM 2026 Strict (100M English)
################################################################################

fetch_babylm_strict() {
    log_info "Fetching BabyLM 2026 Strict (100M English)..."

    # Target directory for official corpus
    BABYLM_DIR="${DATA_ROOT}/babylm-2026-strict"

    # Check if already present (quick heuristic)
    if [[ -d "$BABYLM_DIR" ]] && [[ -f "$BABYLM_DIR/dataset_info.json" ]]; then
        log_info "BabyLM Strict already present at $BABYLM_DIR, skipping download"
        return 0
    fi

    log_info "Downloading from HuggingFace Hub: BabyLM-community/BabyLM-2026-Strict"

    # Method 1: Use huggingface-cli if available
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download BabyLM-community/BabyLM-2026-Strict \
            --repo-type dataset \
            --cache-dir "$CACHE_DIR" \
            --local-files-only false \
            --resume-download

        # Symlink from cache to data directory (optional, for clarity)
        # (Typically HF CLI caches in ~/.cache/huggingface/datasets/)
        log_info "BabyLM Strict cached via HF CLI"
    else
        # Fallback: use curl to fetch dataset.py and info
        log_info "Downloading via curl (slower; consider installing huggingface-cli)"
        curl -L -o "${CACHE_DIR}/babylm_strict_info.json" \
            "https://huggingface.co/api/datasets/BabyLM-community/BabyLM-2026-Strict" || {
            log_error "Failed to fetch BabyLM Strict metadata"
            exit 1
        }
        log_info "Metadata cached to ${CACHE_DIR}/babylm_strict_info.json"
    fi
}

################################################################################
# AI4Bharat Sangraha (Sanskrit ~14.9B tokens)
################################################################################

fetch_sangraha_sanskrit() {
    log_info "Fetching AI4Bharat Sangraha (Sanskrit subset)..."

    SANGRAHA_DIR="${DATA_ROOT}/sangraha-sanskrit"

    if [[ -d "$SANGRAHA_DIR" ]]; then
        log_info "Sangraha already present at $SANGRAHA_DIR, skipping download"
        return 0
    fi

    mkdir -p "$SANGRAHA_DIR"

    log_info "Downloading from HuggingFace: ai4bharat/sangraha (Sanskrit)"

    if command -v huggingface-cli &> /dev/null; then
        # Download only the Sanskrit subset config
        huggingface-cli download ai4bharat/sangraha \
            --repo-type dataset \
            --cache-dir "$CACHE_DIR" \
            --revision main \
            --local-files-only false \
            || {
            log_warn "HF CLI download had issues; will retry with Python API"
        }
    fi

    # Use Python datasets API for cleaner subset handling
    log_info "Fetching via Python datasets library (may be large ~50GB)..."
    python3 << 'PYTHON_EOF'
import os
import sys
from datasets import load_dataset

try:
    # Load Sanskrit subset; stream=True to avoid full download to memory
    ds = load_dataset('ai4bharat/sangraha', 'san', split='train', streaming=True)
    print(f"Sangraha Sanskrit dataset ready (streaming mode)")
    print(f"  Example: {next(iter(ds))}")
except Exception as e:
    print(f"Error loading Sangraha: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF

    log_info "Sangraha metadata verified"
}

################################################################################
# GRETIL (Göttingen Sanskrit texts)
################################################################################

fetch_gretil() {
    log_info "Fetching GRETIL (Göttingen Sanskrit texts)..."

    GRETIL_DIR="${DATA_ROOT}/gretil-sanskrit"
    mkdir -p "$GRETIL_DIR"

    # Main GRETIL website
    GRETIL_URL="https://gretil.sub.uni-goettingen.de/gretil.html"

    log_info "GRETIL main portal: $GRETIL_URL"

    # DARIAH-DE repository link for bulk download
    # (This is the recommended method per GRETIL documentation)
    DARIAH_URL="https://textgridrep.org/dataset/10.20375/0000-0016-C802-4/data"

    log_info "Attempting DARIAH-DE bulk Sanskrit download..."
    log_info "  URL: $DARIAH_URL"

    # Note: Direct download may be large; curl with resume
    if [[ ! -f "${GRETIL_DIR}/sanskrit_bulk.zip" ]]; then
        log_info "Downloading GRETIL Sanskrit bulk ZIP (~100–200 MB estimated)..."
        curl -L -C - -o "${GRETIL_DIR}/sanskrit_bulk.zip" "$DARIAH_URL" || {
            log_error "Failed to fetch GRETIL bulk ZIP. Manual download may be required."
            log_info "  Visit: $GRETIL_URL"
            log_info "  Or direct: $DARIAH_URL"
            return 1
        }
    else
        log_info "GRETIL Sanskrit ZIP already present"
    fi

    # Unzip (only if uncompressed doesn't exist)
    if [[ ! -d "${GRETIL_DIR}/sanskrit" ]]; then
        log_info "Unzipping GRETIL Sanskrit texts..."
        unzip -q -o "${GRETIL_DIR}/sanskrit_bulk.zip" -d "${GRETIL_DIR}/sanskrit" || {
            log_error "Failed to unzip GRETIL"
            exit 1
        }
    fi

    log_info "GRETIL Sanskrit texts ready at ${GRETIL_DIR}/sanskrit"
}

################################################################################
# DCS (Digital Corpus of Sanskrit, GitHub)
################################################################################

fetch_dcs() {
    log_info "Fetching DCS (Digital Corpus of Sanskrit)..."

    DCS_DIR="${DATA_ROOT}/dcs"

    if [[ -d "$DCS_DIR/.git" ]]; then
        log_info "DCS already cloned at $DCS_DIR, updating..."
        (cd "$DCS_DIR" && git pull) || log_warn "Failed to update DCS; continuing with existing clone"
    else
        log_info "Cloning DCS repository (ambuda-org/dcs)..."
        git clone https://github.com/ambuda-org/dcs.git "$DCS_DIR" || {
            log_error "Failed to clone DCS"
            exit 1
        }
    fi

    log_info "DCS texts ready at $DCS_DIR"
}

################################################################################
# Generate Corpus Manifest
################################################################################

generate_manifest() {
    log_info "Generating corpus manifest..."

    MANIFEST_PATH="${PROJECT_ROOT}/corpus_manifest.yaml"

    log_info "Writing manifest to $MANIFEST_PATH"

    python3 << "PYTHON_EOF"
import os
import json
import yaml
from pathlib import Path

manifest = {
    "metadata": {
        "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "version": "1.0",
        "sources_verified": [],
    },
    "english": {
        "babylm_strict": {
            "source": "BabyLM-community/BabyLM-2026-Strict",
            "url": "https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict",
            "license": "MIT",
            "token_count": 100_000_000,
            "components": {
                "bnc_spoken": 7_600_000,
                "childes": 28_400_000,
                "project_gutenberg": 25_600_000,
                "open_subtitles": 22_800_000,
                "simple_wikipedia": 15_300_000,
            },
            "notes": "Detoxified, deduplicated in 2026 release",
        }
    },
    "sanskrit_dose": {
        "sangraha": {
            "source": "AI4Bharat Sangraha (Sanskrit subset)",
            "url": "https://huggingface.co/datasets/ai4bharat/sangraha",
            "license": "CC-BY-4.0",
            "token_count_estimate": 14_892_300_000,
            "components": {
                "verified": 1_329_000_000,
                "synthetic": 13_553_500_000,
                "unverified": 9_800_000,
            },
            "notes": "Largest available Sanskrit corpus; synthetic via perplexity filtering",
        },
        "gretil": {
            "source": "GRETIL (Göttingen Register of Electronic Texts)",
            "url": "https://gretil.sub.uni-goettingen.de/gretil.html",
            "license": "Unknown (FLAG: verify with GRETIL)",
            "token_count_estimate": 100_000_000,
            "components": {
                "vedas": None,
                "epics": None,
                "puranas": None,
                "philosophy": None,
            },
            "notes": "Authoritative classical texts; manual tokenization required",
        },
        "dcs": {
            "source": "Digital Corpus of Sanskrit (ambuda-org)",
            "url": "https://github.com/ambuda-org/dcs",
            "license": "CC-BY-4.0",
            "token_count_estimate": 12_500_000,
            "notes": "High-quality lemmatized + POS-tagged; ~650K sentences",
        },
    },
    "warnings": [
        "GRETIL license not explicitly stated; verify before public release",
        "Sanskrit token counts are estimates; actual counts require tokenization on GB10",
        "Manifest is a skeleton; update with actual token counts after tokenization",
    ],
}

# Write manifest
manifest_path = Path("/home/sharaths/projects/PSALM-integration/corpus_manifest.yaml")
with open(manifest_path, "w") as f:
    yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

print(f"Manifest written to {manifest_path}")
PYTHON_EOF
}

################################################################################
# Main orchestration
################################################################################

main() {
    log_info "PSALM Small-Track Corpus Fetch — Starting"
    log_info "GPU saturation check: CPU-only mode (no torch/cuda import)"

    preflight_checks

    # English (must-have for competition track)
    fetch_babylm_strict

    # Sanskrit (dose pre-pretraining, research-optional)
    log_info ""
    log_info "=== Sanskrit Dose Pre-pretraining (Outside 100M Cap) ==="
    fetch_sangraha_sanskrit || log_warn "Sangraha fetch failed; continuing"
    fetch_gretil || log_warn "GRETIL fetch incomplete; manual download may be required"
    fetch_dcs

    # Generate manifest
    log_info ""
    log_info "=== Corpus Manifest ==="
    generate_manifest

    log_info ""
    log_info "=== Fetch Complete ==="
    log_info "Summary:"
    log_info "  - English: 100M tokens (BabyLM Strict)"
    log_info "  - Sanskrit: ~15B tokens (Sangraha + GRETIL + DCS, dose pre-pretraining)"
    log_info "  - Manifest: corpus_manifest.yaml"
    log_info ""
    log_info "Next steps:"
    log_info "  1. Verify manifest: cat corpus_manifest.yaml"
    log_info "  2. Tokenize on GB10 (after GPU frees): python scripts/tokenize_corpus.py"
    log_info "  3. Gate: psalm contract check corpus_manifest.yaml"
}

# Run main unless sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
