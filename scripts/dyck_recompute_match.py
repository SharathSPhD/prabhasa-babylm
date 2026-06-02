"""Recompute Dyck control match vs sentence-level Saṃsādhanī diversity targets.

Writes ``docs/data/dyck-match-report.md`` and ``docs/data/dyck-match-result.json``.
Uses ``docs/data/vidyut-fixture-stats.json`` when present (U2); otherwise falls
back to ``docs/data/phase2-samsadhani-diversity.json`` and flags pending U2 integration.

    uv run python scripts/dyck_recompute_match.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from psalm.domain.data.diversity import summarize
from psalm.domain.data.dyck import DyckConfig, generate_corpus, hu_replication_config
from psalm.domain.data.matching import (
    byte_length_distance,
    byte_length_histogram,
    match_dyck,
    stat_distance,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"
VIDYUT_FIXTURE = DATA / "vidyut-fixture-stats.json"
SAMSADHANI = DATA / "phase2-samsadhani-diversity.json"
REPORT = DATA / "dyck-match-report.md"
RESULT_JSON = DATA / "dyck-match-result.json"

# Grid for H1 surface-stat matching (Phase 1 style); Hu replication config reported separately.
_CANDIDATES = [
    DyckConfig(bracket_types=2, max_depth=4, n_shuffles=1, min_len=8, max_len=64),
    DyckConfig(bracket_types=3, max_depth=6, n_shuffles=2, min_len=8, max_len=128),
    DyckConfig(bracket_types=4, max_depth=8, n_shuffles=2, min_len=12, max_len=160),
    DyckConfig(bracket_types=5, max_depth=8, n_shuffles=3, min_len=16, max_len=192),
    DyckConfig(bracket_types=8, max_depth=10, n_shuffles=3, min_len=16, max_len=256),
]


def _load_targets() -> tuple[dict[str, float], str, bool]:
    pending_u2 = False
    if VIDYUT_FIXTURE.is_file():
        raw = json.loads(VIDYUT_FIXTURE.read_text(encoding="utf-8"))
        if "samsadhani_sentences" in raw:
            block = raw["samsadhani_sentences"]
            source = f"{VIDYUT_FIXTURE.relative_to(ROOT)}#samsadhani_sentences"
        else:
            # U2 fixture stats published; Dyck DEFAULT_KEYS still track live Saṃsādhanī
            # until sentence-level parity with the fixture export is demonstrated.
            pending_u2 = True
            block = json.loads(SAMSADHANI.read_text(encoding="utf-8"))["samsadhani_sentences"]
            source = (
                f"{SAMSADHANI.relative_to(ROOT)}#samsadhani_sentences "
                f"(U2 {VIDYUT_FIXTURE.name} present; fixture-vs-live TTR reconciliation pending)"
            )
    else:
        pending_u2 = True
        raw = json.loads(SAMSADHANI.read_text(encoding="utf-8"))
        source = str(SAMSADHANI.relative_to(ROOT))
        block = raw["samsadhani_sentences"]
    ttr = block.get("word_type_token_ratio", block.get("type_token_ratio"))
    if ttr is None:
        raise KeyError("baseline block missing word_type_token_ratio / type_token_ratio")
    targets = {
        "type_token_ratio": float(ttr),
        "bigram_entropy": float(block["bigram_entropy"]),
        "trigram_entropy": float(block["trigram_entropy"]),
    }
    return targets, source, pending_u2


def main() -> None:
    targets, source, pending_u2 = _load_targets()
    n, seed = 200, 0
    matched = match_dyck(targets, _CANDIDATES, n=n, seed=seed)
    matched_corpus = generate_corpus(n, matched.config, seed=seed)
    matched_hist = byte_length_histogram(matched_corpus, bins=16)

    hu_cfg = hu_replication_config()
    hu_corpus = generate_corpus(n, hu_cfg, seed=seed)
    hu_stats = summarize(hu_corpus)
    hu_hist = byte_length_histogram(hu_corpus, bins=16)
    hu_dist = stat_distance(hu_stats, targets)
    hu_byte = byte_length_distance(hu_hist, matched_hist)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "baseline_source": source,
        "pending_u2_fixture": pending_u2,
        "targets": targets,
        "matched_config": {
            "bracket_types": matched.config.bracket_types,
            "max_depth": matched.config.max_depth,
            "n_shuffles": matched.config.n_shuffles,
            "min_len": matched.config.min_len,
            "max_len": matched.config.max_len,
        },
        "matched_distance": matched.distance,
        "matched_stats": matched.stats,
        "hu_replication_config": {
            "bracket_types": hu_cfg.bracket_types,
            "max_depth": hu_cfg.max_depth,
            "n_shuffles": hu_cfg.n_shuffles,
            "min_len": hu_cfg.min_len,
            "max_len": hu_cfg.max_len,
        },
        "hu_replication_distance": hu_dist,
        "hu_replication_stats": hu_stats,
        "hu_vs_matched_byte_hist_l1": hu_byte,
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    pending_note = (
        "**PENDING U2:** `vidyut-fixture-stats.json` is published but Dyck targets still "
        "use live Saṃsādhanī sentence stats until fixture-vs-live diversity parity is "
        "reconciled (fixture corpus TTR ≪ live).\n\n"
        if pending_u2
        else "**Baseline:** U2 `vidyut-fixture-stats.json#samsadhani_sentences`.\n\n"
    )
    report = f"""# Dyck surface-stat match report

{pending_note}Generated: `{payload["generated_at"]}`

## Targets (DEFAULT_KEYS)

| Key | Value |
|-----|-------|
| type_token_ratio | {targets["type_token_ratio"]:.4f} |
| bigram_entropy | {targets["bigram_entropy"]:.4f} |
| trigram_entropy | {targets["trigram_entropy"]:.4f} |

Source: `{source}`

## Best grid match (H1 fairness)

| Field | Value |
|-------|-------|
| bracket_types | {matched.config.bracket_types} |
| max_depth | {matched.config.max_depth} |
| n_shuffles | {matched.config.n_shuffles} |
| min_len | {matched.config.min_len} |
| max_len | {matched.config.max_len} |
| Euclidean distance (DEFAULT_KEYS) | {matched.distance:.6f} |

Matched corpus stats: `{json.dumps(matched.stats)}`

## Hu et al. replication config (ADR-0025)

Pinned `hu_replication_config()`: k={hu_cfg.bracket_types}, max_depth={hu_cfg.max_depth},
n_shuffles={hu_cfg.n_shuffles}, max_len={hu_cfg.max_len}.

| Metric | Value |
|--------|-------|
| Distance to targets (DEFAULT_KEYS) | {hu_dist:.6f} |
| L1 byte-length hist vs matched corpus | {hu_byte:.6f} |

Hu replication stats: `{json.dumps(hu_stats)}`

## Artifacts

- Machine-readable: `docs/data/dyck-match-result.json`
"""
    REPORT.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT}")
    print(f"Wrote {RESULT_JSON}")
    if pending_u2:
        print(
            "NOTE: pending_u2_fixture=true — fixture stats published; "
            "Dyck targets still use live Saṃsādhanī until TTR parity."
        )


if __name__ == "__main__":
    main()
