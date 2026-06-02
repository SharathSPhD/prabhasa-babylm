# BabyLM joint tokenizer fertility (ASCII vs IAST Paribhāṣā)

Pre-registered in **ADR-0027**. Competition default: **ASCII** Paribhāṣā strings.

## Setup

- Train joint unigram/BPE on **manifest-eligible text only** (within 10M / 100M word caps).
- Vocab targets: **20k** (Strict-Small), **28k** (Strict) — see `JointTokenizerPlan`.
- Reproduce smoke numbers:

```bash
uv sync --extra data
uv run python scripts/build_babylm_joint_tokenizer.py --track strict_small
```

## Manifest schema (`babylm_manifest_v1`)

| Field | Type | Description |
|---|---|---|
| `schema_version` | string | Always `babylm_manifest_v1` |
| `track` | `strict_small` \| `strict` | 10M / 100M word budget |
| `sources[]` | list | Per-source accounting |
| `sources[].name` | string | Stable source id |
| `sources[].word_count` | int | Whitespace word count (pretokenized) |
| `sources[].dedup_hash` | string | 64-char SHA-256 hex of normalized shard |
| `sources[].epochs` | float | Passes over unique words |
| `sources[].license_spdx` | string | SPDX id (see `licenses.py`) |
| `sources[].stage` | `pre_pretrain` \| `pretrain` | Budget stage |

Derived: `total_words`, `epoch_equivalent_words = Σ(word_count × epochs)`.

Check: `uv run psalm eval manifest check configs/data/babylm-corpus-manifest-strict_small.yaml`

## Smoke fertility (2026-06-02, demo joint corpus)

Joint tokenizer: unigram, requested vocab 20k, realised **89** (tiny demo corpus — not submission).

| Script | tokens/word | n_words (ablation sample) |
|---|---:|---:|
| Paribhāṣā ASCII | **2.5225** | 178 |
| Paribhāṣā IAST | **4.6170** | 178 |
| Δ (IAST − ASCII) | **+2.0946** | — |

Ablation parameters: `n_lines=80`, `seed=13` (`run_paribhasha_fertility_ablation`).

**Interpretation (smoke):** Δ ≫ 0.05 → keep **ASCII** for competition tokenizer training per ADR-0027.
Re-measure on frozen manifest shards before submission; small demo corpora inflate IAST piece fragmentation.

## pyproject.toml tokenizer deps

| Extra | Packages |
|---|---|
| `data` | `sentencepiece>=0.2`, `huggingface-hub`, `datasets`, `pyarrow` |
| `ml` | `tokenizers>=0.19`, `sentencepiece>=0.2`, `transformers`, … |

Core `dependencies` do not include SentencePiece; install `uv sync --extra data` (or `all`) for training.
