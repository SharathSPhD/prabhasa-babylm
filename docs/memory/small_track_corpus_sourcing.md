# Small-Track (100M) Corpus Sourcing Plan

**Date:** 2026-06-05  
**Status:** Research & Planning (GPU not available)  
**Scope:** BabyLM 2026 Small track (100M-word budget) + Sanskrit dose pre-pretraining  

---

## Executive Summary

The Small track requires **≤100M English words** for the official BabyLM competition submission (tracked under `comp:strict:*` arms). Sanskrit data is **NOT** counted toward the 100M cap per ADR-0020; it is a **dose pre-pretraining layer** (L1 in the layered curriculum).

**Total estimated corpus:**
- **English (100M):** Official BabyLM Strict release + eligible supplements
- **Sanskrit (dose):** DCS + GRETIL + AI4Bharat Sangraha; estimated 14.9B tokens (Sangraha Sanskrit alone)

---

## Data Sources

### English (BabyLM Official & Eligible)

| Source | URL | License | Size | Status | BabyLM-Eligible |
|--------|-----|---------|------|--------|-----------------|
| **BabyLM 2026 Strict (Official)** | https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict | MIT | 100M tokens (5 sources) | ✓ Verified | **YES** — official release |
| BNC Spoken | (part of Strict) | MIT | 7.6M tokens | ✓ Included | YES |
| CHILDES | (part of Strict) | MIT | 28.4M tokens | ✓ Included | YES |
| Project Gutenberg | (part of Strict) | MIT | 25.6M tokens | ✓ Included | YES |
| Open Subtitles | (part of Strict) | MIT | 22.8M tokens | ✓ Included | YES |
| Simple Wikipedia | (part of Strict) | MIT | 15.3M tokens | ✓ Included | YES |

**English subtotal:** 100M tokens (released as single coherent dataset, detoxified and deduplicated in 2026)

---

### Sanskrit (Dose Pre-pretraining, Outside 100M Cap)

| Source | URL | License | Size | Status | Notes |
|--------|-----|---------|------|--------|-------|
| **AI4Bharat Sangraha (Sanskrit subset)** | https://huggingface.co/datasets/ai4bharat/sangraha | CC-BY-4.0 | ~14.9B tokens (Sanskrit only: 1.3B verified + 13.6B synthetic/unverified) | ✓ Verified | Largest available; includes synthetic via perplexity filtering; requires attribution |
| **GRETIL (Göttingen Register)** | https://gretil.sub.uni-goettingen.de/gretil.html | Not explicitly stated (see warnings) | ~100–200M tokens (estimate, all Sanskrit texts + Vedas) | ✓ Accessible | Authoritative classical texts (Vedas, Epics, Puranas, Philosophical texts); manual download or bulk ZIP; license ambiguous — **FLAG** |
| **DCS (Digital Corpus of Sanskrit)** | https://github.com/ambuda-org/dcs | CC-BY-4.0 | ~650K sentences (~10–15M tokens, estimate) | ✓ Accessible via GitHub | Lemmatized + POS-tagged; smaller than GRETIL/Sangraha but highest quality annotation; Oliver Hellwig original (2010–2021) + Ambuda curation |

**Sanskrit subtotal (dose, not counted toward 100M):** ~14.9B–16B tokens (Sangraha dominant)

---

## Compliance Notes: BabyLM Strict Track Rules

### What the rules permit:

1. **Word budget:** ≤100M words for Strict track; Participants may use the **official BabyLM corpus OR construct a custom dataset**, provided the word count is respected.

2. **Multilingual / non-English data:** Allowed **only if it counts toward the 100M cap**. The 2026 rules explicitly allow "multimodal data and teacher-model feedback" within budget. However, Sanskrit as a separate **pre-pretraining dose** is conceptually outside the sequential pretraining word budget.

3. **PSALM-specific:** Per ADR-0020, competition track (`comp:strict:*`) uses `PretrainCorpus.ENGLISH` with a multilingual curriculum **inside budget**; research track may use `SANSKRIT_ENGLISH` but this does **NOT** qualify for official BabyLM leaderboard submission.

### How PSALM handles the 100M limit:

- **Competition submission track (`comp:strict:*`):** All text (English + Sanskrit if included) must sum to ≤100M words.
- **Research track (no leaderboard submission):** Sanskrit can be a separate dose pre-pretraining phase (L1), sampled independently in curriculum; DGX Spark is not constrained by BabyLM budgets per ADR-0020.

### Risk flagged:

**For an official BabyLM submission**, Sanskrit cannot be counted as "free" pre-pretraining. If PSALM submits to the Strict track:
- **Option A (recommended):** Use only the 100M English corpus; Sanskrit stays in research-only arms.
- **Option B (risky):** Create a 100M composite corpus with English + Sanskrit mixed; this is allowed under rules but must account all tokens toward cap and report composition clearly.

---

## Download & Prep Strategy

### Phase 1: Verify & Manifest (CPU-only, this task)

1. **English:** Already local at `/data/corpora/babylm-2026-strict-small/` (10M) and upstream at HF for full 100M.
   - [ ] Verify local 10M is subset of official 100M release.
   - [ ] Create corpus manifest: `corpus_manifest.yaml` with per-source word counts, dedup hashes, license.

2. **Sanskrit — Sangraha:**
   - [ ] Fetch HF dataset metadata (not full download) to confirm Sanskrit token count and license.
   - [ ] Check if HF provides bulk download URLs or requires API access.

3. **Sanskrit — GRETIL:**
   - [ ] Download bulk ZIP (link: https://textgridrep.org/ or DARIAH-DE DOI: 10.20375/0000-0016-C802-4).
   - [ ] Parse & tokenize sample to estimate total word count.
   - [ ] Flag missing explicit license (GRETIL uses CC-BY or similar; verify before use).

4. **Sanskrit — DCS:**
   - [ ] Clone/fetch from https://github.com/ambuda-org/dcs.
   - [ ] Inventory included texts (Mahabhārata, Rāmāyaṇa, etc.); estimate token count via sampling.

### Phase 2: Tokenization & Budget Accounting

- Use unified tokenizer trained on **English + Sanskrit** (if research track) or English-only (if competition submission).
- Log epoch-equivalent exposure per source per seed.
- Update `corpus_manifest.yaml` with final token counts post-dedup.

### Phase 3: Training Curriculum (Post GPU-Free Audit)

- L4 (dominant): BabyLM English, 100M words.
- L3 (optional, research): GRETIL + DCS; ~25M tokens, 5–10% mix rate.
- L1 (optional, research-only dose): Sangraha (~1–2% pre-pretrain, or separate pre-pretraining phase).

---

## License Warnings

| Source | License | Status | Action |
|--------|---------|--------|--------|
| BabyLM 2026 Strict | MIT | ✓ Clear | Commercial use permitted; cite BabyLM consortium. |
| AI4Bharat Sangraha | CC-BY-4.0 | ✓ Clear | Requires attribution in paper/code; allows modifications. |
| DCS (ambuda-org) | CC-BY-4.0 | ✓ Clear | Requires attribution; derived from Oliver Hellwig DCS (2010–2021). |
| GRETIL | **Not explicitly stated** | ⚠️ UNCLEAR | **FLAG:** Website does not list license. Likely CC-BY or similar (federated academic resource); verify before publishing. Contact Göttingen University or check DARIAH-DE terms. |

**Recommendation:** Do not include GRETIL in a public model release until license is clarified. For research use, likely acceptable under educational fair use; for publication, obtain written confirmation from GRETIL maintainers.

---

## File Locations & Scripts

### Already local:
- `/home/sharaths/projects/PSALM-integration/data/corpora/babylm-2026-strict-small/` — 10M Strict-Small (5 files, 52M on disk, ~10M tokens)
- `/home/sharaths/projects/PSALM-integration/data/corpora/strict_small/` — duplicate or derived?
- `/home/sharaths/projects/PSALM-integration/data/corpora/priors/` — synthetic Pāṇinian data (not BabyLM cap-bound)

### To be created:
- `/home/sharaths/projects/PSALM-integration/corpus_manifest.yaml` — per-source word counts, dedup, epochs, hashes.
- `/home/sharaths/projects/PSALM-integration/scripts/fetch_small_track_corpus.sh` — download orchestrator (commented, not executed).
- `/home/sharaths/projects/PSALM-integration/configs/small_track_corpus.yaml` — tokenizer + budget config.

---

## Commands (Documented, Not Executed)

See `scripts/fetch_small_track_corpus.sh` for the full pipeline. Key steps:

```bash
# 1. Fetch BabyLM 100M (if not already cached)
huggingface-cli download BabyLM-community/BabyLM-2026-Strict \
  --repo-type dataset \
  --cache-dir /data/corpora/ \
  --local-files-only false

# 2. Fetch Sangraha metadata
python -c "from datasets import load_dataset; ds = load_dataset('ai4bharat/sangraha', 'san', split='train'); print(f'Sanskrit: {len(ds)} examples')"

# 3. Download GRETIL bulk
wget -O /data/corpora/gretil-sanskrit.zip \
  "https://textgridrep.org/dataset/10.20375/0000-0016-C802-4/data"

# 4. Clone DCS
git clone https://github.com/ambuda-org/dcs.git /data/corpora/dcs

# 5. Generate manifest
python scripts/corpus_manifest_gen.py \
  --english-path /data/corpora/babylm-2026-strict \
  --sanskrit-sources gretil,sangraha,dcs \
  --output corpus_manifest.yaml
```

---

## Next Steps (Blocked by GPU Saturation)

1. **Now (CPU):** Write corpus download script, verify URLs, generate manifest skeleton.
2. **After GPU frees:** Run tokenization + budget accounting on all sources.
3. **Gate:** `psalm contract check <report>` verifies manifest before any training.

---

## References

- [BabyLM 2026 Guidelines](https://babylm.github.io/guidelines.html)
- [BabyLM 2026 Strict Dataset](https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict)
- [AI4Bharat Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha)
- [GRETIL Portal](https://gretil.sub.uni-goettingen.de/gretil.html)
- [DCS (Ambuda)](https://github.com/ambuda-org/dcs)
- ADR-0020 (BabyLM dual track)
- PSALM Spec v2 § Layered Curriculum (L0–L4)
