# ADR-0036 — From-scratch BabyLM Strict-Small build: budget-matched arms + dual venue

- Status: accepted
- Date: 2026-06-03
- Supersedes the stale pre-paribhāṣā H1′ measurement (see `p3-info-parity-finding.md`)
- Governs: `corpus-ss`, `tokenizer`, `model-arms`, `train-ss`, `eval-official`, `measure-h1`

## Context

The prior H1′ nulls (60M/100M/150M) were an **under-training floor** — ~180k tokens
on 60–150M-param models left arm A at chance, so no structural-prior effect could
show. No GPU curriculum training has been run on the *new* data engine (Vidyut
realization + faithful Vyutpattivāda Paribhāṣā + Dyck control). We therefore rebuild
the experiment **from scratch** as a BabyLM Strict-Small (10M words) and later Strict
(100M words) build, per `docs/psalm-consolidation-report.md` §4, using the real 10M
budget that the BabyLM baselines show is sufficient to learn the venue (~0.70 BLiMP).

## Decision

### Budget (binding: all tokens count toward 10M)

- **Shared 9M English base** for every arm (BabyLM-2026-Strict-Small: childes,
  gutenberg, open_subtitles, simple_wiki, bnc_spoken, switchboard).
- **1M-token pre-pretrain dose (10%)**, type varies by arm.
- Total **10M / arm**, budget-matched and competition-legal.

### Arm matrix (Strict-Small competition arms)

| Arm | Pre-pretrain dose (1M) | Pretrain (9M) | Role |
|---|---|---|---|
| A | 1M **English** (control) | 9M English | baseline (prior slot = more English) |
| B | 1M **Paninian** (Vidyut realized) | 9M English | H1: grammar prior |
| C | 1M **Dyck** (k-shuffle control) | 9M English | H1 control (Hu et al.) |
| D | 1M **Paribhāṣā** (Vyutpattivāda) | 9M English | Paribhāṣā vs Dyck |

The control for the prior is **equivalent in-domain English** (arm A), so B/C/D differ
from each other *only* in prior **type** at identical total budget and identical 9M
English base. Decisive contrasts: **B vs C** (grammar vs bracket) and **D vs C**
(semantic graph vs bracket). Token-savings (H1) measured against arm A.

Real Sanskrit (IndicCorp v2 `sa.txt`, DCS CoNLL-U) is reserved for research-track
arms **E/F** (Sanskrit competence + cross-lingual), not these competition arms.

### Venues (both, per sign-off)

1. **Official BabyLM** by PLL via the vendored 2025 pipeline: **BLiMP** (+ supplement)
   and **EWoK** — the competition reading and the report's Paribhāṣā→EWoK claim.
2. **Internal COGS/SCAN** compositional generalization — the H1 "compositional
   accuracy / token savings" reading.

### Scale ladder

- **Proxy 60M @ 10M words** first (≈25 min/run) as the **floor-lift probe**: confirm
  arm A lands in **[0.55, 0.75]** (info-parity C5) before the main battery.
- Then **100–150M main battery**, ≥3 seeds/arm, paired bootstrap + Holm–Bonferroni.

## Consequences

- The info-parity C5 gate is re-measured on this curriculum; GATE holds until arm A
  is shown off-floor in-band.
- GPU-only on GB10; no mocks; manifest with per-source word counts + dedup hashes;
  train/eval leakage check fail-closed (the harness already enforces these).
- If arm A still floors at 9M English on the proxy, escalate model size / dose before
  the battery (documented, not silently worked around).
