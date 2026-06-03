# GB10 validation — Vidyut-native realizer (V-gb10, ADR-0033 / ADR-0035)

**Workstream:** `vidyut-realize`
**Date:** 2026-06-03
**Host:** `spark-5208` — NVIDIA **GB10** (DGX Spark), driver `580.159.03`, `aarch64`
**Constraint:** GPU-only / native GB10, no mock (ADR-0035). The realizer itself is a
CPU-side Pāṇinian *derivation* (no GPU kernels); "GPU-only" here means it is
validated **natively on the GB10 box** that runs the training/measurement, on the
same `aarch64` wheel, with no x86 / emulation / mock substitute.

## What was validated

The Vidyut-native frame realizer (`psalm.infrastructure.generators.vidyut_realizer.VidyutFrameRealizer`)
running on the GB10 host against the optional `vidyut==0.4.0` aarch64 wheel.

Command (native, on the GB10):

```bash
uv run --no-sync python infra/dgx_spark/smoke.py --realizer-only --vidyut-min 200
```

Result:

```
=== GB10 ENV INVENTORY ===
uname -m: aarch64
nvidia-smi: NVIDIA GB10, 580.159.03
python: 3.12.3

=== GB10 STACK SMOKE SUMMARY ===
  host_arch                PASS     aarch64
  nvidia_smi               PASS     NVIDIA GB10, 580.159.03
  python                   PASS     3.12.3
  vidyut_realizer          PASS     n=200 gold=100% deriv=100% sample='bAlAH vidyayA pustakam KAdeyuH'
================================
```

- **≥100 realized sentences:** 200 produced (acceptance floor is 100).
- **Gold kāraka parse:** 100% (`has_gold_parse` for every sentence).
- **Derivation traces:** 100% carry a non-empty sūtra-by-sūtra Pāṇinian derivation.
- **Sandhi honesty:** every `meta.sandhi ∈ {full, partial, none}`; partial junctions
  are space-separated padapāṭha, never a fabricated fusion (ADR-0033).

## Fixture corpus

`data/fixtures/paninian_v1.jsonl` — 10,000 real-Sanskrit sentences, all with gold
kāraka parse and derivation traces, exported on this host:

```bash
uv run --no-sync python scripts/export_vidyut_fixtures.py \
    --n 10000 --seed 0 \
    --out data/fixtures/paninian_v1.jsonl \
    --stats-out docs/data/paninian-v1-stats.json
```

- 10,000 lines, 10,000 distinct sentences, `gold_parse_fraction = 1.0`,
  `with_derivation_fraction = 1.0`.
- SHA-256 manifest: `data/fixtures/paninian_v1.sha256`.

### Note on lexical diversity (flagged to `measurement`)

The controlled lexicon (14 stems × 10 dhātus) gives low lexical type–token ratio
(~0.012), which fails the *informational* `validate_diversity_gate` TTR floor
(0.05). This is **intentional** for a minimal-pair structural probe: structural
diversity is maximal (10,000 distinct frames), and the diversity gate is explicitly
"not wired into the training matrix" (ADR-0024). The memorization risk this implies
is handled downstream by **held-out frame/role-sequence splits** in the
`measurement` workstream (ADR-0035), not by inflating the lexicon here.

## Container status

`infra/dgx_spark/Dockerfile.verified` builds the GB10 NGC stack. Host-native
execution is authoritative for the realizer (it is pure-CPU derivation); the
container path is exercised by `infra/dgx_spark/build-validation.sh RUN_DOCKER=1`
and is not required to re-run for realizer acceptance. The realizer check is now
part of `smoke.py` (`check_vidyut_realizer`) so the container smoke covers it when
built.
