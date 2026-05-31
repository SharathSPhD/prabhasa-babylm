# GB10 / aarch64 stack de-risk note (Phase 1)

Status date: 2026-05-31. Host: DGX Spark, GB10 Grace-Blackwell, `aarch64`.

## What this note de-risks

Phase 1's contract requires confirming that the data-engine stack builds on the
Spark's `aarch64` architecture before we commit to it for the training phases.
This is an honest status snapshot, not a claim that the full training stack is
built.

## Confirmed working (light data stack, base env, no container)

The Phase 1 data engine runs in the plain `uv` environment with no GPU and no
container:

- `aarch64` architecture confirmed (`uname -m`).
- `sentencepiece==0.2.1` installs from wheel and trains a tokenizer end-to-end
  (`scripts/build_tokenizer.py` round-trips, preserving the `·` morpheme marker).
- `huggingface-hub==1.17.0` installs and imports; network reachability to
  `huggingface.co` confirmed (HTTP 200).
- Full technical gate green here: ruff + mypy clean, 114 tests, ~97% coverage.

Note: installing `huggingface-hub` pinned `typer` to 0.25.1; the CLI tests still
pass, but this constraint is recorded so a future `typer` bump is intentional.

## Deferred to the NGC container (ADR-0007), not yet built here

The heavy training stack is intentionally **not** installed in the base env. It
belongs in the NGC PyTorch Blackwell/`aarch64` image used for Phases 2+:

- `torch` (Blackwell/CUDA build) — not importable in base env, by design.
- `flash-attn`, `unsloth` — build status to be verified inside the NGC image; the
  Pramana repo's NGC + Unsloth setup is the reference (per ADR-0007).
- `datasets` — listed in the `data` extra but not yet installed; the HF corpus
  source therefore raises `SourceNotProvisionedError` (verified by test), which
  is the correct fail-closed behaviour.

## Open de-risk item carried into the Phase 1 gate

The one thing the data engine cannot self-provision is the **Saṃsādhanī Pāṇinian
generator** and the license-clean Sanskrit corpora (DCS/GRETIL/HF). Until these
are provisioned, the Phase 1 *empirical* go/no-go (`generator_diversity_sufficient`)
cannot be computed on real data — the engine is built and tested, but the finding
is blocked on external provisioning. This is flagged explicitly at the Phase 1
closure gate rather than papered over.
