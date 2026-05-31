# PSALM — architecture (Serena memory)

Hexagonal, config-driven, TDD. Dependencies point inward only.

```
src/psalm/
  domain/          PURE logic (no torch/IO)
    contracts/closure.py        -> Ralph-loop closure contract (binding)
    experiments/models.py       -> ExperimentArm, RunResult, MetricResult, Finding
    rewards/  validators/       -> H2/H3 domain logic (to build)
  application/     use cases (orchestration)
    data/  training/  evaluation/
  infrastructure/  external systems
    generators/   -> Saṃsādhanī wrapper, Dyck control (Phase 1)
    tokenizer/    -> sandhi-aware SentencePiece (Phase 1)
    ml/           -> torch/transformers/TRL/PEFT adapters (mypy-excluded)
    storage/knowledge_store.py  -> SQLite/vector knowledge store
    ledger/sqlite_ledger.py     -> experiment ledger
    verification/ -> Z3 vyāpti verifier (H3)
  config/          pydantic-settings + YAML loader (config_hash for repro)
  cli/main.py      typer app: version, config show, contract check
  analysis/comparison_tests.py  bootstrap CI, permutation test, Holm-Bonferroni
  benchmarks/      eval-suite runners (Vyāpti Probe, compositional)
```

Tests in tests/{unit,integration,e2e}. TECHNICAL gate: ruff + mypy(strict) +
pytest, coverage ≥ 80%. ML infra adapters are mypy-excluded (wrap untyped libs)
and coverage-omitted (run on GPU).

Key entrypoints already implemented and tested: closure contract, experiment
models, ledger, config loader+hash, statistical comparison tests, knowledge
store, CLI.
