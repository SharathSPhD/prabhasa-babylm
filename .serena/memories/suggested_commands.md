# PSALM — suggested commands (Serena memory)

```bash
# Environment
uv sync --extra dev --extra stats            # CPU dev env
uv sync --extra all                          # everything incl. ml/verification/memory

# TECHNICAL closure gate
make gate                                    # ruff + mypy + format-check + tests/coverage
uv run ruff check                            # lint
uv run ruff format                           # format
uv run mypy                                  # strict types
uv run pytest --cov=psalm                    # tests + coverage (gate: 80%)

# CLI
uv run psalm --help
uv run psalm version
uv run psalm config show configs/phase2/arm_B_paninian_en.yaml
uv run psalm contract check <report.json>    # exit 0 closed+signed, 1 closed no-signoff, 2 not closed

# Container (GB10)
make docker                                  # docker build -t psalm:dev .

# Git: one worktree per phase
git worktree add ../PSALM-phase1 -b phase-1-data
```

System: Linux aarch64 (DGX Spark), bash. Use ripgrep (`rg`), not grep.
