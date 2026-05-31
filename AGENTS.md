# AGENTS.md

This repository's agent operating guide lives in [`CLAUDE.md`](CLAUDE.md). It
applies to every coding agent (Cursor, Claude Code, Codex, or otherwise),
regardless of platform.

Read `CLAUDE.md` before acting. In particular:

- The **Ralph-loop closure contract** is binding — a phase is not done on green
  CI alone. See `src/psalm/domain/contracts/closure.py` and
  `docs/contracts/closure-contract.md`.
- **TDD, config-driven, hexagonal** architecture; 80% coverage gate.
- **Statistical honesty** and **citation integrity** are non-negotiable;
  the fabricated arXiv:2605.12548 reference must never be cited.
- Single **DGX Spark GB10** budget; `uv` + NGC container.
