# Plugins & MCP servers

PSALM uses two custom Cursor plugins for reasoning support during the program.
Both are upstream Claude Code plugins; they are installed as **Cursor local
plugins** with a `.cursor-plugin/plugin.json` adapter and their MCP servers are
registered in `~/.cursor/mcp.json`.

## triz-engine — contradiction resolution

TRIZ 40 Inventive Principles, the 39x39 contradiction matrix, ARIZ-85C, and an
IFR (ideal final result) workflow. Used whenever a research or engineering
**contradiction** appears (e.g. "more grammar coverage" vs "tokenizer stays
small"; "constrained decoding" vs "fluency"). The MCP server exposes the
principle/matrix knowledge base over stdio via FastMCP.

- Local install: `~/.cursor/plugins/local/triz-engine/`
- MCP server: `uv run --script .../servers/triz_server.py` (PEP-723 inline deps)

## attractor-flow — divergent→convergent ideation & orchestration steering

Monitors multi-agent trajectories with finite-time Lyapunov exponents over
semantic embeddings and classifies the regime (CONVERGING / CYCLING / EXPLORING
/ DIVERGING / STUCK / OSCILLATING / PLATEAU) to prescribe interventions before a
phase derails. Used to steer the divergent ideation that opens each phase toward
convergence, and to detect when the Ralph-loop intervention loop is itself
cycling without progress.

- Local install: `~/.cursor/plugins/local/attractor-flow/`
- MCP server: `uv run --script .../attractorflow/mcp-server/server.py`

## How they map onto the program

| Situation | Tool |
|---|---|
| A design decision pits two desirable properties against each other | triz-engine (contradiction matrix / IFR) |
| Opening a phase with many candidate approaches | attractor-flow explorer → convergence agents |
| An intervention loop keeps missing the threshold without converging | attractor-flow regime classification (detect STUCK / CYCLING) |
| Choosing among matched experimental arms | triz-engine IFR to sharpen the decisive variable |

## Reload note

Newly registered MCP servers and local plugins load after a Cursor reload. Both
servers were smoke-tested at install time (deps resolve via `uv`; clean stdio
startup).
