# 1. Record architecture decisions

Date: 2026-05-31
Status: Accepted

## Context

PSALM is a long, complex, multi-phase research+product program. Decisions made
early (scope, thresholds, hardware, model identity) constrain everything later,
and the closure contract requires that pre-registered thresholds only change
through a documented decision.

## Decision

We record significant decisions as Architecture Decision Records (ADRs) in
`docs/decisions/`, numbered sequentially, in the Nygard format (Context,
Decision, Consequences). Changing a pre-registered go/no-go threshold, a hardware
constraint, or the experimental design requires a new ADR that supersedes the
relevant prior one.

## Consequences

The decision history is auditable and survives across phases and agents. ADRs
are part of the MEMORY layer of the closure contract. The overhead is one short
markdown file per significant decision.
