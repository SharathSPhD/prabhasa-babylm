# 8. Citation integrity and the excluded fabricated reference

Date: 2026-05-31
Status: Accepted

## Context

The paper requires a rigorous, honest literature review with no fabricated
citations and no plagiarism. During background research, a plausible-looking
reference — "Cubical Type Theoretic Navya-Nyāya", arXiv:2605.12548 — was
identified as **fabricated** (it does not resolve to a real work).

## Decision

Every citation in the paper must resolve to a verifiable real work (DOI, arXiv
id, ACL Anthology id, or stable URL checked at cite time). The fabricated
arXiv:2605.12548 reference is **permanently excluded** and must never be cited.
A citation-integrity rule (`.cursor/rules/`) and an automated check enforce this;
the paper's bibliography is verified before each dissemination closure.

## Consequences

Slower citing (each reference is verified) but a defensible, plagiarism-free
manuscript. Any future "too good to be true" citation is treated as suspect until
verified.
