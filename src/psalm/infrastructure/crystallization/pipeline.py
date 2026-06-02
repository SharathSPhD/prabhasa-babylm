"""Milestone 1 crystallization pipeline — bounded domain yield measurement."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from psalm.application.data.ports import AnnotatedSentence
from psalm.domain.data.karaka_frames import KarakaFrame
from psalm.infrastructure.crystallization.domain_triples import (
    enumerate_domain_triples,
    triple_mapping_stats,
)
from psalm.infrastructure.crystallization.entity_lexicon import EntityLexicon
from psalm.infrastructure.crystallization.frames import enumerate_frames_for_triples
from psalm.infrastructure.generators.samsadhani import (
    SamsadhaniiGenerator,
)

# Charter falsifiable targets (crystallization-track-charter-2026-06-01.md §5).
CHARTER_MIN_UNIQUE_SENTENCES = 100_000
CHARTER_MIN_NET_MAPPING_RATE = 0.25
CHARTER_TOKEN_BASELINE = 179_876
CHARTER_TOKEN_EXPANSION_FACTOR = 10
CHARTER_MIN_GRAMMATICALITY_RATE = 0.80
CHARTER_GRAMMAR_SAMPLE_SIZE = 100

# Frame coverage patterns (arm-D enrichment beyond binary kartā/karma).
FRAME_COVERAGE_PATTERNS: tuple[str, ...] = (
    "karwA",
    "karwA karma",
    "karwA aXikaraNam",
    "karwA karaNam",
    "karwA apAxAnam",
    "karwA sampraxAnam",
)


@dataclass
class CharterVerdict:
    target: str
    required: str
    measured: str
    passed: bool


@dataclass
class CrystallizationM1Result:
    domain: str
    lexicon_entries: int
    mapping_stats: dict[str, int | float]
    net_mapping_rate: float
    frame_count: int
    unique_sentences: int
    no_repeat_tokens: int
    frame_coverage: dict[str, float]
    grammaticality_rate: float | None
    grammaticality_sample_size: int
    generator_configured: bool
    generator_mode: str
    charter_verdicts: list[CharterVerdict] = field(default_factory=list)
    overall_verdict: str = "PENDING"
    limitation_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "milestone": "crystallization_m1",
            "domain": self.domain,
            "lexicon_entries": self.lexicon_entries,
            "mapping_stats": self.mapping_stats,
            "net_mapping_rate": round(self.net_mapping_rate, 4),
            "frame_count": self.frame_count,
            "unique_sentences": self.unique_sentences,
            "no_repeat_tokens": self.no_repeat_tokens,
            "frame_coverage": self.frame_coverage,
            "grammaticality_rate": self.grammaticality_rate,
            "grammaticality_sample_size": self.grammaticality_sample_size,
            "generator_configured": self.generator_configured,
            "generator_mode": self.generator_mode,
            "charter_targets": {
                "min_unique_sentences": CHARTER_MIN_UNIQUE_SENTENCES,
                "min_net_mapping_rate": CHARTER_MIN_NET_MAPPING_RATE,
                "token_baseline": CHARTER_TOKEN_BASELINE,
                "token_expansion_factor": CHARTER_TOKEN_EXPANSION_FACTOR,
                "min_grammaticality_rate": CHARTER_MIN_GRAMMATICALITY_RATE,
            },
            "charter_verdicts": [
                {
                    "target": v.target,
                    "required": v.required,
                    "measured": v.measured,
                    "passed": v.passed,
                }
                for v in self.charter_verdicts
            ],
            "overall_verdict": self.overall_verdict,
            "limitation_notes": self.limitation_notes,
        }


def _token_count(sentences: Sequence[str]) -> int:
    return sum(len(s.split()) for s in sentences)


def _frame_role_sequence(frame: KarakaFrame) -> str:
    words = frame.structure["words"]
    if not isinstance(words, list):
        return ""
    roles = [str(w["karaka"]) for w in words if w.get("pos") == "noun"]
    return " ".join(roles)


def _pattern_coverage(role_seqs: Sequence[str]) -> dict[str, float]:
    n = len(role_seqs)
    if n == 0:
        return dict.fromkeys(FRAME_COVERAGE_PATTERNS, 0.0)
    return {pat: sum(1 for seq in role_seqs if pat in seq) / n for pat in FRAME_COVERAGE_PATTERNS}


def _annotate_frame(gen: SamsadhaniiGenerator, frame: KarakaFrame) -> AnnotatedSentence | None:
    return gen._annotate(frame)  # noqa: SLF001 — M1 harness; adapter tested separately


def crystallize_frames(
    frames: list[KarakaFrame],
    *,
    target_unique: int,
    seed: int = 0,
    base_url: str | None = None,
) -> tuple[list[AnnotatedSentence], bool, str]:
    """Realise frames via Saṃsādhanī until ``target_unique`` distinct surfaces or exhaust."""
    gen = SamsadhaniiGenerator(
        base_url=base_url,
        fail_closed=False,
        dedup=True,
    )
    if not gen.is_configured:
        return [], False, "offline"

    rng = random.Random(seed)
    order = list(frames)
    rng.shuffle(order)

    seen: set[str] = set()
    out: list[AnnotatedSentence] = []
    for frame in order:
        if len(out) >= target_unique:
            break
        sentence = _annotate_frame(gen, frame)
        if sentence is None or not sentence.has_gold_parse:
            continue
        if sentence.text in seen:
            continue
        seen.add(sentence.text)
        out.append(sentence)
    return out, True, "samsadhani"


def grammaticality_probe(
    frames: list[KarakaFrame],
    *,
    sample_size: int = CHARTER_GRAMMAR_SAMPLE_SIZE,
    seed: int = 0,
    base_url: str | None = None,
) -> tuple[float | None, int]:
    """Fraction of sampled frames the generator accepts (proxy for grammaticality)."""
    gen = SamsadhaniiGenerator(base_url=base_url, fail_closed=False)
    if not gen.is_configured:
        return None, 0

    rng = random.Random(seed)
    sample = list(frames)
    rng.shuffle(sample)
    sample = sample[:sample_size]
    if not sample:
        return None, 0

    ok = 0
    for frame in sample:
        if _annotate_frame(gen, frame) is not None:
            ok += 1
    return ok / len(sample), len(sample)


def run_milestone_1(
    *,
    lexicon: EntityLexicon | None = None,
    target_unique: int = CHARTER_MIN_UNIQUE_SENTENCES,
    max_frames: int | None = None,
    seed: int = 0,
    base_url: str | None = None,
    skip_generation: bool = False,
) -> CrystallizationM1Result:
    """Execute bounded-domain M1 measurement against charter targets."""
    lex = lexicon or EntityLexicon.load()
    triples = enumerate_domain_triples(lex)
    stats = triple_mapping_stats(lex, triples)
    net_rate = float(stats.get("net_mapping_rate", 0.0))

    frames = enumerate_frames_for_triples(triples, lex, max_frames=max_frames)
    limitation_notes: list[str] = []

    sentences: list[AnnotatedSentence] = []
    configured = False
    mode = "frames_only"
    if not skip_generation:
        sentences, configured, mode = crystallize_frames(
            frames,
            target_unique=target_unique,
            seed=seed,
            base_url=base_url,
        )
        if not configured:
            limitation_notes.append(
                "Saṃsādhanī offline: unique-sentence and grammaticality targets "
                "cannot be measured on surfaces; frame signatures counted only."
            )
    else:
        limitation_notes.append("Generation skipped (--skip-generation).")

    unique_count = len({s.text for s in sentences}) if sentences else 0
    texts = [s.text for s in sentences]
    tokens = _token_count(texts)

    role_seqs = [_frame_role_sequence(f) for f in frames[: min(len(frames), 50_000)]]
    coverage = _pattern_coverage(role_seqs)

    gram_rate, gram_n = (None, 0)
    if not skip_generation and configured:
        gen_frames = enumerate_frames_for_triples(
            triples, lex, max_frames=CHARTER_GRAMMAR_SAMPLE_SIZE * 2, require_generator_ready=True
        )
        gram_rate, gram_n = grammaticality_probe(gen_frames, seed=seed + 1, base_url=base_url)

    verdicts: list[CharterVerdict] = [
        CharterVerdict(
            target="unique_crystallized_sentences",
            required=f">= {CHARTER_MIN_UNIQUE_SENTENCES:,}",
            measured=f"{unique_count:,}",
            passed=unique_count >= CHARTER_MIN_UNIQUE_SENTENCES,
        ),
        CharterVerdict(
            target="net_mapping_rate",
            required=f">= {CHARTER_MIN_NET_MAPPING_RATE:.0%}",
            measured=f"{net_rate:.1%}",
            passed=net_rate >= CHARTER_MIN_NET_MAPPING_RATE,
        ),
        CharterVerdict(
            target="no_repeat_token_ceiling",
            required=f">= {CHARTER_TOKEN_EXPANSION_FACTOR}× {CHARTER_TOKEN_BASELINE:,}",
            measured=f"{tokens:,}",
            passed=tokens >= CHARTER_TOKEN_BASELINE * CHARTER_TOKEN_EXPANSION_FACTOR,
        ),
        CharterVerdict(
            target="spot_grammaticality",
            required=f">= {CHARTER_MIN_GRAMMATICALITY_RATE:.0%} on {CHARTER_GRAMMAR_SAMPLE_SIZE}",
            measured=(
                f"{gram_rate:.1%} (n={gram_n})"
                if gram_rate is not None
                else "skipped (generator offline)"
            ),
            passed=gram_rate is not None and gram_rate >= CHARTER_MIN_GRAMMATICALITY_RATE,
        ),
    ]

    all_pass = all(v.passed for v in verdicts)
    any_pass = any(v.passed for v in verdicts)
    if all_pass:
        overall = "VOCAB-LAYER-PROVEN"
    elif any_pass:
        overall = "PARTIAL — see charter_verdicts"
    else:
        overall = "STOP — constraint bound; do not expand domain"

    return CrystallizationM1Result(
        domain="natural_kinds_physical_action",
        lexicon_entries=len(lex),
        mapping_stats=stats,
        net_mapping_rate=net_rate,
        frame_count=len(frames),
        unique_sentences=unique_count,
        no_repeat_tokens=tokens,
        frame_coverage=coverage,
        grammaticality_rate=gram_rate,
        grammaticality_sample_size=gram_n,
        generator_configured=configured,
        generator_mode=mode,
        charter_verdicts=verdicts,
        overall_verdict=overall,
        limitation_notes=limitation_notes,
    )


def write_results(
    result: CrystallizationM1Result,
    *,
    md_path: Path,
    json_path: Path,
) -> None:
    """Persist human-readable and machine-readable M1 artifacts."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    lines = [
        "# Crystallization Milestone 1 — bounded-domain results",
        "",
        f"**Domain:** {result.domain}",
        f"**Overall verdict:** {result.overall_verdict}",
        "",
        "## Mapping (Attack-1 vocabulary layer)",
        "",
        f"- Lexicon entries: **{result.lexicon_entries}**",
        f"- Candidate entity–entity triples: **{result.mapping_stats.get('candidate_entity_entity_triples')}**",
        f"- Both endpoints mapped: **{result.mapping_stats.get('both_endpoints_mapped')}**",
        f"- **Net mapping rate:** {result.net_mapping_rate:.1%}",
        "",
        "## Crystallization yield",
        "",
        f"- Distinct kāraka frames enumerated: **{result.frame_count:,}** "
        f"(charter sentence target: {CHARTER_MIN_UNIQUE_SENTENCES:,})",
        f"- Unique annotated sentences (Saṃsādhanī): **{result.unique_sentences:,}**",
        f"- No-repeat token count: **{result.no_repeat_tokens:,}**",
        f"- Generator: {result.generator_mode} (configured={result.generator_configured})",
        "",
        "## Charter targets (PASS/FAIL)",
        "",
        "| Target | Required | Measured | PASS |",
        "|--------|----------|----------|------|",
    ]
    for v in result.charter_verdicts:
        mark = "PASS" if v.passed else "FAIL"
        lines.append(f"| {v.target} | {v.required} | {v.measured} | {mark} |")

    lines.extend(
        [
            "",
            "## Frame coverage (kāraka role patterns)",
            "",
        ]
    )
    for pat, frac in sorted(result.frame_coverage.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{pat}`: {frac:.1%}")

    if result.grammaticality_rate is not None:
        lines.append(
            f"\n**Grammaticality probe:** {result.grammaticality_rate:.1%} "
            f"(n={result.grammaticality_sample_size})"
        )

    if result.limitation_notes:
        lines.append("\n## Limitations\n")
        for note in result.limitation_notes:
            lines.append(f"- {note}")

    lines.append(f"\nMachine-readable: `{json_path}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
