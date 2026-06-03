"""Integration: CI Vidyut fixtures → aligned pairs with ≥80% coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psalm.application.data.ports import AnnotatedSentence
from psalm.infrastructure.generators.paribhasha.shabdabodha import (
    ShabdabodhaSuccess,
    compile_shabdabodha,
    measure_coverage,
    to_aligned_record,
    validate_aligned_record,
)

_FIXTURE = Path(__file__).resolve().parents[3] / "data/fixtures/vidyut-fixtures-ci.jsonl"


def _load_fixtures() -> list[AnnotatedSentence]:
    out: list[AnnotatedSentence] = []
    with _FIXTURE.open(encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            out.append(
                AnnotatedSentence(
                    text=obj["text"],
                    language=obj.get("language", "sa"),
                    karaka_parse=tuple(tuple(p) for p in obj["karaka_parse"]),
                    meta=dict(obj.get("meta", {})),
                )
            )
    return out


@pytest.fixture(scope="module")
def ci_sentences() -> list[AnnotatedSentence]:
    assert _FIXTURE.exists(), "run export_vidyut_fixtures.py --ci-sample first"
    return _load_fixtures()


def test_ci_fixture_coverage_at_least_80_percent(ci_sentences: list[AnnotatedSentence]) -> None:
    stats = measure_coverage(ci_sentences)
    assert stats["coverage_fraction"] >= 0.80


def test_every_ci_success_validates_against_schema(ci_sentences: list[AnnotatedSentence]) -> None:
    # Faithful ākāṅkṣā/yogyatā gating (ADR-0034 D3.4) means some frames are *skipped*
    # (e.g. a guṇa stem in kartā position). Every surviving success must still validate.
    n_success = 0
    for s in ci_sentences:
        outcome = compile_shabdabodha(s)
        if not isinstance(outcome, ShabdabodhaSuccess):
            continue
        n_success += 1
        record = to_aligned_record(s, outcome)
        validate_aligned_record(record)
        assert record["paribhasha_string"]
        assert record["shabdabodha_graph"]["nodes"]
    assert n_success >= 1


def test_transitive_visayata_present_in_every_success(
    ci_sentences: list[AnnotatedSentence],
) -> None:
    # ADR-0034 D1 / P-visaya-lossless: the karma object relation survives the
    # linearization for 100% of transitive frames (regression for the dropped-edge bug).
    for s in ci_sentences:
        outcome = compile_shabdabodha(s)
        if not isinstance(outcome, ShabdabodhaSuccess):
            continue
        if any(role == "karma" for _tok, role in s.karaka_parse):
            assert "VISAYATA(" in outcome.rendered.ascii, f"VISAYATA dropped for {s.text!r}"


def test_padartha_assignment_uses_multiple_categories(
    ci_sentences: list[AnnotatedSentence],
) -> None:
    # ADR-0034 D3.1 / P-padartha: padārtha is morphology-derived, not all-DRAVYA.
    categories: set[str] = set()
    for s in ci_sentences:
        outcome = compile_shabdabodha(s)
        if isinstance(outcome, ShabdabodhaSuccess):
            categories |= {n.category.value for n in outcome.graph.nodes}
    assert len(categories) >= 3, categories
