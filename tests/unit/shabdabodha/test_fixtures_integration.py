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
    for s in ci_sentences:
        outcome = compile_shabdabodha(s)
        assert isinstance(outcome, ShabdabodhaSuccess), f"expected success for {s.text!r}"
        record = to_aligned_record(s, outcome)
        validate_aligned_record(record)
        assert record["paribhasha_string"]
        assert record["shabdabodha_graph"]["nodes"]
