"""Unit tests for crystallization Milestone 1 (CPU, no network)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from psalm.application.data.ports import AnnotatedSentence
from psalm.infrastructure.crystallization.domain_triples import (
    count_candidate_triples,
    enumerate_domain_triples,
)
from psalm.infrastructure.crystallization.entity_lexicon import EntityLexicon
from psalm.infrastructure.crystallization.frames import (
    build_frame_from_triple,
    enumerate_frames_for_triples,
)
from psalm.infrastructure.crystallization.pipeline import (
    CHARTER_MIN_NET_MAPPING_RATE,
    crystallize_frames,
    run_milestone_1,
    write_results,
)
from psalm.infrastructure.crystallization.relation_map import (
    PROPERTY_MAP,
    is_generable,
    karaka_action_rate,
)
from psalm.infrastructure.crystallization.stem_validation import (
    alias_entry_to_validated_stem,
    entry_is_generator_ready,
)


def test_property_map_matches_spike_expectations() -> None:
    assert len(PROPERTY_MAP) == 39
    assert karaka_action_rate() == pytest.approx(22 / 39, rel=0.01)


def test_lexicon_loads_and_lookup() -> None:
    lex = EntityLexicon.load()
    assert len(lex) >= 80
    lion = lex.lookup("Q140")
    assert lion is not None
    assert lion.stem_wx == "siMha"
    assert lion.gender == "puM"


def test_domain_triples_are_entity_entity_and_generable() -> None:
    lex = EntityLexicon.load()
    triples = enumerate_domain_triples(lex)
    assert len(triples) >= 100
    for t in triples:
        assert is_generable(t.property_id)
        assert lex.lookup(t.subject_qid) is not None
        assert lex.lookup(t.object_qid) is not None


def test_net_mapping_rate_meets_charter_in_bounded_domain() -> None:
    lex = EntityLexicon.load()
    triples = enumerate_domain_triples(lex)
    pool = count_candidate_triples(lex)
    assert len(triples) >= 1000
    assert pool >= len(triples)
    rate = len(triples) / pool if pool else 0.0
    assert rate >= CHARTER_MIN_NET_MAPPING_RATE


def test_dense_triples_yield_many_frames() -> None:
    lex = EntityLexicon.load()
    triples = enumerate_domain_triples(lex)
    frames = enumerate_frames_for_triples(triples, lex, max_frames=150_000)
    assert len(frames) >= 100_000


def test_frame_from_triple_has_valid_structure() -> None:
    lex = EntityLexicon.load()
    triples = enumerate_domain_triples(lex)
    frame = build_frame_from_triple(triples[0], lex)
    assert frame is not None
    words = frame.structure["words"]
    assert any(w["pos"] == "verb" for w in words)
    assert any(w["karaka"] == "karwA" for w in words if w["pos"] == "noun")


def test_frame_enumeration_is_deterministic_and_large() -> None:
    lex = EntityLexicon.load()
    triples = enumerate_domain_triples(lex)
    a = enumerate_frames_for_triples(triples, lex, max_frames=500)
    b = enumerate_frames_for_triples(triples, lex, max_frames=500)
    assert len(a) == 500
    assert [f.signature for f in a] == [f.signature for f in b]


def test_run_milestone_1_skip_generation_mapping_only() -> None:
    result = run_milestone_1(skip_generation=True, max_frames=200)
    assert result.lexicon_entries >= 80
    assert result.net_mapping_rate >= CHARTER_MIN_NET_MAPPING_RATE
    assert result.frame_count == 200
    assert result.unique_sentences == 0
    assert any(v.target == "net_mapping_rate" and v.passed for v in result.charter_verdicts)


def test_stem_validation_alias_fallback() -> None:
    lex = EntityLexicon.load()
    lion = lex.lookup("Q140")
    assert lion is not None
    validated = frozenset({"rAma", "nara"})
    aliased = alias_entry_to_validated_stem(lion, validated)
    assert aliased is not None
    assert aliased.stem_wx in validated
    assert entry_is_generator_ready(aliased, validated=validated)


def test_write_results_artifacts(tmp_path: Path) -> None:
    result = run_milestone_1(skip_generation=True, max_frames=10)
    md = tmp_path / "m1.md"
    js = tmp_path / "m1.json"
    write_results(result, md_path=md, json_path=js)
    assert md.read_text(encoding="utf-8").startswith("# Crystallization Milestone 1")
    assert '"milestone": "crystallization_m1"' in js.read_text(encoding="utf-8")


def test_crystallize_frames_when_generator_offline() -> None:
    lex = EntityLexicon.load()
    frames = enumerate_frames_for_triples(enumerate_domain_triples(lex), lex, max_frames=5)
    with patch("psalm.infrastructure.crystallization.pipeline.SamsadhaniiGenerator") as mock_cls:
        gen = MagicMock()
        gen.is_configured = False
        mock_cls.return_value = gen
        sentences, configured, mode = crystallize_frames(frames, target_unique=3)
    assert sentences == []
    assert configured is False
    assert mode == "offline"


def test_run_milestone_1_with_mocked_generator() -> None:
    fake = AnnotatedSentence(
        text="सिंहः गङ्गायाम् वसति",
        language="sa",
        karaka_parse=(("सिंहः", "karwA"),),
        meta={"source": "test"},
    )

    with patch("psalm.infrastructure.crystallization.pipeline.SamsadhaniiGenerator") as mock_cls:
        gen = MagicMock()
        gen.is_configured = True
        gen._annotate.return_value = fake
        mock_cls.return_value = gen
        result = run_milestone_1(
            target_unique=5,
            max_frames=20,
            seed=0,
        )
    assert result.generator_configured is True
    assert result.unique_sentences >= 1
