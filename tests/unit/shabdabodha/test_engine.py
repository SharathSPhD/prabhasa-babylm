"""Unit tests for Vyutpattivāda rule engine (hand-built AnnotatedSentence inputs)."""

from __future__ import annotations

import pytest

from psalm.application.data.ports import AnnotatedSentence
from psalm.infrastructure.generators.paribhasha.relations import validate_graph
from psalm.infrastructure.generators.paribhasha.shabdabodha import (
    RULE_KARMA_VISAYATA,
    RULE_KARW_SAMYOGATA,
    ShabdabodhaSkip,
    ShabdabodhaSuccess,
    compile_shabdabodha,
    to_aligned_record,
)
from psalm.infrastructure.generators.paribhasha.types import SansaType


def _sentence(
    *,
    text: str,
    parse: list[tuple[str, str]],
    signature: str,
) -> AnnotatedSentence:
    return AnnotatedSentence(
        text=text,
        karaka_parse=tuple(parse),
        meta={"frame_signature": signature, "fixture_id": "test-1"},
    )


class TestCompileShabdabodha:
    def test_transitive_karwA_karma(self) -> None:
        s = _sentence(
            text="rAma.eka.karwA Pala.eka.karma paW1.viXiH",
            parse=[("rAma", "karwA"), ("Pala", "karma")],
            signature="paW1|viXiH|rAma|eka|Pala|eka",
        )
        out = compile_shabdabodha(s)
        assert isinstance(out, ShabdabodhaSuccess)
        assert RULE_KARW_SAMYOGATA in out.rules_applied
        assert RULE_KARMA_VISAYATA in out.rules_applied
        validate_graph(out.graph)
        sansa = {e.sansa for e in out.graph.edges}
        assert SansaType.SAMYOGATA in sansa
        assert SansaType.VISAYATA in sansa
        assert out.rendered.ascii

    def test_intransitive_with_adhikaraNa(self) -> None:
        s = _sentence(
            text="guru.eka.karwA aSva.eka.aXikaraNam vas1.varwamAnaH",
            parse=[("guru", "karwA"), ("aSva", "aXikaraNam")],
            signature="vas1|varwamAnaH|guru|eka|-|aXikaraNam|aSva",
        )
        out = compile_shabdabodha(s)
        assert isinstance(out, ShabdabodhaSuccess)
        sansa = {e.sansa for e in out.graph.edges}
        assert SansaType.SAMYOGATA in sansa
        assert SansaType.PRAKARATA in sansa

    def test_oblique_transitive_karaNam(self) -> None:
        s = _sentence(
            text="bAla.bahu.karwA puswaka.eka.karma vixyA.eka.karaNam KAx1.viXiH",
            parse=[
                ("bAla", "karwA"),
                ("puswaka", "karma"),
                ("vixyA", "karaNam"),
            ],
            signature="KAx1|viXiH|bAla|bahu|puswaka|karaNam|vixyA",
        )
        out = compile_shabdabodha(s)
        assert isinstance(out, ShabdabodhaSuccess)
        assert out.rule_coverage == 1.0

    def test_skip_empty_parse(self) -> None:
        s = AnnotatedSentence(text="noop", karaka_parse=(), meta={})
        assert isinstance(compile_shabdabodha(s), ShabdabodhaSkip)

    def test_aligned_record_meta(self) -> None:
        s = _sentence(
            text="nara.eka.karwA gfha.eka.karma paW1.viXiH",
            parse=[("nara", "karwA"), ("gfha", "karma")],
            signature="paW1|viXiH|nara|eka|gfha|eka",
        )
        out = compile_shabdabodha(s)
        assert isinstance(out, ShabdabodhaSuccess)
        rec = to_aligned_record(s, out)
        assert rec["meta"]["schema_version"] == "paribhasha_aligned_v1"
        assert rec["meta"]["source"] == "shabdabodha-vyutpattivada"
        assert "paribhasha_iast" in rec["meta"]

    @pytest.mark.parametrize("bad_role", ["hetu", "samA"])
    def test_skip_unknown_karaka(self, bad_role: str) -> None:
        s = _sentence(
            text="x",
            parse=[("a", "karwA"), ("b", bad_role)],
            signature="gam1|viXiH|a|eka",
        )
        assert isinstance(compile_shabdabodha(s), ShabdabodhaSkip)
