"""Unit tests for Vyutpattivāda rule engine (hand-built AnnotatedSentence inputs)."""

from __future__ import annotations

import pytest

from psalm.application.data.ports import AnnotatedSentence
from psalm.infrastructure.generators.paribhasha.relations import validate_graph
from psalm.infrastructure.generators.paribhasha.shabdabodha import (
    RULE_KARMA_VISAYATA,
    RULE_KARW_SAMYOGATA,
    RULE_SANKHYA_VISESANA,
    SKIP_AKANKSA_AKARMAKA_KARMA,
    SKIP_AKANKSA_NO_KARTA,
    SKIP_YOGYATA_KARTA,
    ShabdabodhaSkip,
    ShabdabodhaSuccess,
    compile_shabdabodha,
    to_aligned_record,
)
from psalm.infrastructure.generators.paribhasha.types import PadarthaCategory, SansaType


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


class TestFaithfulSemantics:
    def test_sankhya_visesana_encodes_number(self) -> None:
        # The number (bahu) is absent from karaka_parse but must surface as a guṇa
        # viśeṣaṇa (prakāratā) on the kartā — the information beyond the kāraka parse.
        s = _sentence(
            text="bAla.bahu.karwA Pala.eka.karma KAx1.viXiH",
            parse=[("bAla", "karwA"), ("Pala", "karma")],
            signature="KAx1|viXiH|bAla|bahu|Pala|eka",
        )
        out = compile_shabdabodha(s)
        assert isinstance(out, ShabdabodhaSuccess)
        assert RULE_SANKHYA_VISESANA in out.rules_applied
        sankhya = [n for n in out.graph.nodes if n.label.startswith("saMKyA")]
        assert {n.label for n in sankhya} == {"saMKyA_bahu", "saMKyA_eka"}
        assert all(n.category is PadarthaCategory.GUNA for n in sankhya)

    def test_guna_instrument_qualifies_karta_by_prakarata(self) -> None:
        # "acts by knowledge": vidyā (guṇa) cannot contact the action, so it qualifies
        # the kartā via prakāratā rather than being forced to DRAVYA.
        s = _sentence(
            text="nara.eka.karwA gfha.eka.karma vixyA.eka.karaNam KAx1.viXiH",
            parse=[("nara", "karwA"), ("gfha", "karma"), ("vixyA", "karaNam")],
            signature="KAx1|viXiH|nara|eka|gfha|karaNam|vixyA",
        )
        out = compile_shabdabodha(s)
        assert isinstance(out, ShabdabodhaSuccess)
        guna_fillers = [
            n
            for n in out.graph.nodes
            if n.category is PadarthaCategory.GUNA and not n.label.startswith("saMKyA")
        ]
        assert any(n.label == "vixyA" for n in guna_fillers)

    def test_yogyata_skips_guna_karta(self) -> None:
        s = _sentence(
            text="vixyA.eka.karwA gfha.eka.karma KAx1.viXiH",
            parse=[("vixyA", "karwA"), ("gfha", "karma")],
            signature="KAx1|viXiH|vixyA|eka|gfha|eka",
        )
        out = compile_shabdabodha(s)
        assert isinstance(out, ShabdabodhaSkip)
        assert out.rule_id == SKIP_YOGYATA_KARTA

    def test_akanksa_skips_akarmaka_with_karma(self) -> None:
        s = _sentence(
            text="nara.eka.karwA gfha.eka.karma vas1.varwamAnaH",
            parse=[("nara", "karwA"), ("gfha", "karma")],
            signature="vas1|varwamAnaH|nara|eka|gfha|eka",
        )
        out = compile_shabdabodha(s)
        assert isinstance(out, ShabdabodhaSkip)
        assert out.rule_id == SKIP_AKANKSA_AKARMAKA_KARMA

    def test_akanksa_skips_missing_karta(self) -> None:
        s = _sentence(
            text="Pala.eka.karma KAx1.viXiH",
            parse=[("Pala", "karma")],
            signature="KAx1|viXiH|-|eka|Pala|eka",
        )
        out = compile_shabdabodha(s)
        assert isinstance(out, ShabdabodhaSkip)
        assert out.rule_id == SKIP_AKANKSA_NO_KARTA

    def test_obliques_have_distinct_qualified_shapes(self) -> None:
        s = _sentence(
            text="nara.eka.karwA gfha.eka.apAxAnam vana.eka.aXikaraNam gam1.viXiH",
            parse=[("nara", "karwA"), ("gfha", "apAxAnam"), ("vana", "aXikaraNam")],
            signature="gam1|viXiH|nara|eka|-|apAxAnam|gfha|aXikaraNam|vana",
        )
        out = compile_shabdabodha(s)
        assert isinstance(out, ShabdabodhaSuccess)
        quals = {e.qualifier for e in out.graph.edges if e.sansa is SansaType.SAMYOGATA}
        assert {"apAxAna", "aXikaraNa"} <= quals
