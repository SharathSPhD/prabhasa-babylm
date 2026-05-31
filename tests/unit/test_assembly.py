"""Tests for pre-pretraining corpus assembly across H1 arms."""

from __future__ import annotations

import pytest

from psalm.application.data.assembly import (
    PrePretrainAssembler,
    aux_targets,
    serialize_line,
)
from psalm.application.data.ports import AnnotatedSentence
from psalm.domain.experiments.models import PrePretrainSource


class _FakeGen:
    """Deterministic fake SentenceGenerator yielding annotated forms."""

    def __init__(self, prefix: str, with_derivation: bool) -> None:
        self.prefix = prefix
        self.with_derivation = with_derivation

    def stream(self, n: int, *, seed: int = 0):
        for i in range(n):
            yield AnnotatedSentence(
                text=f"{self.prefix}{seed}_{i}",
                language="sa" if self.with_derivation else "dyck",
                derivation=("1.3.1", "8.4.68") if self.with_derivation else (),
            )


def _assembler() -> PrePretrainAssembler:
    return PrePretrainAssembler(
        paninian=_FakeGen("pan", with_derivation=True),
        dyck=_FakeGen("(()) ", with_derivation=False),
    )


def test_none_source_yields_no_items() -> None:
    a = _assembler()
    assert list(a.items(PrePretrainSource.NONE, 5, seed=0)) == []
    assert list(a.lines(PrePretrainSource.NONE, 5, seed=0)) == []


def test_paninian_streams_requested_count() -> None:
    a = _assembler()
    items = list(a.items(PrePretrainSource.PANINIAN, 4, seed=1))
    assert len(items) == 4
    assert all(it.derivation for it in items)


def test_dyck_streams_unannotated() -> None:
    a = _assembler()
    items = list(a.items(PrePretrainSource.DYCK, 3, seed=0))
    assert len(items) == 3
    assert all(not it.derivation for it in items)


def test_karaka_aux_uses_same_input_stream_as_paninian() -> None:
    # Arm D must share arm B's *input* and differ only by the auxiliary target,
    # otherwise B vs D is confounded.
    a = _assembler()
    b_lines = list(a.lines(PrePretrainSource.PANINIAN, 4, seed=2))
    d_lines = list(a.lines(PrePretrainSource.PANINIAN_KARAKA_AUX, 4, seed=2))
    assert b_lines == d_lines


def test_serialize_line_paninian_is_surface_form() -> None:
    s = AnnotatedSentence(text="Bavati", derivation=("1.3.1", "8.4.68"))
    assert serialize_line(s, PrePretrainSource.PANINIAN) == "Bavati"


def test_serialize_line_dyck_is_text() -> None:
    s = AnnotatedSentence(text="( ( ) )", language="dyck")
    assert serialize_line(s, PrePretrainSource.DYCK) == "( ( ) )"


def test_aux_targets_present_only_for_karaka_arm() -> None:
    s = AnnotatedSentence(text="Bavati", derivation=("1.3.1", "8.4.68"))
    assert aux_targets(s, PrePretrainSource.PANINIAN_KARAKA_AUX) == ("1.3.1", "8.4.68")
    # Non-aux arms carry no auxiliary target.
    assert aux_targets(s, PrePretrainSource.PANINIAN) == ()
    assert aux_targets(s, PrePretrainSource.DYCK) == ()


def test_aux_targets_prefer_gold_karaka_roles() -> None:
    # ADR-0012: arm D's auxiliary target is the gold kāraka role sequence from
    # the Saṃsādhanī generator, preferred over the derivation when present.
    s = AnnotatedSentence(
        text="रामः वनम् गच्छति",
        karaka_parse=(("रामः", "karwA"), ("वनम्", "karma"), ("गच्छति", "kriyA")),
        derivation=("1.3.1",),
    )
    assert aux_targets(s, PrePretrainSource.PANINIAN_KARAKA_AUX) == (
        "karwA",
        "karma",
        "kriyA",
    )


def test_lines_require_configured_generator() -> None:
    bare = PrePretrainAssembler(paninian=None, dyck=None)
    with pytest.raises(RuntimeError, match="no Pāṇinian generator"):
        list(bare.items(PrePretrainSource.PANINIAN, 1, seed=0))
    with pytest.raises(RuntimeError, match="no Dyck generator"):
        list(bare.items(PrePretrainSource.DYCK, 1, seed=0))


def test_take_until_tokens_respects_budget() -> None:
    a = _assembler()
    # 1 token per line under whitespace counting; budget 3 -> 3 lines.
    lines = list(a.take_until_tokens(PrePretrainSource.PANINIAN, budget_tokens=3, seed=0))
    assert len(lines) == 3
