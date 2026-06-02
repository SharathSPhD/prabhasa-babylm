"""Unit tests for Saṃsādhanī adapter hardening (dedup, fail-closed, metrics)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from psalm.application.data.ports import AnnotatedSentence
from psalm.infrastructure.generators.samsadhani import (
    SamsadhaniCapacityError,
    SamsadhaniiGenerator,
    samsadhani_diversity_metrics,
)


def _fake_sentence(text: str = "रामः पठति") -> AnnotatedSentence:
    return AnnotatedSentence(
        text=text,
        language="sa",
        karaka_parse=(("रामः", "karwA"), ("पठति", "kriyA")),
        meta={"source": "samsadhani-generator"},
    )


def test_dedup_skips_duplicate_surfaces() -> None:
    gen = SamsadhaniiGenerator(base_url="http://127.0.0.1:1", fail_closed=False)
    calls = 0

    def _annotate(_frame: object) -> AnnotatedSentence | None:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _fake_sentence()
        if calls == 2:
            return _fake_sentence()
        if calls == 3:
            return _fake_sentence("अन्य")
        return None

    with (
        patch.object(gen, "_ensure_client", return_value=MagicMock()),
        patch.object(gen, "_annotate", side_effect=_annotate),
    ):
        items = list(gen.stream(2, seed=0))
    assert len(items) == 2
    assert len({s.text for s in items}) == 2


def test_fail_closed_raises_capacity_error() -> None:
    gen = SamsadhaniiGenerator(base_url="http://127.0.0.1:1", fail_closed=True)
    with (
        patch.object(gen, "_ensure_client", return_value=MagicMock()),
        patch.object(gen, "_annotate", return_value=None),
        pytest.raises(SamsadhaniCapacityError),
    ):
        list(gen.stream(2, seed=0))


def test_stream_skips_items_without_gold_parse() -> None:
    gen = SamsadhaniiGenerator(base_url="http://127.0.0.1:1", fail_closed=False)
    no_gold = AnnotatedSentence(text="रामः", language="sa", karaka_parse=())
    with (
        patch.object(gen, "_ensure_client", return_value=MagicMock()),
        patch.object(gen, "_annotate", side_effect=[no_gold, _fake_sentence()]),
    ):
        items = list(gen.stream(1, seed=0))
    assert len(items) == 1
    assert items[0].has_gold_parse


def test_diversity_metrics_keys() -> None:
    items = [_fake_sentence(f"वाक्य {i}") for i in range(5)]
    report = samsadhani_diversity_metrics(items)
    assert report["n_sentences"] == 5.0
    assert report["pct_with_gold_parse"] == 100.0
    assert "bigram_entropy" in report
