"""Live integration tests for the Saṃsādhanī generator + toolkit corpus.

These exercise the real Docker container (default localhost:8090) and the
locally provisioned corpora. They are skipped automatically when the toolkit is
absent, the container is unreachable, or the corpus data is not present — so the
suite stays green in CI without these external dependencies.
"""

from __future__ import annotations

import pytest

from psalm.application.data.ports import AnnotatedSentence
from psalm.infrastructure.corpora.panini_toolkit import PaniniToolkitCorpusSource
from psalm.infrastructure.generators.samsadhani import (
    SamsadhaniiGenerator,
    SamsadhaniNotConfiguredError,
)

pytest.importorskip("panini_data_toolkit")


def _gen_or_skip() -> SamsadhaniiGenerator:
    gen = SamsadhaniiGenerator()
    if not gen.is_configured:
        pytest.skip("Saṃsādhanī container not reachable at localhost:8090")
    return gen


def test_generator_streams_annotated_sentences() -> None:
    gen = _gen_or_skip()
    items = list(gen.stream(5, seed=0))
    assert len(items) == 5
    for s in items:
        assert isinstance(s, AnnotatedSentence)
        assert s.language == "sa"
        assert s.text.strip()
        assert s.has_gold_parse  # gold kāraka roles present
        roles = [role for _tok, role in s.karaka_parse]
        assert "kriyA" in roles  # every sentence has a verb
        assert any(r == "karwA" for r in roles)  # and an agent


def test_generator_is_deterministic_by_seed() -> None:
    gen = _gen_or_skip()
    a = [s.text for s in gen.stream(4, seed=11)]
    b = [s.text for s in gen.stream(4, seed=11)]
    assert a == b


def test_unreachable_container_raises_not_configured() -> None:
    gen = SamsadhaniiGenerator(base_url="http://127.0.0.1:1")  # closed port
    assert gen.is_configured is False
    with pytest.raises(SamsadhaniNotConfiguredError):
        list(gen.stream(1, seed=0))


def test_corpus_source_streams_when_available() -> None:
    src = PaniniToolkitCorpusSource("dcs")
    if not src.is_available():
        pytest.skip("DCS corpus not provisioned (set PANINI_DATA_DIR)")
    n = 0
    for s in src.stream():
        assert isinstance(s, AnnotatedSentence)
        assert s.text.strip()
        n += 1
        if n >= 10:
            break
    assert n == 10
