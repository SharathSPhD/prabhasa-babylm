"""JSON Schema conformance for ``paribhasha_aligned_v1`` exports."""

from __future__ import annotations

import json

import jsonschema
import pytest

from psalm.application.data.ports import AnnotatedSentence
from psalm.infrastructure.generators.paribhasha.shabdabodha import (
    ShabdabodhaSuccess,
    aligned_pair_schema_path,
    compile_shabdabodha,
    to_aligned_record,
    validate_aligned_record,
)

_SCHEMA = json.loads(aligned_pair_schema_path().read_text(encoding="utf-8"))
_VALIDATOR = jsonschema.Draft202012Validator(_SCHEMA)


@pytest.fixture
def sample_record() -> dict[str, object]:
    s = AnnotatedSentence(
        text="rAma.eka.karwA Pala.eka.karma paW1.viXiH",
        karaka_parse=(("rAma", "karwA"), ("Pala", "karma")),
        meta={"frame_signature": "paW1|viXiH|rAma|eka|Pala|eka", "fixture_id": "schema-1"},
    )
    out = compile_shabdabodha(s)
    assert isinstance(out, ShabdabodhaSuccess)
    return to_aligned_record(s, out)


def test_validate_aligned_record_helper(sample_record: dict[str, object]) -> None:
    validate_aligned_record(sample_record)  # noqa: ARG001


def test_required_top_level_keys(sample_record: dict[str, object]) -> None:
    for key in ("text", "karaka_parse", "shabdabodha_graph", "paribhasha_string", "meta"):
        assert key in sample_record


def test_meta_schema_version_const(sample_record: dict[str, object]) -> None:
    meta = sample_record["meta"]
    assert isinstance(meta, dict)
    assert meta.get("schema_version") == "paribhasha_aligned_v1"


def test_graph_nodes_edges_schema(sample_record: dict[str, object]) -> None:
    graph = sample_record["shabdabodha_graph"]
    assert isinstance(graph, dict)
    _VALIDATOR.validate(sample_record)
