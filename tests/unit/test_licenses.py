"""Tests for license modelling and source-manifest cleanliness checks."""

from __future__ import annotations

import pytest

from psalm.domain.data.licenses import (
    Redistribution,
    SourceManifest,
    assembly_is_clean,
    license_for,
    summarize_assembly,
)


def test_known_license_lookup() -> None:
    cc_by = license_for("CC-BY-4.0")
    assert cc_by.is_redistributable
    assert cc_by.requires_attribution


def test_cc0_is_redistributable_without_attribution() -> None:
    cc0 = license_for("CC0-1.0")
    assert cc0.is_redistributable
    assert not cc0.requires_attribution


def test_unknown_license_defaults_to_forbidden() -> None:
    lic = license_for("some-weird-license")
    assert lic.redistribution is Redistribution.FORBIDDEN
    assert not lic.is_redistributable


def test_research_only_is_not_redistributable() -> None:
    assert not license_for("research-only").is_redistributable


def test_source_manifest_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        SourceManifest("bad", license_for("PD"), n_docs=-1)


def test_assembly_clean_when_all_clean() -> None:
    manifests = [
        SourceManifest("gretil", license_for("CC-BY-SA-4.0"), n_docs=10, n_tokens=1000),
        SourceManifest("dcs", license_for("CC-BY-SA-3.0"), n_docs=5, n_tokens=500),
    ]
    assert assembly_is_clean(manifests)


def test_assembly_blocked_by_forbidden_source() -> None:
    manifests = [
        SourceManifest("ok", license_for("CC0-1.0"), n_docs=1, n_tokens=100),
        SourceManifest("bad", license_for("unknown"), n_docs=1, n_tokens=100),
    ]
    assert not assembly_is_clean(manifests)
    summary = summarize_assembly(manifests)
    assert summary["all_clean"] is False
    assert summary["blocked"] == ["bad"]
    assert summary["total_tokens"] == 200
    assert summary["n_sources"] == 2
