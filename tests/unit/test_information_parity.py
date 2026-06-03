"""Tests for the information-parity preflight audit (ADR-0035 D6, M-info-parity)."""

from __future__ import annotations

from pathlib import Path

from psalm.analysis.information_parity import (
    InfoParityInputs,
    audit_information_parity,
    scan_mock_cpu_paths,
)


def _passing_inputs(**overrides: object) -> InfoParityInputs:
    base: dict[str, object] = {
        "h_structure_given_karaka": 3.18,
        "visayata_transitive_fraction": 1.0,
        "template_count": 499,
        "acceptance_fraction": 0.81,
        "acceptance_target": 0.80,
        "venue_early_accuracy": 0.65,
        "mock_cpu_path_count": 0,
    }
    base.update(overrides)
    return InfoParityInputs(**base)  # type: ignore[arg-type]


class TestAudit:
    def test_all_six_pass(self) -> None:
        report = audit_information_parity(_passing_inputs())
        assert report.passed is True
        assert len(report.checks) == 6
        assert {c.id for c in report.checks} == {
            "C1_H_structure_given_karaka",
            "C2_visayata_transitive",
            "C3_template_count",
            "C4_acceptance_ge_target",
            "C5_venue_off_saturation",
            "C6_zero_mock_cpu_paths",
        }

    def test_relabel_fails_c1(self) -> None:
        report = audit_information_parity(_passing_inputs(h_structure_given_karaka=0.2))
        assert report.passed is False
        c1 = next(c for c in report.checks if c.id == "C1_H_structure_given_karaka")
        assert c1.passed is False

    def test_partial_visayata_fails_c2(self) -> None:
        report = audit_information_parity(_passing_inputs(visayata_transitive_fraction=0.97))
        assert report.passed is False

    def test_degenerate_templates_fail_c3(self) -> None:
        report = audit_information_parity(_passing_inputs(template_count=2))
        assert report.passed is False

    def test_below_acceptance_target_fails_c4(self) -> None:
        report = audit_information_parity(
            _passing_inputs(acceptance_fraction=0.70, acceptance_target=0.80)
        )
        assert report.passed is False

    def test_saturated_venue_fails_c5(self) -> None:
        report = audit_information_parity(_passing_inputs(venue_early_accuracy=0.95))
        assert report.passed is False

    def test_too_hard_venue_fails_c5(self) -> None:
        report = audit_information_parity(_passing_inputs(venue_early_accuracy=0.40))
        assert report.passed is False

    def test_mock_path_present_fails_c6(self) -> None:
        report = audit_information_parity(_passing_inputs(mock_cpu_path_count=1))
        assert report.passed is False


class TestScanMockCpuPaths:
    def test_clean_source_has_no_hits(self, tmp_path: Path) -> None:
        f = tmp_path / "clean.py"
        f.write_text("def f():\n    return resolve_training_device('cuda')\n", encoding="utf-8")
        assert scan_mock_cpu_paths([f]) == []

    def test_detects_mock_baseline(self, tmp_path: Path) -> None:
        f = tmp_path / "dirty.py"
        f.write_text("x = MockUniformBaseline(vocab_size=10)\n", encoding="utf-8")
        hits = scan_mock_cpu_paths([f])
        assert len(hits) == 1
        assert "MockUniformBaseline" in hits[0]

    def test_detects_runmode_mock_and_cpu_fallback(self, tmp_path: Path) -> None:
        f = tmp_path / "dirty.py"
        f.write_text("mode = RunMode.MOCK\n# silently falling back to CPU here\n", encoding="utf-8")
        hits = scan_mock_cpu_paths([f])
        names = " ".join(hits)
        assert "RunMode.MOCK" in names
        assert "silent-cpu-fallback" in names

    def test_missing_file_skipped(self, tmp_path: Path) -> None:
        assert scan_mock_cpu_paths([tmp_path / "nope.py"]) == []
