"""Tests for the RQ-A (F2) per-paradigm analyzer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "analyze_rqA", Path(__file__).resolve().parents[2] / "scripts" / "analyze_rqA.py"
)
assert _spec and _spec.loader
analyze_rqA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze_rqA)


def _write_log(tmp_path: Path, name: str, lines: dict[str, float]) -> Path:
    body = "\n".join(f"{k}: {v}" for k, v in lines.items())
    p = tmp_path / name
    p.write_text(body + "\n### AVERAGE ACCURACY\n70.0\n", encoding="utf-8")
    return p


def test_extract_per_paradigm(tmp_path: Path) -> None:
    p = _write_log(
        tmp_path,
        "k.log",
        {
            "determiner_noun_agreement_2": 99.46,
            "regular_plural_subject_verb_agreement_1": 89.55,
            "argument_structure_x": 60.0,
            "irregular_forms_1": 80.0,
        },
    )
    d = analyze_rqA.extract_per_paradigm(p)
    assert d["determiner_noun_agreement_2"] == 99.46
    assert len(d) == 4
    sub = analyze_rqA.targeted_subset(d)
    assert "irregular_forms_1" not in sub  # not agreement/arg
    assert set(sub) == {
        "determiner_noun_agreement_2",
        "regular_plural_subject_verb_agreement_1",
        "argument_structure_x",
    }


def test_compare_paired_bootstrap(tmp_path: Path) -> None:
    common = {
        "subject_verb_agreement_1": None,
        "determiner_noun_agreement_1": None,
        "argument_structure_a": None,
        "anaphor_agreement_1": None,
    }
    k = _write_log(tmp_path, "k.log", dict.fromkeys(common, 70.0))
    c = _write_log(tmp_path, "c.log", dict.fromkeys(common, 68.0))  # K beats C by +2 everywhere
    r = analyze_rqA.compare(k, c)
    assert r["n_targeted_paradigms"] == 4
    assert abs(r["mean_diff_K_minus_C"] - 2.0) < 1e-6
    assert r["significant"] is True  # constant +2 → CI excludes 0
