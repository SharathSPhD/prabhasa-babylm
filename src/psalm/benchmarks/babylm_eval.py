"""BabyLM 2025 official evaluation pipeline adapter.

Wraps ``babylm/evaluation-pipeline-2025`` when installed via
``scripts/setup_babylm_eval_pipeline.sh`` (pinned commit, not vendored in-repo).
Uses a local BLiMP/EWoK-style minimal-pair PLL harness (``ElcPsalmEvaluator`` or any
``PseudoLogLikelihoodModel``) for smoke and CI; ``invoke_official_zero_shot`` targets
HF-exported checkpoints on the full suite.

See ``docs/decisions/0020-babylm-dual-track.md`` and ``docs/data/eval-train-status.md``.
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

# Pinned in scripts/setup_babylm_eval_pipeline.sh — keep in sync.
BABYLM_EVAL_PIPELINE_COMMIT = "bf55c1131e53654c2a87418f5629c26959acd710"
BABYLM_EVAL_PIPELINE_REPO = "https://github.com/babylm/evaluation-pipeline-2025.git"
DEFAULT_EVAL_ROOT = Path("vendor/babylm-evaluation-pipeline-2025")


class RunMode(StrEnum):
    """Eval data path: built-in minimal pairs vs official pipeline layout detected."""

    LOCAL = "local"
    OFFICIAL = "official"
    MOCK = "mock"


@runtime_checkable
class PseudoLogLikelihoodModel(Protocol):
    """BabyLM-submittable models must score strings (PLL or causal log-prob)."""

    def pseudo_log_likelihood(self, text: str) -> float: ...


@dataclass(frozen=True)
class SmokeTask:
    """A tiny minimal-pair probe (BLiMP / EWoK style)."""

    name: str
    good: str
    bad: str


@dataclass
class SmokeEvalResult:
    mode: RunMode
    task_scores: dict[str, float]
    aggregate_score: float
    report_path: Path | None = None
    pipeline_root: Path | None = None
    notes: str = ""
    pipeline_installed: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "task_scores": self.task_scores,
            "aggregate_score": self.aggregate_score,
            "report_path": str(self.report_path) if self.report_path else None,
            "pipeline_root": str(self.pipeline_root) if self.pipeline_root else None,
            "pipeline_installed": self.pipeline_installed,
            "notes": self.notes,
            "pipeline_commit": BABYLM_EVAL_PIPELINE_COMMIT,
        }


class MockUniformBaseline:
    """Random log-uniform scores — wiring-only baseline when ``--mock`` is set."""

    def __init__(self, vocab_size: int = 1000, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._vocab_size = max(vocab_size, 2)

    def pseudo_log_likelihood(self, text: str) -> float:
        total = 0.0
        for _ in text.split():
            p = self._rng.random() / self._vocab_size
            total += math.log(max(p, 1e-12))
        return total


def smoke_tasks() -> tuple[SmokeTask, ...]:
    """Built-in minimal pairs for smoke / CI (not the full BLiMP release)."""
    return (
        SmokeTask("blimp_agreement", "the cat sleeps", "the cat sleep"),
        SmokeTask("blimp_determiner", "a big dog runs", "a big dogs runs"),
        SmokeTask("ewok_simple", "birds have wings", "birds have engines"),
        SmokeTask("blimp_island", "who did john see", "who did john sees"),
    )


def detect_pipeline_install(root: Path | None = None) -> bool:
    """True when the official pipeline package directory is present."""
    base = root or eval_root()
    pipeline_pkg = base / "evaluation_pipeline"
    return pipeline_pkg.is_dir() and (pipeline_pkg / "sentence_zero_shot").is_dir()


def eval_root() -> Path:
    env = os.environ.get("BABYLM_EVAL_ROOT")
    if env:
        return Path(env)
    return DEFAULT_EVAL_ROOT


def minimal_pair_accuracy(model: PseudoLogLikelihoodModel, task: SmokeTask) -> float:
    """Zero-shot accuracy: 1.0 iff PLL(good) > PLL(bad)."""
    good = model.pseudo_log_likelihood(task.good)
    bad = model.pseudo_log_likelihood(task.bad)
    return 1.0 if good > bad else 0.0


def run_pll_minimal_pair_eval(
    model: PseudoLogLikelihoodModel,
    tasks: tuple[SmokeTask, ...],
) -> dict[str, float]:
    """Score each minimal pair with real pseudo-log-likelihoods."""
    return {t.name: minimal_pair_accuracy(model, t) for t in tasks}


def run_smoke_eval(
    model: PseudoLogLikelihoodModel,
    *,
    mode: RunMode | None = None,
    tasks: tuple[SmokeTask, ...] | None = None,
    output_path: Path | None = None,
    pipeline_root: Path | None = None,
    force_mock_baseline: bool = False,
) -> SmokeEvalResult:
    """Run fast zero-shot smoke eval with real PLL minimal-pair scoring."""
    root = pipeline_root or eval_root()
    installed = detect_pipeline_install(root)
    resolved_mode = mode
    if resolved_mode is None:
        resolved_mode = RunMode.OFFICIAL if installed else RunMode.LOCAL

    task_list = tasks or smoke_tasks()
    scores = run_pll_minimal_pair_eval(model, task_list)

    if force_mock_baseline or resolved_mode is RunMode.MOCK:
        notes = (
            "MOCK baseline model: scores are not from ELC-PSALM. "
            "Use --elc (default) for real PLL minimal-pair accuracy."
        )
        resolved_mode = RunMode.MOCK
    elif resolved_mode is RunMode.OFFICIAL:
        notes = (
            "Real PLL minimal-pair accuracy on built-in smoke tasks. "
            "Official full-suite zero-shot: export HF checkpoint and call "
            "invoke_official_zero_shot (see docs/data/eval-train-status.md)."
        )
    else:
        notes = (
            "Real PLL minimal-pair accuracy (local harness). "
            "Install official pipeline via scripts/setup_babylm_eval_pipeline.sh "
            "for invoke_official_zero_shot on BLiMP/EWoK releases."
        )

    aggregate = sum(scores.values()) / len(scores) if scores else 0.0
    result = SmokeEvalResult(
        mode=resolved_mode,
        task_scores=scores,
        aggregate_score=aggregate,
        pipeline_root=root if installed else None,
        pipeline_installed=installed,
        notes=notes,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        result.report_path = output_path
    return result


def invoke_official_zero_shot(
    *,
    model_path: str,
    task: str = "blimp",
    backend: str = "mlm",
    data_path: Path | None = None,
    output_dir: Path | None = None,
    pipeline_root: Path | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one official zero-shot task (requires HF model dir + psalm[ml] in pipeline env).

    Raises FileNotFoundError if the pipeline is not installed.
    """
    root = pipeline_root or eval_root()
    if not detect_pipeline_install(root):
        raise FileNotFoundError(f"BabyLM pipeline missing at {root}")

    data = data_path or (root / "evaluation_pipeline" / "sentence_zero_shot" / "data")
    out = output_dir or (root / "results" / "psalm_run")
    cmd = [
        "python",
        "-m",
        "evaluation_pipeline.sentence_zero_shot.run",
        "--data_path",
        str(data),
        "--task",
        task,
        "--model_path_or_name",
        model_path,
        "--backend",
        backend,
        "--output_dir",
        str(out),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(
        cmd,
        cwd=str(root),
        check=False,
        capture_output=True,
        text=True,
    )
