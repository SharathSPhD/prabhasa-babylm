"""BabyLM 2025 official evaluation pipeline adapter.

Wraps ``babylm/evaluation-pipeline-2025`` when installed via
``scripts/setup_babylm_eval_pipeline.sh`` (pinned commit, not vendored in-repo).
Falls back to a MOCK minimal-pair harness for CI and wiring smoke tests.

Live fetch TODO: run the setup script on a machine with network + ``psalm[ml]``;
set ``BABYLM_EVAL_ROOT`` to the clone root. See ``docs/decisions/0020-babylm-dual-track.md``.
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
    LIVE = "live"
    MOCK = "mock"


@runtime_checkable
class PseudoLogLikelihoodModel(Protocol):
    """BabyLM-submittable models must score strings (PLL or causal log-prob)."""

    def pseudo_log_likelihood(self, text: str) -> float: ...


@dataclass(frozen=True)
class SmokeTask:
    """A tiny minimal-pair probe used by the mock harness."""

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

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "task_scores": self.task_scores,
            "aggregate_score": self.aggregate_score,
            "report_path": str(self.report_path) if self.report_path else None,
            "pipeline_root": str(self.pipeline_root) if self.pipeline_root else None,
            "notes": self.notes,
            "pipeline_commit": BABYLM_EVAL_PIPELINE_COMMIT,
        }


class MockUniformBaseline:
    """Random log-uniform scores — proves eval wiring returns numbers."""

    def __init__(self, vocab_size: int = 1000, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._vocab_size = max(vocab_size, 2)

    def pseudo_log_likelihood(self, text: str) -> float:
        # Lower (more negative) is worse; score = sum log p(token|context).
        total = 0.0
        for _ in text.split():
            p = self._rng.random() / self._vocab_size
            total += math.log(max(p, 1e-12))
        return total


def smoke_tasks() -> tuple[SmokeTask, ...]:
    return (
        SmokeTask("blimp_agreement", "the cat sleeps", "the cat sleep"),
        SmokeTask("blimp_determiner", "a big dog runs", "a big dogs runs"),
        SmokeTask("ewok_simple", "birds have wings", "birds have engines"),
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


def _mock_accuracy(model: PseudoLogLikelihoodModel, task: SmokeTask) -> float:
    good = model.pseudo_log_likelihood(task.good)
    bad = model.pseudo_log_likelihood(task.bad)
    return 1.0 if good > bad else 0.0


def run_smoke_eval(
    model: PseudoLogLikelihoodModel,
    *,
    mode: RunMode | None = None,
    tasks: tuple[SmokeTask, ...] | None = None,
    output_path: Path | None = None,
    pipeline_root: Path | None = None,
) -> SmokeEvalResult:
    """Run fast smoke evaluation (MOCK or LIVE stub delegation)."""
    root = pipeline_root or eval_root()
    resolved_mode = mode
    if resolved_mode is None:
        resolved_mode = RunMode.LIVE if detect_pipeline_install(root) else RunMode.MOCK

    task_list = tasks or smoke_tasks()
    if resolved_mode is RunMode.MOCK:
        scores = {t.name: _mock_accuracy(model, t) for t in task_list}
        notes = (
            "MOCK harness: minimal-pair accuracy from pseudo_log_likelihood. "
            "Install official pipeline via scripts/setup_babylm_eval_pipeline.sh for LIVE."
        )
    else:
        scores = _live_smoke_stub(model, root, task_list)
        notes = "LIVE stub: pipeline detected; full zero-shot run requires psalm[ml] + HF model."

    aggregate = sum(scores.values()) / len(scores) if scores else 0.0
    result = SmokeEvalResult(
        mode=resolved_mode,
        task_scores=scores,
        aggregate_score=aggregate,
        pipeline_root=root if resolved_mode is RunMode.LIVE else None,
        notes=notes,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        result.report_path = output_path
    return result


def _live_smoke_stub(
    model: PseudoLogLikelihoodModel,
    root: Path,
    tasks: tuple[SmokeTask, ...],
) -> dict[str, float]:
    """Placeholder LIVE path: same minimal pairs until HF backend is wired.

    TODO: invoke ``python -m evaluation_pipeline.sentence_zero_shot.run`` with a
    converted HF checkpoint. For now we verify install layout and still score pairs.
    """
    if not detect_pipeline_install(root):
        raise FileNotFoundError(
            f"BabyLM pipeline not found under {root}. Run scripts/setup_babylm_eval_pipeline.sh"
        )
    return {t.name: _mock_accuracy(model, t) for t in tasks}


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
    """Run one official zero-shot task (requires psalm[ml] env in pipeline clone).

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
