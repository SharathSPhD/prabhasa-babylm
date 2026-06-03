"""Information-parity preflight audit runner (ADR-0035 D6, acceptance M-info-parity).

Combines the six checks into a single PASS/FAIL gate and writes
``docs/data/info-parity-audit.json``. Exits non-zero on FAIL so it can BLOCK GPU
training in CI / the orchestrator (GATE 3).

Inputs:
* metrics (1–4) from the paribhāṣā workstream's parity report JSON (``--parity-report``);
  any field may be overridden on the CLI.
* venue early accuracy (5) from a smoke-venue measurement (``--venue-early-accuracy``).
* mock/CPU path count (6) from scanning the battery source (always run here).

    uv run python scripts/run_info_parity_audit.py \
        --parity-report ../PSALM-paribhasha-core/docs/data/paribhasha-parity.json \
        --venue-early-accuracy 0.65
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from psalm.analysis.information_parity import (
    InfoParityInputs,
    audit_information_parity,
    scan_mock_cpu_paths,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data" / "info-parity-audit.json"

# Battery source files that must stay mock/CPU-clean (check C6).
_BATTERY_SRC = [
    ROOT / "src/psalm/benchmarks/babylm_eval.py",
    ROOT / "src/psalm/benchmarks/babylm_models.py",
    ROOT / "src/psalm/cli/eval.py",
    ROOT / "src/psalm/infrastructure/ml/trainer.py",
    ROOT / "src/psalm/infrastructure/ml/device.py",
    ROOT / "src/psalm/infrastructure/ml/h1_runner.py",
    ROOT / "src/psalm/analysis/comparison_tests.py",
    ROOT / "src/psalm/domain/eval/go_no_go.py",
]


def _load_parity(path: Path | None) -> dict[str, float]:
    if path is None or not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Tolerate either the role-multiset or full-parse entropy key naming.
    h = raw.get("h_structure_given_karaka")
    if h is None:
        h = raw.get("h_structure_given_full_parse", raw.get("H_structure_given_full_parse"))
    return {
        "h_structure_given_karaka": float(h) if h is not None else float("nan"),
        "visayata_transitive_fraction": float(
            raw.get("visayata_transitive_fraction", raw.get("transitive_visayata_fraction", "nan"))
        ),
        "template_count": float(raw.get("template_count", raw.get("unique_templates", "nan"))),
        "acceptance_fraction": float(
            raw.get("acceptance_fraction", raw.get("coverage_fraction", "nan"))
        ),
        "acceptance_target": float(raw.get("acceptance_target", 0.80)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity-report", type=Path, default=None)
    parser.add_argument("--venue-early-accuracy", type=float, required=True)
    parser.add_argument("--h-structure-given-karaka", type=float, default=None)
    parser.add_argument("--visayata-transitive-fraction", type=float, default=None)
    parser.add_argument("--template-count", type=int, default=None)
    parser.add_argument("--acceptance-fraction", type=float, default=None)
    parser.add_argument("--acceptance-target", type=float, default=None)
    args = parser.parse_args()

    parity = _load_parity(args.parity_report)

    def pick(cli: object, key: str) -> float:
        if cli is not None:
            return float(cli)  # type: ignore[arg-type]
        if key in parity:
            return parity[key]
        raise SystemExit(
            f"missing metric {key!r}: provide --parity-report with it or pass the CLI flag"
        )

    mock_cpu_hits = scan_mock_cpu_paths(_BATTERY_SRC)
    inputs = InfoParityInputs(
        h_structure_given_karaka=pick(args.h_structure_given_karaka, "h_structure_given_karaka"),
        visayata_transitive_fraction=pick(
            args.visayata_transitive_fraction, "visayata_transitive_fraction"
        ),
        template_count=int(pick(args.template_count, "template_count")),
        acceptance_fraction=pick(args.acceptance_fraction, "acceptance_fraction"),
        acceptance_target=(
            float(args.acceptance_target)
            if args.acceptance_target is not None
            else parity.get("acceptance_target", 0.80)
        ),
        venue_early_accuracy=args.venue_early_accuracy,
        mock_cpu_path_count=len(mock_cpu_hits),
    )
    report = audit_information_parity(inputs)

    payload = report.to_dict()
    payload["mock_cpu_path_hits"] = mock_cpu_hits
    payload["parity_report"] = str(args.parity_report) if args.parity_report else None
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    verdict = "PASS" if report.passed else "FAIL"
    print(f"Wrote {OUT}")
    for check in report.checks:
        mark = "ok" if check.passed else "XX"
        print(f"  [{mark}] {check.id}: {check.value} ({check.threshold})")
    print(f"information-parity audit -> {verdict}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
