"""H1′ Paribhāṣā vs Dyck pilot harness (ADR-0030).

Wires L2 pre-pretrain sources (Śabdabodha-aligned / Paribhāṣā / matched Dyck /
optional Pāṇinian L1) through :class:`H1Runner` and the ADR-0016 sample-efficiency
readout. **Does not auto-launch** the battery.

Primary venue at scale: official BabyLM EWoK + BLiMP argument-structure minimal pairs
(requires eval-train zero-shot — see ``docs/experiments/h1prime-plan.md``).

``--smoke``: tiny CPU run using COGS noncanonical discrimination as a stand-in venue
so the pipeline is proven end-to-end before eval shards exist.

    uv sync --extra dev --extra ml
    uv run python scripts/run_h1prime_pilot.py --smoke
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Iterator
from pathlib import Path

import yaml

from psalm.analysis.comparison_tests import mean_ci
from psalm.application.data.assembly import PrePretrainAssembler
from psalm.application.data.ports import AnnotatedSentence
from psalm.application.data.tokenizer import TokenizerSpec
from psalm.domain.data.diversity import summarize
from psalm.domain.data.dyck import DyckConfig
from psalm.domain.data.matching import match_dyck
from psalm.domain.eval.discrimination import CORRUPTIONS
from psalm.domain.experiments.models import (
    ExperimentArm,
    PrePretrainSource,
    PretrainCorpus,
)
from psalm.domain.model.config import ModelConfig, preset_for
from psalm.domain.model.training import Precision, TrainConfig
from psalm.infrastructure.eval.cogs import load_cogs
from psalm.infrastructure.generators.dyck_source import DyckSentenceSource
from psalm.infrastructure.generators.jsonl_source import JsonlSentenceSource
from psalm.infrastructure.generators.paribhasha_source import ParibhashaSentenceSource
from psalm.infrastructure.generators.shabdabodha_aligned_source import DEFAULT_ALIGNED_FIXTURE
from psalm.infrastructure.ml.h1_runner import EvalSets, H1Runner
from psalm.infrastructure.tokenizer.sentencepiece_tokenizer import SentencePieceTrainer

EOS_ID = 2
ROOT = Path(__file__).resolve().parents[1]
GO_NO_GO = ROOT / "configs/research/h1p/go_no_go.yaml"
SMOKE_CFG = ROOT / "configs/research/h1p/pilot_smoke.yaml"

# Pre-registered constants (ADR-0030 D5/D6) — mirrored in go_no_go.yaml.
THETA_DEFAULT = 0.70
EVAL_FRACS_BATTERY = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8, 1.0)
DELTA_AUC_MIN = 0.02
EARLY_FRAC = 0.1
CORRUPTION_SMOKE = "swap"
NONCANONICAL_CATEGORIES = frozenset(
    {"active_to_passive", "passive_to_active", "do_dative_to_pp_dative", "pp_dative_to_do_dative"}
)

_DYCK_CANDIDATES = [
    DyckConfig(bracket_types=2, max_depth=4, n_shuffles=1, min_len=8, max_len=64),
    DyckConfig(bracket_types=3, max_depth=6, n_shuffles=2, min_len=8, max_len=128),
    DyckConfig(bracket_types=4, max_depth=8, n_shuffles=2, min_len=12, max_len=160),
    DyckConfig(bracket_types=5, max_depth=8, n_shuffles=3, min_len=16, max_len=192),
    DyckConfig(bracket_types=8, max_depth=10, n_shuffles=3, min_len=16, max_len=256),
]

_PRAKARATA = re.compile(
    r"PRAKARATA\((?P<gun>[^:]+):(?P<gnode>[^,]+),(?P<drav>[^:]+):(?P<dnode>[^)]+)\)"
)


class ParibhashaStringSource:
    """Streams ``paribhasha_string`` from aligned JSONL (L2 training line for h1p:B)."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def stream(self, n: int, *, seed: int = 0) -> Iterator[AnnotatedSentence]:
        if n < 0:
            raise ValueError("n must be non-negative")
        rows: list[dict[str, object]] = []
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if not rows:
            return
        import random

        order = list(range(len(rows)))
        random.Random(seed).shuffle(order)
        for i in range(n):
            obj = rows[order[i % len(order)]]
            pb = str(obj.get("paribhasha_string", ""))
            if not pb:
                continue
            yield AnnotatedSentence(
                text=pb,
                language="sa",
                meta={"source": "shabdabodha_aligned", "schema_version": "paribhasha_aligned_v1"},
            )


def h1p_arm(
    suffix: str,
    *,
    source: PrePretrainSource,
    label: str,
    param_count_m: float,
    token_budget: int,
) -> ExperimentArm:
    return ExperimentArm(
        arm_id=f"h1p:{suffix}",
        label=label,
        pre_pretrain=source,
        pretrain_corpus=PretrainCorpus.ENGLISH,
        param_count_m=param_count_m,
        token_budget=token_budget,
    )


def default_h1p_arms(
    param_count_m: float, *, nl_budget: int, include_l1: bool
) -> dict[str, ExperimentArm]:
    arms = {
        "A": h1p_arm(
            "A",
            source=PrePretrainSource.NONE,
            label="baseline",
            param_count_m=param_count_m,
            token_budget=nl_budget,
        ),
        "B": h1p_arm(
            "B",
            source=PrePretrainSource.SHABDABODHA_ALIGNED,
            label="sabdabodha_aligned_l2",
            param_count_m=param_count_m,
            token_budget=nl_budget,
        ),
        "C": h1p_arm(
            "C",
            source=PrePretrainSource.DYCK,
            label="matched_dyck_control",
            param_count_m=param_count_m,
            token_budget=nl_budget,
        ),
    }
    if include_l1:
        arms["L"] = h1p_arm(
            "L",
            source=PrePretrainSource.PANINIAN,
            label="l1_paninian_contrast",
            param_count_m=param_count_m,
            token_budget=nl_budget,
        )
    return arms


def match_dyck_to_paribhasha(
    paribhasha_lines: list[str], *, n: int = 500, seed: int = 0
) -> DyckConfig:
    targets = summarize(paribhasha_lines)
    return match_dyck(targets, _DYCK_CANDIDATES, n=n, seed=seed).config


def cogs_line(sent: str, lf: str) -> str:
    return f"IN: {sent} OUT: {lf}"


def build_disc_pairs(
    pairs: list[tuple[str, str]], corruption: str = "swap"
) -> list[tuple[str, str]]:
    fn = CORRUPTIONS[corruption]
    out: list[tuple[str, str]] = []
    for sent, lf in pairs:
        corrupt = fn(lf)
        if corrupt is not None:
            out.append((cogs_line(sent, lf), cogs_line(sent, corrupt)))
    return out


def corrupt_graph_prakarata(pb: str) -> str | None:
    """Swap PRAKARATA gun/dravya node ids — secondary graph probe (ADR-0030 D4)."""
    m = _PRAKARATA.search(pb)
    if not m:
        return None
    repl = f"PRAKARATA({m.group('drav')}:{m.group('dnode')},{m.group('gun')}:{m.group('gnode')})"
    return pb[: m.start()] + repl + pb[m.end() :]


def build_graph_pairs(path: Path, limit: int = 200) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if len(out) >= limit:
                break
            obj = json.loads(line)
            gold = str(obj.get("paribhasha_string", ""))
            bad = corrupt_graph_prakarata(gold)
            if bad and bad != gold:
                out.append((gold, bad))
    return out


def curve_points(metrics: dict[str, float]) -> list[tuple[int, float]]:
    pts = [
        (int(k.removeprefix("acc_at_")), v) for k, v in metrics.items() if k.startswith("acc_at_")
    ]
    return sorted(pts)


def normalized_auc(pts: list[tuple[int, float]]) -> float:
    if len(pts) < 2:
        return pts[0][1] if pts else 0.0
    area = 0.0
    for (t0, a0), (t1, a1) in zip(pts, pts[1:], strict=False):
        area += (a0 + a1) / 2.0 * (t1 - t0)
    span = pts[-1][0] - pts[0][0]
    return area / span if span > 0 else pts[-1][1]


def tokens_to_threshold(pts: list[tuple[int, float]], theta: float) -> float | None:
    for (t0, a0), (t1, a1) in zip(pts, pts[1:], strict=False):
        if a1 >= theta:
            if a0 >= theta:
                return float(t0)
            frac = (theta - a0) / (a1 - a0) if a1 != a0 else 0.0
            return t0 + frac * (t1 - t0)
    if pts and pts[0][1] >= theta:
        return float(pts[0][0])
    return None


def early_accuracy(pts: list[tuple[int, float]], budget: int, frac: float) -> float:
    target = frac * budget
    return min(pts, key=lambda p: abs(p[0] - target))[1] if pts else 0.0


def load_yaml(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def verdict_h1prime(
    *,
    auc: dict[str, list[float]],
    curves: dict[str, list[list[tuple[int, float]]]],
    t_theta: dict[str, list[float | None]],
    early: dict[str, list[float]],
    seeds: int,
    theta: float,
) -> tuple[str, float, float, float]:
    """Return (verdict, d_auc, d_auc_lo, d_auc_hi)."""
    if not all(k in auc for k in ("A", "B", "C")):
        return (
            "INCOMPLETE: need arms h1p:A, h1p:B, h1p:C.",
            float("nan"),
            float("nan"),
            float("nan"),
        )

    diffs = [auc["B"][i] - auc["C"][i] for i in range(seeds)]
    d_auc, d_auc_lo, d_auc_hi = mean_ci(diffs)
    a_reaches = any(t is not None for t in t_theta["A"])
    earliest_all_ceiling = all(
        curves[a][s][0][1] >= theta for a in ("A", "B", "C") for s in range(seeds)
    )
    t_b = [t for t in t_theta["B"] if t is not None]
    t_c = [t for t in t_theta["C"] if t is not None]
    d_eff = (sum(t_c) / len(t_c) - sum(t_b) / len(t_b)) if (t_b and t_c) else float("nan")
    early_gap = mean_ci([early["B"][i] - early["C"][i] for i in range(seeds)])[0]
    concordant = (d_eff == d_eff and d_eff > 0) and early_gap > 0

    if earliest_all_ceiling:
        return (
            "VENUE_SATURATES: all arms already ≥θ at earliest checkpoint "
            "(parallel to H1 COGS saturation; do not escalate on this venue).",
            d_auc,
            d_auc_lo,
            d_auc_hi,
        )
    if not a_reaches:
        return (
            f"NOT_LEARNED_AT_SCALE: baseline h1p:A never reaches θ={theta}.",
            d_auc,
            d_auc_lo,
            d_auc_hi,
        )
    if d_auc >= DELTA_AUC_MIN and d_auc_lo > 0 and concordant:
        return (
            f"LAUNCH_RECOMMENDED: ΔAUC={d_auc:+.3f} (CI {d_auc_lo:+.3f}..{d_auc_hi:+.3f}), "
            f"Δeff={d_eff:+.0f} tok, early gap={early_gap:+.3f}. Human sign-off required.",
            d_auc,
            d_auc_lo,
            d_auc_hi,
        )
    return (
        f"NO_OR_WEAK_EFFICIENCY_GAIN: ΔAUC={d_auc:+.3f} "
        f"(CI {d_auc_lo:+.3f}..{d_auc_hi:+.3f}), early gap={early_gap:+.3f}. "
        "Report as a real null (ADR-0030 stopping rule).",
        d_auc,
        d_auc_lo,
        d_auc_hi,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="H1′ pilot harness (ADR-0030)")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU wiring run")
    ap.add_argument("--config", type=Path, default=None, help="YAML overrides")
    ap.add_argument("--aligned", type=Path, default=DEFAULT_ALIGNED_FIXTURE)
    ap.add_argument("--cache", default="data/cache/samsadhani_75k.jsonl")
    ap.add_argument("--size", type=float, default=100.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", default="A,B,C")
    ap.add_argument("--include-l1", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg_doc = load_yaml(SMOKE_CFG if args.smoke else GO_NO_GO)
    if args.config:
        cfg_doc = {**cfg_doc, **load_yaml(args.config)}

    import torch

    cuda_ok = torch.cuda.is_available()
    device = args.device if (not args.device.startswith("cuda") or cuda_ok) else "cpu"
    if args.require_cuda and (not cuda_ok or not device.startswith("cuda")):
        raise SystemExit(f"--require-cuda set but CUDA unreachable (is_available={cuda_ok}).")
    if device.startswith("cuda"):
        print(f"CUDA OK: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")

    if args.smoke:
        theta = float(cfg_doc.get("theta", THETA_DEFAULT))
        eval_fracs: tuple[float, ...] = tuple(cfg_doc.get("eval_fracs", [0.25, 0.5, 1.0]))
        max_steps = int(cfg_doc.get("max_steps", 40))
        batch_size = int(cfg_doc.get("batch_size", 4))
        seq_len = int(cfg_doc.get("seq_len", 64))
        vocab_size = int(cfg_doc.get("vocab_size", 512))
        pre_budget = int(cfg_doc.get("pre_budget_tokens", 800))
        pre_epochs = int(cfg_doc.get("pre_epochs", 1))
        seeds = int(cfg_doc.get("seeds", 1))
        size_m = float(cfg_doc.get("param_count_m", 10.0))
        cogs_train = 200
        lexical_test = 80
        venue = str(cfg_doc.get("venue_smoke", "cogs_noncanonical_discrimination"))
        aligned = Path(cfg_doc.get("aligned_fixture", args.aligned))
        cache = str(cfg_doc.get("paninian_cache", args.cache))
        nl_budget = max_steps * batch_size * seq_len
    else:
        readout = cfg_doc.get("readout", {}) if isinstance(cfg_doc.get("readout"), dict) else {}
        theta = float(readout.get("theta", THETA_DEFAULT))
        eval_fracs = tuple(readout.get("eval_fracs", EVAL_FRACS_BATTERY))
        dose = cfg_doc.get("dose", {}) if isinstance(cfg_doc.get("dose"), dict) else {}
        pre_budget = int(dose.get("pre_budget_tokens", 60_000))
        pre_epochs = int(dose.get("pre_epochs", 4))
        max_steps = 3000
        batch_size = 24
        seq_len = 256
        vocab_size = 2000
        seeds = args.seeds
        size_m = args.size
        cogs_train = 12_000
        lexical_test = 400
        venue = "babylm_ewok_blimp_arg_BLOCKED"
        aligned = args.aligned
        cache = args.cache
        nl_budget = int(dose.get("nl_budget_proxy_tokens", 13_000_000))

    arm_map = default_h1p_arms(size_m, nl_budget=nl_budget, include_l1=args.include_l1)
    suffixes = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    if args.smoke:
        suffixes = [s.removeprefix("h1p:") for s in suffixes]

    paribhasha_src = ParibhashaStringSource(Path(aligned))
    sample_pb = [s.text for s in paribhasha_src.stream(400, seed=0)]
    if not sample_pb:
        sample_pb = [s.text for s in ParibhashaSentenceSource().stream(400, seed=0)]
    dyck_cfg = match_dyck_to_paribhasha(sample_pb)
    dyck = DyckSentenceSource(dyck_cfg)
    paninian = JsonlSentenceSource(cache) if Path(cache).exists() else None

    assembler = PrePretrainAssembler(
        paninian=paninian,
        dyck=dyck,
        paribhasha=ParibhashaSentenceSource(),
        shabdabodha_aligned=paribhasha_src,
    )

    # Downstream NL + eval pairs (smoke: COGS; battery: blocked on eval-train).
    train_pairs = load_cogs("train", limit=cogs_train)
    lex_test = load_cogs("gen", categories=NONCANONICAL_CATEGORIES, limit=lexical_test)
    curve_pairs = build_disc_pairs(lex_test, CORRUPTION_SMOKE)
    train_lines = [cogs_line(s, lf) for s, lf in train_pairs]
    graph_pairs = build_graph_pairs(Path(aligned), limit=60)

    tok_dir = Path(".cache/h1prime_tok")
    tok_dir.mkdir(parents=True, exist_ok=True)
    dyck_sample = [s.text for s in dyck.stream(500, seed=0)]
    spec = TokenizerSpec(vocab_size=vocab_size, model_type="unigram", sandhi_aware=True)
    tok = SentencePieceTrainer().train(train_lines + sample_pb + dyck_sample, spec, tok_dir)
    print(
        f"H1′ harness venue={venue} curve_pairs={len(curve_pairs)} "
        f"graph_pairs={len(graph_pairs)} dyck_match={dyck_cfg} device={device}"
    )

    if args.smoke:
        model_cfg = ModelConfig(
            vocab_size=tok.vocab_size,
            d_model=64,
            n_layers=2,
            n_heads=4,
            max_seq_len=seq_len,
        )
    else:
        model_cfg = preset_for(size_m, vocab_size=tok.vocab_size, max_seq_len=seq_len)
    train_cfg = TrainConfig(
        max_steps=max_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=3e-4,
        warmup_steps=max(max_steps // 20, 10),
        precision=Precision.FP32 if device == "cpu" else Precision.BF16,
        device=device,
        log_every=max(max_steps // 10, 1),
    )
    runner = H1Runner(
        assembler=assembler,
        nl_lines=lambda: train_lines,
        encode=tok.encode,
        eos_id=EOS_ID,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        eval_sets=EvalSets(compositional=[], minimal_pairs=[]),
        disc_curve_pairs=curve_pairs,
        disc_eval_sets={"graph_prakarata": graph_pairs} if graph_pairs else {},
        decode=lambda ids: tok.decode(ids),
        append_eos_to_prompt=False,
        pre_budget_tokens=pre_budget,
        pre_epochs=pre_epochs,
        eval_fracs=eval_fracs,
        nl_budget_tokens=nl_budget if args.smoke else None,
    )

    start = time.time()
    curves: dict[str, list[list[tuple[int, float]]]] = {s: [] for s in suffixes}
    auc: dict[str, list[float]] = {s: [] for s in suffixes}
    t_theta: dict[str, list[float | None]] = {s: [] for s in suffixes}
    early: dict[str, list[float]] = {s: [] for s in suffixes}
    graph_disc: dict[str, list[float]] = {s: [] for s in suffixes}
    budget = max_steps * train_cfg.tokens_per_step

    for suffix in suffixes:
        arm = arm_map[suffix]
        for seed in range(seeds):
            metrics = runner(arm, seed)
            pts = curve_points(metrics)
            curves[suffix].append(pts)
            auc[suffix].append(round(normalized_auc(pts), 4))
            t_theta[suffix].append(tokens_to_threshold(pts, theta))
            early[suffix].append(round(early_accuracy(pts, budget, EARLY_FRAC), 4))
            graph_disc[suffix].append(metrics.get("graph_prakarata_disc", float("nan")))
            print(
                f"{arm.arm_id} seed {seed}: auc={auc[suffix][-1]} "
                f"T@{theta}={t_theta[suffix][-1]} early={early[suffix][-1]} "
                f"graph={graph_disc[suffix][-1]}"
            )
    wall = time.time() - start

    verdict, d_auc, d_auc_lo, d_auc_hi = verdict_h1prime(
        auc=auc,
        curves=curves,
        t_theta=t_theta,
        early=early,
        seeds=seeds,
        theta=theta,
    )
    if venue.startswith("babylm") and not args.smoke:
        verdict = (
            "BLOCKED_ON_EVAL_TRAIN: primary EWoK/BLiMP shards require official zero-shot "
            f"PLL; smoke venue only. Pilot metrics above use COGS stand-in. {verdict}"
        )

    payload = {
        "task": "h1prime_sample_efficiency",
        "adr": "0030",
        "namespace": "h1p",
        "smoke": args.smoke,
        "venue": venue,
        "venue_note": (
            "COGS noncanonical discrimination is a wiring stand-in until eval-train "
            "enables EWoK + BLiMP-arg (go_no_go.yaml primary_venue)."
        ),
        "theta": theta,
        "eval_fracs": list(eval_fracs),
        "dyck_config": {
            "bracket_types": dyck_cfg.bracket_types,
            "max_depth": dyck_cfg.max_depth,
            "n_shuffles": dyck_cfg.n_shuffles,
            "min_len": dyck_cfg.min_len,
            "max_len": dyck_cfg.max_len,
        },
        "size_m": size_m,
        "seeds": seeds,
        "pre_budget_tokens": pre_budget,
        "pre_epochs": pre_epochs,
        "wall_seconds": round(wall, 1),
        "auc": {f"h1p:{k}": v for k, v in auc.items()},
        "tokens_to_theta": {f"h1p:{k}": t_theta[k] for k in suffixes},
        "early_accuracy": {f"h1p:{k}": early[k] for k in suffixes},
        "graph_prakarata_disc": {f"h1p:{k}": graph_disc[k] for k in suffixes},
        "delta_auc_B_minus_C": {"mean": d_auc, "ci": [d_auc_lo, d_auc_hi]},
        "verdict": verdict,
    }
    out_path = args.out or (
        ROOT / "docs/data/h1prime-smoke.json"
        if args.smoke
        else ROOT / f"docs/data/h1prime-pilot-{int(size_m)}m.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== H1′ PILOT (vs pre-registered ADR-0030) ===")
    for suffix in suffixes:
        m_auc, _, _ = mean_ci(auc[suffix])
        print(f"h1p:{suffix}: AUC≈{m_auc:.3f}")
    print(f"ΔAUC (B−C): {d_auc:+.3f} (CI {d_auc_lo:+.3f}..{d_auc_hi:+.3f})")
    print(f"wall: {wall:.0f}s\nVERDICT: {verdict}\nwrote {out_path}")


if __name__ == "__main__":
    main()
