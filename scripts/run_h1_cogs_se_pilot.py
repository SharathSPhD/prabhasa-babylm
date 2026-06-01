"""H1 COGS **sample-efficiency** pilot (ADR-0016, amended) — the learning-curve readout.

Final-state metrics are saturated for all arms (EM floored ~0.02; discrimination
ceilinged 0.91–0.99 across every corruption tier). Per TRIZ separation-by-time and
the program's sample-efficiency thesis, the H1 readout becomes the
**discrimination learning curve**: train each arm, score role-discrimination
(distractor-rebinding corruption) at fine downstream checkpoints, and ask whether
the Pāṇinian prior (B) climbs to role competence with fewer tokens than Dyck (C).

Pre-registered BLIND in ADR-0016 (amended): PRIMARY = normalized AUC of the
discrimination-vs-tokens curve; concordant SECONDARIES = tokens-to-threshold
(θ=0.80) and early-checkpoint (0.1-budget) accuracy gap; conjunctive verdict;
explicit steepness ("UNRESOLVED_AT_PROXY") and never-reach ("NOT_LEARNED")
branches. Does NOT auto-launch the battery.

Run on ``.venv-gpu`` with ``--require-cuda``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from run_h1_cogs_disc_pilot import build_disc_pairs, cogs_line  # type: ignore[import-not-found]

from psalm.analysis.comparison_tests import mean_ci
from psalm.application.data.assembly import PrePretrainAssembler
from psalm.application.data.tokenizer import TokenizerSpec
from psalm.domain.experiments.matrix import default_h1_matrix
from psalm.domain.model.config import preset_for
from psalm.domain.model.training import Precision, TrainConfig
from psalm.infrastructure.eval.cogs import load_cogs
from psalm.infrastructure.generators.dyck_source import DyckSentenceSource
from psalm.infrastructure.generators.jsonl_source import JsonlSentenceSource
from psalm.infrastructure.generators.scramble_source import ScramblingGenerator
from psalm.infrastructure.ml.h1_runner import EvalSets, H1Runner
from psalm.infrastructure.tokenizer.sentencepiece_tokenizer import SentencePieceTrainer

EOS_ID = 2

# --- Pre-registered constants (ADR-0016 D2 amended). DO NOT tune after results.
THETA = 0.80
EVAL_FRACS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8, 1.0)
DELTA_AUC_MIN = 0.02
EARLY_FRAC = 0.1
CORRUPTION = "distractor"  # most role-faithful, lowest final ceiling (0.914)


def curve_points(metrics: dict[str, float]) -> list[tuple[int, float]]:
    """Extract sorted (nl_tokens, disc_accuracy) checkpoints from a run's metrics."""
    pts = [
        (int(k.removeprefix("acc_at_")), v)
        for k, v in metrics.items()
        if k.startswith("acc_at_")
    ]
    return sorted(pts)


def normalized_auc(pts: list[tuple[int, float]]) -> float:
    """Trapezoidal integral of accuracy vs tokens / token span = mean curve height."""
    if len(pts) < 2:
        return pts[0][1] if pts else 0.0
    area = 0.0
    for (t0, a0), (t1, a1) in zip(pts, pts[1:], strict=False):
        area += (a0 + a1) / 2.0 * (t1 - t0)
    span = pts[-1][0] - pts[0][0]
    return area / span if span > 0 else pts[-1][1]


def tokens_to_threshold(pts: list[tuple[int, float]], theta: float) -> float | None:
    """Interpolated tokens to first reach ``theta``; None if never reached."""
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
    """Accuracy at the checkpoint closest to ``frac`` of the token budget."""
    target = frac * budget
    return min(pts, key=lambda p: abs(p[0] - target))[1] if pts else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache/samsadhani_75k.jsonl")
    ap.add_argument("--size", type=float, default=100.0, choices=[60.0, 100.0, 150.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cogs-train", type=int, default=12000)
    ap.add_argument("--lexical-test", type=int, default=400)
    ap.add_argument("--vocab", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--pre-epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pre-budget", type=int, default=None)
    ap.add_argument("--arms", default="A,B,C")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch

    cuda_ok = torch.cuda.is_available()
    device = args.device if (not args.device.startswith("cuda") or cuda_ok) else "cpu"
    if args.require_cuda and (not cuda_ok or not device.startswith("cuda")):
        raise SystemExit(f"--require-cuda set but CUDA unreachable (is_available={cuda_ok}).")
    if device.startswith("cuda"):
        print(f"CUDA OK: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")

    pre_budget = args.pre_budget
    if pre_budget is None:
        div = Path("docs/data/phase2-structural-diversity.json")
        pre_budget = (
            int(json.loads(div.read_text())["recommended_pre_budget_tokens"])
            if div.exists()
            else 60_000
        )

    paninian = JsonlSentenceSource(args.cache)
    dyck = DyckSentenceSource()
    scrambled = ScramblingGenerator(paninian)

    train_pairs = load_cogs("train", limit=args.cogs_train)
    lex_test = load_cogs("gen", tier="lexical", limit=args.lexical_test)
    curve_pairs = build_disc_pairs(lex_test, CORRUPTION)
    train_lines = [cogs_line(s, lf) for s, lf in train_pairs]
    pan_sample = [s.text for s in paninian.stream(4000, seed=0)]
    dyck_sample = [s.text for s in dyck.stream(1500, seed=0)]
    print(f"COGS train={len(train_lines)} curve_pairs={len(curve_pairs)} ({CORRUPTION})")

    tok_dir = Path(".cache/cogs_se_tok")
    tok_dir.parent.mkdir(parents=True, exist_ok=True)
    spec = TokenizerSpec(vocab_size=args.vocab, model_type="unigram", sandhi_aware=True)
    tok = SentencePieceTrainer().train(train_lines + pan_sample + dyck_sample, spec, tok_dir)
    print(f"tokenizer vocab={tok.vocab_size} device={device}")

    assembler = PrePretrainAssembler(paninian=paninian, dyck=dyck, paninian_scrambled=scrambled)
    model_cfg = preset_for(args.size, vocab_size=tok.vocab_size, max_seq_len=args.seq_len)
    train_cfg = TrainConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        warmup_steps=max(args.max_steps // 20, 10),
        precision=Precision.FP32 if device == "cpu" else Precision.BF16,
        device=device,
        log_every=250,
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
        decode=lambda ids: tok.decode(ids),
        append_eos_to_prompt=False,
        pre_budget_tokens=pre_budget,
        pre_epochs=args.pre_epochs,
        eval_fracs=EVAL_FRACS,
    )

    matrix = default_h1_matrix(param_count_m=args.size)
    arm_ids = [a.strip() for a in args.arms.split(",") if a.strip()]

    start = time.time()
    curves: dict[str, list[list[tuple[int, float]]]] = {a: [] for a in arm_ids}
    auc: dict[str, list[float]] = {a: [] for a in arm_ids}
    t_theta: dict[str, list[float | None]] = {a: [] for a in arm_ids}
    early: dict[str, list[float]] = {a: [] for a in arm_ids}
    budget = args.max_steps * train_cfg.tokens_per_step
    for arm_id in arm_ids:
        arm = matrix.arm(arm_id)
        for seed in range(args.seeds):
            m = runner(arm, seed)
            pts = curve_points(m)
            curves[arm_id].append(pts)
            auc[arm_id].append(round(normalized_auc(pts), 4))
            t_theta[arm_id].append(tokens_to_threshold(pts, THETA))
            early[arm_id].append(round(early_accuracy(pts, budget, EARLY_FRAC), 4))
            print(
                f"arm {arm_id} seed {seed}: auc={auc[arm_id][-1]} "
                f"T@{THETA}={t_theta[arm_id][-1]} early={early[arm_id][-1]} "
                f"final={pts[-1][1] if pts else 'NA'}"
            )
    wall = time.time() - start

    summary = {a: {"auc_mean": mean_ci(auc[a])[0], "early_mean": mean_ci(early[a])[0]} for a in arm_ids}

    verdict = "INCOMPLETE: need arms A, B, C."
    d_auc = d_auc_lo = d_auc_hi = float("nan")
    if all(a in auc for a in ("A", "B", "C")):
        diffs = [auc["B"][i] - auc["C"][i] for i in range(args.seeds)]
        d_auc, d_auc_lo, d_auc_hi = mean_ci(diffs)
        a_reaches = any(t is not None for t in t_theta["A"])
        # steepness: every arm already >= theta at the earliest checkpoint.
        earliest_all_ceiling = all(
            curves[a][s][0][1] >= THETA for a in ("A", "B", "C") for s in range(args.seeds)
        )
        t_b = [t for t in t_theta["B"] if t is not None]
        t_c = [t for t in t_theta["C"] if t is not None]
        d_eff = (
            (sum(t_c) / len(t_c) - sum(t_b) / len(t_b)) if (t_b and t_c) else float("nan")
        )
        early_gap = mean_ci([early["B"][i] - early["C"][i] for i in range(args.seeds)])[0]
        concordant = (d_eff == d_eff and d_eff > 0) and early_gap > 0
        if earliest_all_ceiling:
            verdict = (
                "UNRESOLVED_AT_PROXY: all arms already ≥θ at the earliest checkpoint; "
                "the curve cannot separate arms at proxy scale. Escalate to Phase-3 (Option C)."
            )
        elif not a_reaches:
            verdict = (
                f"NOT_LEARNED_AT_SCALE: baseline A never reaches θ={THETA}. Escalate to Phase-3."
            )
        elif d_auc >= DELTA_AUC_MIN and d_auc_lo > 0 and concordant:
            verdict = (
                f"LAUNCH_RECOMMENDED: ΔAUC={d_auc:+.3f} (CI {d_auc_lo:+.3f}..{d_auc_hi:+.3f}), "
                f"Δeff={d_eff:+.0f} tok, early gap={early_gap:+.3f} (concordant). "
                "Recommendation only — human sign-off required."
            )
        else:
            verdict = (
                f"NO_OR_WEAK_EFFICIENCY_GAIN: ΔAUC={d_auc:+.3f} "
                f"(CI {d_auc_lo:+.3f}..{d_auc_hi:+.3f}), early gap={early_gap:+.3f}. "
                "Pāṇinian prior gives no sample-efficiency advantage over Dyck at proxy scale. "
                "Report as a real null (ADR-0016 stopping rule — no sixth instrument)."
            )

    payload = {
        "task": "cogs_sample_efficiency",
        "adr": "0016",
        "corruption": CORRUPTION,
        "theta": THETA,
        "eval_fracs": list(EVAL_FRACS),
        "size_m": args.size,
        "seeds": args.seeds,
        "max_steps": args.max_steps,
        "pre_epochs": args.pre_epochs,
        "budget_tokens": budget,
        "curve_pairs": len(curve_pairs),
        "vocab": tok.vocab_size,
        "wall_seconds": round(wall, 1),
        "auc": auc,
        "tokens_to_theta": {a: [None if t is None else round(t, 1) for t in t_theta[a]] for a in arm_ids},
        "early_accuracy": early,
        "curves": {a: [[[t, round(v, 4)] for t, v in s] for s in curves[a]] for a in arm_ids},
        "summary": summary,
        "delta_auc_B_minus_C": {"mean": d_auc, "ci": [d_auc_lo, d_auc_hi]},
        "verdict": verdict,
    }
    out_path = Path(args.out or f"docs/data/phase2-h1-cogs-se-{int(args.size)}m.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print("\n=== COGS SAMPLE-EFFICIENCY PILOT (vs pre-registered ADR-0016 bar) ===")
    for a in arm_ids:
        print(f"{a}: AUC={summary[a]['auc_mean']:.3f}  early@{EARLY_FRAC}={summary[a]['early_mean']:.3f}")
    print(f"ΔAUC (B-C): {d_auc:+.3f} (CI {d_auc_lo:+.3f}..{d_auc_hi:+.3f})")
    print(f"wall: {wall:.0f}s\nVERDICT: {verdict}\nwrote {out_path}")


if __name__ == "__main__":
    main()
