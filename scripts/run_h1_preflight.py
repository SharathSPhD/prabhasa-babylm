"""H1 dose-and-floor pre-flight (gates the full GB10 battery).

Per the battery-launch decision brief
(``docs/contracts/phase-2-battery-launch-decision-2026-05-31.md``): before
committing ~20 GB10-hours to the full A–H × 3 battery, run a cheap probe that
answers the two questions that actually gate the spend —

  1. **Floor:** is the chosen SCAN split learnable at this size, or does it floor
     every arm at ~0% (in which case B-vs-C is undecidable regardless of prior)?
  2. **Dose:** at the diversity-capped structural budget (~0.06–0.2% of the
     downstream budget), does *any* structural prior (B or C) separate from the
     no-prior baseline A — i.e. can the battery be informative at all?

It trains arms A, B, C × N seeds on a *signal* split (default ``simple``, where a
from-scratch model is known to learn) and runs a one-seed *floor probe* of arm A
on a hard split (default ``length``). It then prints per-arm bootstrap CIs, the
B-vs-C and (B/C)-vs-A gaps, and a decision-tree verdict. It deliberately does
**not** auto-launch the battery: the human owns that sign-off.

Run on ``.venv-gpu`` with ``--require-cuda``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from psalm.analysis.comparison_tests import mean_ci
from psalm.application.data.assembly import PrePretrainAssembler
from psalm.application.data.tokenizer import TokenizerSpec
from psalm.domain.experiments.matrix import default_h1_matrix
from psalm.domain.model.config import preset_for
from psalm.domain.model.training import Precision, TrainConfig
from psalm.infrastructure.eval.scan import load_scan
from psalm.infrastructure.generators.dyck_source import DyckSentenceSource
from psalm.infrastructure.generators.jsonl_source import JsonlSentenceSource
from psalm.infrastructure.generators.scramble_source import ScramblingGenerator
from psalm.infrastructure.ml.h1_runner import DEFAULT_EVAL_FRACS, EvalSets, H1Runner
from psalm.infrastructure.tokenizer.sentencepiece_tokenizer import SentencePieceTrainer

EOS_ID = 2
FLOOR_EPS = 0.03  # below this, treat an arm/split as floored
CI_HALF_WIDTH_GATE = 0.03  # < 3 pts -> adequately powered


def scan_line(cmd: str, act: str) -> str:
    return f"IN: {cmd} OUT: {act}"


def _ci(samples: list[float]) -> tuple[float, float, float, float]:
    mean, lo, hi = mean_ci(samples)
    return mean, lo, hi, (hi - lo) / 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache/samsadhani.jsonl")
    ap.add_argument("--size", type=float, default=100.0, choices=[60.0, 100.0, 150.0])
    ap.add_argument("--signal-split", default="simple")
    ap.add_argument("--floor-split", default="length")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--scan-train", type=int, default=12000)
    ap.add_argument("--scan-test", type=int, default=300)
    ap.add_argument("--vocab", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pre-budget", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument(
        "--checkpoints",
        action="store_true",
        help="Record the within-run efficiency curve (4 extra evals/run). Off by "
        "default for the pilot, which only needs each arm's final accuracy.",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch

    cuda_ok = torch.cuda.is_available()
    device = args.device if (not args.device.startswith("cuda") or cuda_ok) else "cpu"
    if args.require_cuda and (not cuda_ok or not device.startswith("cuda")):
        raise SystemExit(
            f"--require-cuda set but CUDA unreachable "
            f"(is_available={cuda_ok}, device={device!r}). Use .venv-gpu on the GB10."
        )
    if device.startswith("cuda"):
        print(f"CUDA OK: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")

    # pre_budget: CLI > diversity report > default.
    pre_budget = args.pre_budget
    if pre_budget is None:
        div = Path("docs/data/phase2-structural-diversity.json")
        pre_budget = (
            int(json.loads(div.read_text())["recommended_pre_budget_tokens"])
            if div.exists()
            else 60_000
        )
    print(f"pre_budget={pre_budget} tokens  size={args.size}M  seeds={args.seeds}")

    paninian = JsonlSentenceSource(args.cache)
    dyck = DyckSentenceSource()
    scrambled = ScramblingGenerator(paninian)

    # Shared tokenizer (SCAN vocab is identical across splits, so one suffices).
    sig_train = load_scan(args.signal_split, which="train", limit=args.scan_train)
    sig_test = load_scan(args.signal_split, which="test", limit=args.scan_test)
    sig_train_lines = [scan_line(c, a) for c, a in sig_train]
    pan_sample = [s.text for s in paninian.stream(4000, seed=0)]
    dyck_sample = [s.text for s in dyck.stream(1500, seed=0)]

    tok_dir = Path(".cache/preflight_tok")
    tok_dir.parent.mkdir(parents=True, exist_ok=True)
    spec = TokenizerSpec(vocab_size=args.vocab, model_type="unigram", sandhi_aware=True)
    tok = SentencePieceTrainer().train(sig_train_lines + pan_sample + dyck_sample, spec, tok_dir)
    vocab = tok.vocab_size
    print(f"tokenizer vocab={vocab} device={device}")

    assembler = PrePretrainAssembler(paninian=paninian, dyck=dyck, paninian_scrambled=scrambled)
    model_cfg = preset_for(args.size, vocab_size=vocab, max_seq_len=args.seq_len)
    train_cfg = TrainConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        warmup_steps=max(args.max_steps // 20, 10),
        precision=Precision.FP32 if device == "cpu" else Precision.BF16,
        device=device,
        log_every=200,
    )

    def make_runner(test_pairs: list[tuple[str, str]], train_lines: list[str]) -> H1Runner:
        return H1Runner(
            assembler=assembler,
            nl_lines=lambda: train_lines,
            encode=tok.encode,
            eos_id=EOS_ID,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            eval_sets=EvalSets(
                compositional=[(f"IN: {c} OUT:", a) for c, a in test_pairs], minimal_pairs=[]
            ),
            decode=lambda ids: tok.decode(ids),
            append_eos_to_prompt=False,
            pre_budget_tokens=pre_budget,
            eval_fracs=DEFAULT_EVAL_FRACS if args.checkpoints else (),
        )

    matrix = default_h1_matrix(param_count_m=args.size)
    sig_runner = make_runner(sig_test, sig_train_lines)

    start = time.time()
    # --- Dose probe: A, B, C on the signal split, N seeds.
    acc: dict[str, list[float]] = {"A": [], "B": [], "C": []}
    for arm_id in ("A", "B", "C"):
        arm = matrix.arm(arm_id)
        for seed in range(args.seeds):
            m = sig_runner(arm, seed)
            a = round(m["compositional_accuracy"], 4)
            acc[arm_id].append(a)
            print(f"[signal:{args.signal_split}] arm {arm_id} seed {seed}: acc={a}")

    # --- Floor probe: arm A on the hard split, 1 seed.
    floor_train = [
        scan_line(c, a)
        for c, a in load_scan(args.floor_split, which="train", limit=args.scan_train)
    ]
    floor_test = load_scan(args.floor_split, which="test", limit=args.scan_test)
    floor_runner = make_runner(floor_test, floor_train)
    floor_a = round(floor_runner(matrix.arm("A"), 0)["compositional_accuracy"], 4)
    print(f"[floor:{args.floor_split}] arm A seed 0: acc={floor_a}")
    wall = time.time() - start

    a_mean, a_lo, a_hi, a_hw = _ci(acc["A"])
    b_mean, b_lo, b_hi, b_hw = _ci(acc["B"])
    c_mean, c_lo, c_hi, c_hw = _ci(acc["C"])
    struct_best = max(b_mean, c_mean)
    struct_vs_a = struct_best - a_mean
    b_vs_c = b_mean - c_mean
    max_hw = max(a_hw, b_hw, c_hw)

    # --- Decision-tree verdict (advisory; the human owns the launch call).
    if a_mean < FLOOR_EPS:
        verdict = "HARNESS_BROKEN: arm A floored on the signal split — fix wiring before any battery."
    elif struct_vs_a < max_hw:
        verdict = (
            "DOSE_TOO_SMALL: no structural arm separates from A beyond noise. "
            "Apply matched-epoch dose and/or expand the generator; re-pilot. DO NOT launch."
        )
    elif max_hw >= CI_HALF_WIDTH_GATE:
        verdict = (
            f"UNDERPOWERED: separation present (struct−A={struct_vs_a:.3f}) but CI half-width "
            f"{max_hw:.3f} ≥ {CI_HALF_WIDTH_GATE}. Raise seeds/budget, then launch."
        )
    else:
        verdict = (
            f"DOSE_ADEQUATE: structural arm separates from A (struct−A={struct_vs_a:.3f}, "
            f"half-width {max_hw:.3f}). Floor split usable={floor_a >= FLOOR_EPS}. "
            "Cleared to launch the full A–H × ≥3 battery (human sign-off)."
        )

    payload = {
        "size_m": args.size,
        "signal_split": args.signal_split,
        "floor_split": args.floor_split,
        "seeds": args.seeds,
        "max_steps": args.max_steps,
        "pre_budget_tokens": pre_budget,
        "vocab": vocab,
        "wall_seconds": round(wall, 1),
        "accuracy": acc,
        "summary": {
            "A": {"mean": a_mean, "ci": [a_lo, a_hi], "half_width": a_hw},
            "B": {"mean": b_mean, "ci": [b_lo, b_hi], "half_width": b_hw},
            "C": {"mean": c_mean, "ci": [c_lo, c_hi], "half_width": c_hw},
            "struct_best_minus_A": struct_vs_a,
            "B_minus_C": b_vs_c,
            "floor_split_A": floor_a,
        },
        "verdict": verdict,
    }
    out_path = Path(args.out or f"docs/data/phase2-h1-preflight-{int(args.size)}m.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print("\n=== PRE-FLIGHT SUMMARY ===")
    print(f"A: {a_mean:.3f} (CI {a_lo:.3f}-{a_hi:.3f}, ±{a_hw:.3f})")
    print(f"B: {b_mean:.3f} (CI {b_lo:.3f}-{b_hi:.3f}, ±{b_hw:.3f})")
    print(f"C: {c_mean:.3f} (CI {c_lo:.3f}-{c_hi:.3f}, ±{c_hw:.3f})")
    print(f"struct_best − A : {struct_vs_a:+.3f}")
    print(f"B − C          : {b_vs_c:+.3f}")
    print(f"floor split A  : {floor_a:.3f} (split floored={floor_a < FLOOR_EPS})")
    print(f"wall           : {wall:.0f}s")
    print(f"VERDICT: {verdict}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
