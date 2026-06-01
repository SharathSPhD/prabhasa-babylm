"""Arm-A-only discrimination CALIBRATION sweep (no pre-registration needed).

The first discrimination probe (agent↔theme swap on all lexical items) ceilinged
(A=0.97, B=1.0, C=0.99) — too easy to reveal a B-vs-C effect. Per TRIZ
separation-by-condition, rather than guess one corruption difficulty we sweep a
**graded ladder** and read each tier on the *same* trained baseline:

  swap / distractor  ×  all-lexical / non-canonical(passive+dative)

Crucially this trains **arm A only** and reveals **no B-vs-C comparison**, so it
is a same-instrument diagnostic (like the EM floor-lift) and needs no new ADR.
Its sole purpose: find the corruption tier that lands baseline A in the
informative middle band (~0.60–0.85), off both floor and ceiling. That tier is
then blind-pre-registered (ADR-0015 amendment) before any B/C run.

Run on ``.venv-gpu`` with ``--require-cuda``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from run_h1_cogs_disc_pilot import (  # type: ignore[import-not-found]
    NONCANONICAL_CATEGORIES,
    build_disc_pairs,
    cogs_line,
)

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache/samsadhani_75k.jsonl")
    ap.add_argument("--size", type=float, default=100.0, choices=[60.0, 100.0, 150.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cogs-train", type=int, default=12000)
    ap.add_argument("--lexical-test", type=int, default=400)
    ap.add_argument("--noncanonical-test", type=int, default=400)
    ap.add_argument("--vocab", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--out", default="docs/data/phase2-h1-cogs-calib.json")
    args = ap.parse_args()

    import torch

    cuda_ok = torch.cuda.is_available()
    device = args.device if (not args.device.startswith("cuda") or cuda_ok) else "cpu"
    if args.require_cuda and (not cuda_ok or not device.startswith("cuda")):
        raise SystemExit(f"--require-cuda set but CUDA unreachable (is_available={cuda_ok}).")
    if device.startswith("cuda"):
        print(f"CUDA OK: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")

    paninian = JsonlSentenceSource(args.cache)
    dyck = DyckSentenceSource()
    scrambled = ScramblingGenerator(paninian)

    train_pairs = load_cogs("train", limit=args.cogs_train)
    lex = load_cogs("gen", tier="lexical", limit=args.lexical_test)
    noncanon = load_cogs("gen", categories=NONCANONICAL_CATEGORIES, limit=args.noncanonical_test)
    train_lines = [cogs_line(s, lf) for s, lf in train_pairs]

    # The graded ladder — all scored on the same trained model.
    disc_sets: dict[str, list[tuple[str, str]]] = {
        "swap_all": build_disc_pairs(lex, "swap"),
        "distractor_all": build_disc_pairs(lex, "distractor"),
        "swap_noncanon": build_disc_pairs(noncanon, "swap"),
        "distractor_noncanon": build_disc_pairs(noncanon, "distractor"),
    }
    for name, pairs in disc_sets.items():
        print(f"  {name}: {len(pairs)} pairs")

    pan_sample = [s.text for s in paninian.stream(4000, seed=0)]
    dyck_sample = [s.text for s in dyck.stream(1500, seed=0)]
    tok_dir = Path(".cache/cogs_calib_tok")
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
        disc_eval_sets=disc_sets,
        decode=lambda ids: tok.decode(ids),
        append_eos_to_prompt=False,
        eval_fracs=(),
    )

    arm_a = default_h1_matrix(param_count_m=args.size).arm("A")
    start = time.time()
    per_tier: dict[str, list[float]] = {k: [] for k in disc_sets}
    for seed in range(args.seeds):
        m = runner(arm_a, seed)
        for name in disc_sets:
            per_tier[name].append(round(m.get(f"{name}_disc", 0.0), 4))
        print(f"seed {seed}: " + "  ".join(f"{n}={per_tier[n][-1]}" for n in disc_sets))
    wall = time.time() - start

    print("\n=== ARM-A CALIBRATION (find informative ~0.60-0.85 band) ===")
    summary = {}
    for name in disc_sets:
        mean, lo, hi = mean_ci(per_tier[name])
        band = "FLOOR" if mean < 0.55 else "CEILING" if mean > 0.90 else "INFORMATIVE"
        summary[name] = {
            "mean": mean,
            "ci": [lo, hi],
            "n_pairs": len(disc_sets[name]),
            "band": band,
        }
        print(f"  {name:22s} A={mean:.3f} (CI {lo:.3f}-{hi:.3f})  [{band}]")

    payload = {
        "task": "cogs_discrimination_calibration",
        "purpose": "arm-A-only sweep to locate informative corruption tier (no B/C revealed)",
        "size_m": args.size,
        "seeds": args.seeds,
        "max_steps": args.max_steps,
        "arm_A_by_tier": per_tier,
        "summary": summary,
        "wall_seconds": round(wall, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwall: {wall:.0f}s\nwrote {args.out}")


if __name__ == "__main__":
    main()
