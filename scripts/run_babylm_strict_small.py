#!/usr/bin/env python3
"""From-scratch BabyLM Strict-Small training for one arm/seed (ADR-0036).

Two-stage curriculum on the GB10 (GPU-only): pre-pretrain on the token-matched
prior dose (``arms/dose_{ARM}.txt``) for ``--dose-epochs``, then pretrain on the
shared 9M English base for ``--english-epochs``. One continuous warmup+cosine LR
schedule spans both stages. Saves an ELC-PSALM checkpoint for PLL evaluation.

Budget is epoch-based off the frozen token counts in ``strict-small-arms.json``
so every arm trains for the same number of English steps and the same total
compute; only the stage-1 dose type differs (fair H1). GPU-only (ADR-0035).

    uv run python scripts/run_babylm_strict_small.py --arm A --seed 0 --require-cuda \
        --dose-epochs 4 --english-epochs 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sentencepiece as spm

from psalm.domain.model.training import Precision, TrainConfig
from psalm.infrastructure.ml.elc_psalm import ElcPsalmEvaluator
from psalm.infrastructure.ml.elc_trainer import save_elc_checkpoint, train_elc_two_stage

SS = Path("data/corpora/strict_small")
TOK = Path("data/tokenizer/strict_small/spm.model")
ARMS_MANIFEST = Path("docs/data/strict-small-arms.json")
EOS_ID = 2  # SentencePiece default eos

_SMOKE = [
    ("the cat sleeps", "the cat sleep"),
    ("a big dog runs", "a big dogs runs"),
    ("she has finished her work", "she have finished her work"),
    ("the books are on the table", "the books is on the table"),
]


def _read_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "B", "C", "D"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="elc_psalm_s")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dose-epochs", type=float, default=4.0)
    ap.add_argument("--english-epochs", type=float, default=10.0)
    ap.add_argument("--peak-lr", type=float, default=1e-3)
    ap.add_argument("--warmup-frac", type=float, default=0.06, help="warmup as frac of total steps")
    ap.add_argument("--dropout", type=float, default=0.1, help="0.0 = preset default")
    ap.add_argument("--mlm-prob", type=float, default=0.3, help="MLM mask rate (BabyLM small ~0.3)")
    ap.add_argument("--vocab", type=int, default=20000)
    ap.add_argument("--out", default="data/checkpoints/strict_small")
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--smoke-eval", action="store_true")
    ap.add_argument(
        "--progressive-seq",
        action="store_true",
        help="Enable progressive sequence length schedule (64->128->256) during stage-2",
    )
    args = ap.parse_args()

    import torch

    torch.set_float32_matmul_precision("high")  # TF32 matmuls — speed, no quality cost
    cuda_ok = torch.cuda.is_available()
    device = "cuda" if cuda_ok else "cpu"
    if args.require_cuda and not cuda_ok:
        raise SystemExit(f"--require-cuda set but CUDA not reachable (is_available={cuda_ok}).")
    if cuda_ok:
        print(f"CUDA: {torch.cuda.get_device_name(0)} | torch {torch.__version__}", flush=True)

    sp = spm.SentencePieceProcessor()
    sp.Load(str(TOK))
    vocab = sp.GetPieceSize()
    assert vocab == args.vocab, f"tokenizer vocab {vocab} != --vocab {args.vocab}"

    manifest = json.loads(ARMS_MANIFEST.read_text(encoding="utf-8"))
    dose_tokens = int(manifest["arms"][args.arm]["tokens"])
    base_tokens = int(manifest["english_base"]["tokens"])
    tok_per_step = args.batch_size * args.seq_len
    stage1_steps = max(int(args.dose_epochs * dose_tokens / tok_per_step), 1)
    stage2_steps = max(int(args.english_epochs * base_tokens / tok_per_step), 1)
    total_steps = stage1_steps + stage2_steps
    warmup = max(int(args.warmup_frac * total_steps), 1)
    print(
        f"arm {args.arm} seed {args.seed}: dose_tok={dose_tokens} base_tok={base_tokens} "
        f"| stage1={stage1_steps} stage2={stage2_steps} total={total_steps} warmup={warmup} "
        f"| batch={args.batch_size} seq={args.seq_len} peak_lr={args.peak_lr} "
        f"progressive_seq={args.progressive_seq}",
        flush=True,
    )

    dose = _read_lines(SS / "arms" / f"dose_{args.arm}.txt")
    base = _read_lines(SS / "english_base.txt")

    precision = Precision.FP32 if device == "cpu" else Precision.BF16
    stage1_cfg = TrainConfig(
        max_steps=stage1_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.peak_lr,
        warmup_steps=warmup,
        precision=precision,
        device=device,
        seed=args.seed,
        log_every=50,
    )
    stage2_cfg = TrainConfig(
        max_steps=stage2_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.peak_lr,
        warmup_steps=warmup,
        precision=precision,
        device=device,
        seed=args.seed,
        log_every=200,
    )

    t0 = time.time()
    model, outcome, mask_id = train_elc_two_stage(
        args.arch,
        stage1_cfg,
        stage2_cfg,
        lambda: dose,
        lambda: base,
        encode=lambda s: sp.EncodeAsIds(s),
        vocab_size=vocab,
        eos_id=EOS_ID,
        dropout=args.dropout,
        mlm_probability=args.mlm_prob,
    )
    wall = time.time() - t0

    out_dir = Path(args.out) / f"arm_{args.arm}_seed_{args.seed}"
    ckpt = out_dir / "elc.pt"
    save_elc_checkpoint(
        ckpt,
        model,
        mask_id=mask_id,
        extra={
            "arm": args.arm,
            "seed": args.seed,
            "arch": args.arch,
            "tokenizer": str(TOK),
            "outcome": outcome.model_dump(),
            "dose_epochs": args.dose_epochs,
            "english_epochs": args.english_epochs,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "peak_lr": args.peak_lr,
            "dropout": args.dropout,
            "mlm_prob": args.mlm_prob,
            "progressive_seq": getattr(args, "progressive_seq", False),
        },
    )
    print(
        f"DONE arm {args.arm} seed {args.seed}: steps={outcome.steps} "
        f"tokens={outcome.tokens_seen} final_loss={outcome.final_loss:.4f} "
        f"best_loss={outcome.best_loss:.4f} wall={wall / 60:.1f}min -> {ckpt}",
        flush=True,
    )

    summary: dict[str, object] = {
        "arm": args.arm,
        "seed": args.seed,
        "arch": args.arch,
        "steps": outcome.steps,
        "tokens_seen": outcome.tokens_seen,
        "final_loss": outcome.final_loss,
        "best_loss": outcome.best_loss,
        "wall_seconds": wall,
        "checkpoint": str(ckpt),
        "dose_epochs": args.dose_epochs,
        "english_epochs": args.english_epochs,
    }
    if args.smoke_eval:
        ev = ElcPsalmEvaluator(model, lambda s: sp.EncodeAsIds(s), mask_id=mask_id, device=device)
        acc = sum(
            1.0 for g, b in _SMOKE if ev.pseudo_log_likelihood(g) > ev.pseudo_log_likelihood(b)
        ) / len(_SMOKE)
        summary["smoke_acc"] = acc
        print(f"smoke PLL acc (4 pairs, NOT evidence): {acc:.3f}", flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
