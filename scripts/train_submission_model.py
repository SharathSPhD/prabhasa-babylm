#!/usr/bin/env python3
"""Leaderboard submission-track trainer (ADR-0038) — GPU-only, kept off the H1' path.

This trainer is the *submission* counterpart to ``run_babylm_strict_small.py``. It applies
the leaderboard levers from ``leaderboard_levers`` — Muon (matrices) + AdamW (rest),
decaying / frequency-informed MLM masking, and a progressive sequence-length schedule — at
a larger English budget. It deliberately shares nothing with the ablation training path so
the budget-controlled H1' comparison stays clean. Within-budget data only (Strict-Small):
the dose(s) plus the shared English base, identical token sources to the ablation.

    uv run python scripts/train_submission_model.py --dose-arms A B C D --english-epochs 40 \
        --require-cuda --out data/checkpoints/submission

Run only when the GPU is free (the await-watcher launches it as part of close-out if wired).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import sentencepiece as spm
import torch
from torch import nn

from psalm.infrastructure.ml.elc_psalm import HybridObjective, make_mlm_mask
from psalm.infrastructure.ml.elc_trainer import (
    build_elc_encoder,
    cosine_warmup_lr,
    save_elc_checkpoint,
)
from psalm.infrastructure.ml.leaderboard_levers import (
    build_submission_optimizers,
    make_freq_informed_mlm_mask,
    progressive_seq_len,
    scheduled_mask_prob,
)
from psalm.infrastructure.ml.packing import TokenPacker

SS = Path("data/corpora/strict_small")
TOK = Path("data/tokenizer/strict_small/spm.model")
ARMS_MANIFEST = Path("docs/data/strict-small-arms.json")
EOS_ID = 2


def _read_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _token_log_freq(lines: list[str], encode, vocab: int) -> torch.Tensor:
    counts = Counter()
    for ln in lines:
        counts.update(encode(ln))
    freq = torch.ones(vocab, dtype=torch.float32)  # add-one smoothing
    for tid, c in counts.items():
        if 0 <= tid < vocab:
            freq[tid] += c
    return torch.log(freq)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dose-arms",
        nargs="+",
        default=["A", "B", "C", "D"],
        help="dose corpora to concatenate for stage 1 (within-budget)",
    )
    ap.add_argument("--arch", default="elc_psalm_s")
    ap.add_argument("--max-seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dose-epochs", type=float, default=4.0)
    ap.add_argument("--english-epochs", type=float, default=40.0)
    ap.add_argument("--peak-lr", type=float, default=1e-3)
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--warmup-frac", type=float, default=0.06)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--mask-start", type=float, default=0.30)
    ap.add_argument("--mask-end", type=float, default=0.15)
    ap.add_argument("--mask-kind", default="cosine")
    ap.add_argument("--freq-alpha", type=float, default=0.5, help="0 disables freq-informed mask")
    ap.add_argument("--no-muon", action="store_true", help="use plain AdamW (ablate the lever)")
    ap.add_argument("--vocab", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/checkpoints/submission")
    ap.add_argument("--require-cuda", action="store_true")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    cuda_ok = torch.cuda.is_available()
    device = "cuda" if cuda_ok else "cpu"
    if args.require_cuda and not cuda_ok:
        raise SystemExit("--require-cuda set but CUDA not reachable.")
    torch.manual_seed(args.seed)
    if cuda_ok:
        torch.cuda.manual_seed_all(args.seed)
        print(f"CUDA: {torch.cuda.get_device_name(0)} | torch {torch.__version__}", flush=True)

    sp = spm.SentencePieceProcessor()
    sp.Load(str(TOK))
    vocab = sp.GetPieceSize()
    assert vocab == args.vocab, f"tokenizer vocab {vocab} != --vocab {args.vocab}"
    encode = lambda s: sp.EncodeAsIds(s)  # noqa: E731

    manifest = json.loads(ARMS_MANIFEST.read_text(encoding="utf-8"))
    dose_lines: list[str] = []
    dose_tokens = 0
    for arm in args.dose_arms:
        dose_lines += _read_lines(SS / "arms" / f"dose_{arm}.txt")
        dose_tokens += int(manifest["arms"][arm]["tokens"])
    base = _read_lines(SS / "english_base.txt")
    base_tokens = int(manifest["english_base"]["tokens"])

    tok_per_step = args.batch_size * args.max_seq_len
    stage1_steps = max(int(args.dose_epochs * dose_tokens / tok_per_step), 1)
    stage2_steps = max(int(args.english_epochs * base_tokens / tok_per_step), 1)
    total_steps = stage1_steps + stage2_steps
    warmup = max(int(args.warmup_frac * total_steps), 1)
    print(
        f"submission: doses={args.dose_arms} dose_tok={dose_tokens} base_tok={base_tokens} "
        f"| stage1={stage1_steps} stage2={stage2_steps} total={total_steps} "
        f"| muon={'off' if args.no_muon else 'on'} mask={args.mask_start}->{args.mask_end} "
        f"freq_alpha={args.freq_alpha} max_seq={args.max_seq_len}",
        flush=True,
    )

    seq_schedule = [
        (0.0, args.max_seq_len // 4),
        (0.4, args.max_seq_len // 2),
        (0.75, args.max_seq_len),
    ]
    log_freq = _token_log_freq(base + dose_lines, encode, vocab)

    model, cfg = build_elc_encoder(
        args.arch,
        vocab_size=vocab,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        mlm_probability=args.mask_start,
    )
    model = model.to(device)
    mask_id = cfg.vocab_size - 1

    if args.no_muon:
        opts = [
            torch.optim.AdamW(
                model.parameters(),
                lr=args.peak_lr,
                weight_decay=0.01,
                betas=(0.9, 0.95),
                fused=cuda_ok,
            )
        ]
    else:
        opts = build_submission_optimizers(
            model, muon_lr=args.muon_lr, adamw_lr=args.peak_lr, weight_decay=0.01
        )

    packer = TokenPacker(encode, eos_id=EOS_ID, seq_len=args.max_seq_len)
    autocast = cuda_ok
    dtype = torch.bfloat16

    def stream(lines: list[str]):
        while True:
            yield from lines

    def lr_at(step: int) -> float:
        return cosine_warmup_lr(
            step, peak_lr=args.peak_lr, warmup_steps=warmup, total_steps=total_steps
        )

    def run_stage(lines: list[str], n_steps: int, offset: int) -> tuple[float, float]:
        it = packer.packed_batches(stream(lines), batch_size=args.batch_size, device=device)
        last = best = float("inf")
        ema = None
        t0 = time.time()
        model.train()
        for local in range(n_steps):
            gstep = offset + local
            batch = next(it)
            cur_seq = progressive_seq_len(gstep, total_steps, seq_schedule)
            if cur_seq < batch.size(1):
                batch = batch[:, :cur_seq].contiguous()
            lr = lr_at(gstep)
            for o in opts:
                for g in o.param_groups:
                    # AdamW group follows the cosine LR; Muon keeps its own (scale-invariant) LR.
                    if not (not args.no_muon and o is opts[0]):
                        g["lr"] = lr
            for o in opts:
                o.zero_grad(set_to_none=True)
            mask_prob = scheduled_mask_prob(
                gstep,
                total_steps,
                p_start=args.mask_start,
                p_end=args.mask_end,
                kind=args.mask_kind,
            )
            ctx = torch.autocast(device_type="cuda", dtype=dtype) if autocast else _null()
            with ctx:
                if gstep % 2 == 1:
                    _, aux = model(batch, objective=HybridObjective.CLM, labels=batch)
                    loss = aux["loss"]
                else:
                    if args.freq_alpha > 0:
                        masked, loss_mask = make_freq_informed_mlm_mask(
                            batch,
                            mask_id=mask_id,
                            probability=mask_prob,
                            token_log_freq=log_freq,
                            alpha=args.freq_alpha,
                            exclude={EOS_ID, mask_id},
                        )
                    else:
                        masked, loss_mask = make_mlm_mask(
                            batch, mask_id=mask_id, probability=mask_prob, exclude={EOS_ID, mask_id}
                        )
                    _, aux = model(
                        masked, objective=HybridObjective.MLM, labels=batch, mlm_mask=loss_mask
                    )
                    loss = aux["loss"]
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            for o in opts:
                o.step()
            v = float(loss.detach())
            last = v
            best = min(best, v)
            ema = v if ema is None else 0.98 * ema + 0.02 * v
            if (local + 1) % 200 == 0:
                rate = (local + 1) / max(time.time() - t0, 1e-6)
                print(
                    f"  step {gstep + 1}/{total_steps} loss={v:.4f} ema={ema:.4f} "
                    f"lr={lr:.2e} seq={cur_seq} maskp={mask_prob:.3f} {rate:.2f} step/s",
                    flush=True,
                )
        return last, best

    t0 = time.time()
    _, b1 = run_stage(dose_lines, stage1_steps, 0)
    last, b2 = run_stage(base, stage2_steps, stage1_steps)
    wall = time.time() - t0
    best = min(b1, b2)

    out_dir = Path(args.out)
    ckpt = out_dir / "elc.pt"
    save_elc_checkpoint(
        ckpt,
        model,
        mask_id=mask_id,
        extra={
            "track": "leaderboard_submission",
            "dose_arms": args.dose_arms,
            "arch": args.arch,
            "tokenizer": str(TOK),
            "muon": not args.no_muon,
            "freq_alpha": args.freq_alpha,
            "mask_start": args.mask_start,
            "mask_end": args.mask_end,
            "max_seq_len": args.max_seq_len,
            "english_epochs": args.english_epochs,
            "final_loss": last,
            "best_loss": best,
        },
    )
    print(
        f"DONE submission: steps={total_steps} final_loss={last:.4f} best_loss={best:.4f} "
        f"wall={wall / 60:.1f}min -> {ckpt}",
        flush=True,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "track": "leaderboard_submission",
                "dose_arms": args.dose_arms,
                "steps": total_steps,
                "final_loss": last,
                "best_loss": best,
                "wall_seconds": wall,
                "checkpoint": str(ckpt),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
