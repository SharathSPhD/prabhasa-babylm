#!/usr/bin/env python3
"""Leaderboard submission-track trainer (ADR-0038) — GPU-only, kept off the H1' path.

This trainer is the *submission* counterpart to ``run_babylm_strict_small.py``. It applies
the leaderboard levers from ``leaderboard_levers`` — Muon (matrices) + AdamW (rest),
decaying / frequency-informed MLM masking, and a progressive sequence-length schedule — at
a larger English budget. It deliberately shares nothing with the ablation training path so
the budget-controlled H1' comparison stays clean. Within-budget data only (Strict-Small):
the dose(s) plus the shared English base, identical token sources to the ablation.

    uv run python scripts/train_submission_model.py --dose-arms A B C D --dose-epochs 3 --english-epochs 7 \
        --require-cuda --out data/checkpoints/submission

BabyLM 2026 compliance: dose_epochs=3, english_epochs=7 (total ≤10).  Intermediate checkpoints
are saved every --checkpoint-interval-words (default 1M) for developmental trajectory analysis.

Run only when the GPU is free (the await-watcher launches it as part of close-out if wired).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
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
from psalm.infrastructure.ml.nhot_embeddings import NhotEmbedding, build_nhot_matrix
from psalm.infrastructure.ml.packing import TokenPacker
from psalm.infrastructure.ml.structured_masking import (
    KarakaRoleLookup,
    SalienceTransfer,
    StructuredMaskConfig,
    make_structured_mlm_mask,
)

SS = Path("data/corpora/strict_small")
TOK = Path("data/tokenizer/strict_small/spm.model")
ARMS_MANIFEST = Path("docs/data/strict-small-arms.json")
EOS_ID = 2

# BabyLM 2026 official checkpoint schedule (word counts, compliant with guidelines)
# §: "every 1M words until 10M seen, every 10M words until 100M seen"
_BABYLM_MILESTONES: tuple[int, ...] = (
    *range(1_000_000, 10_000_001, 1_000_000),  # 1M, 2M, ..., 10M
    *range(20_000_000, 200_000_001, 10_000_000),  # 20M, 30M, ..., 200M (covers any epoch budget)
)

# BPE piece suffix patterns → viśeṣaṇa (modifier) role
_SUFFIX_ROLES = (
    "ing",
    "ed",
    "er",
    "est",
    "ly",
    "tion",
    "ness",
    "ment",
    "ful",
    "less",
    "ous",
    "ive",
    "al",
)
_WORD_START = "▁"  # ▁  SentencePiece word-boundary prefix


def _build_bpe_karaka_lookup(sp: spm.SentencePieceProcessor, vocab: int) -> KarakaRoleLookup:
    """BPE-heuristic kāraka role assignment from tokenizer structure.

    Maps each token to a kāraka role purely from its surface form:
      ▁XXX  (word-start)  → karta  (p=0.50) — content-word heads are agent proxies
      continuation ending in common suffixes → visesana (p=0.20) — modifier suffixes
      other continuations → separator (p=0.10) — sub-word glue
    """
    role_map: dict[int, str] = {}
    for tid in range(vocab):
        piece = sp.IdToPiece(tid)
        if piece.startswith("<") and piece.endswith(">"):
            role_map[tid] = "separator"  # special control tokens
        elif piece.startswith(_WORD_START):
            bare = piece[len(_WORD_START) :]
            if any(bare.endswith(sfx) for sfx in _SUFFIX_ROLES):
                role_map[tid] = "visesana"
            else:
                role_map[tid] = "karta"
        else:
            if any(piece.endswith(sfx) for sfx in _SUFFIX_ROLES):
                role_map[tid] = "visesana"
            else:
                role_map[tid] = "separator"
    return KarakaRoleLookup(role_map)


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
    ap.add_argument("--dose-epochs", type=float, default=3.0)
    ap.add_argument("--english-epochs", type=float, default=7.0)
    ap.add_argument(
        "--pos-encoding",
        choices=["absolute", "rope"],
        default="absolute",
        help="Position encoding: 'absolute' (learned embeddings) or 'rope' (rotary embeddings, BLiMP-optimized)",
    )
    ap.add_argument(
        "--checkpoint-interval-words",
        type=int,
        default=0,
        help="Legacy: fixed-interval checkpoints (0=off). Prefer --babylm-checkpoints.",
    )
    ap.add_argument(
        "--babylm-checkpoints",
        action="store_true",
        default=True,
        help="Save at BabyLM 2026 official milestones: 1M–10M every 1M, then every 10M.",
    )
    ap.add_argument("--no-babylm-checkpoints", dest="babylm_checkpoints", action="store_false")
    ap.add_argument("--peak-lr", type=float, default=1e-3)
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--warmup-frac", type=float, default=0.06)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--mask-start", type=float, default=0.30)
    ap.add_argument("--mask-end", type=float, default=0.15)
    ap.add_argument("--mask-kind", default="cosine")
    ap.add_argument("--freq-alpha", type=float, default=0.5, help="0 disables freq-informed mask")
    ap.add_argument(
        "--objective",
        default="hybrid",
        choices=["hybrid", "mlm", "clm"],
        help="training objective: hybrid=50/50 MLM/CLM alternation; mlm=pure MLM (BLiMP-optimized); clm=pure CLM",
    )
    ap.add_argument(
        "--ffn-type",
        default="gelu",
        choices=["gelu", "geglu"],
        help="feed-forward type: gelu (default) or geglu (LTG-BERT gated FFN)",
    )
    ap.add_argument(
        "--norm-type",
        default="layernorm",
        choices=["layernorm", "rmsnorm"],
        help="normalization: layernorm (default) or rmsnorm",
    )
    ap.add_argument("--no-muon", action="store_true", help="use plain AdamW (ablate the lever)")
    ap.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model (result-neutral throughput; auto-falls-back to eager on failure)",
    )
    ap.add_argument(
        "--nhot-embeddings",
        action="store_true",
        default=True,
        help="Vidyut N-hot morpheme-boundary embeddings (H1_MECHANISM)",
    )
    ap.add_argument("--no-nhot-embeddings", dest="nhot_embeddings", action="store_false")
    ap.add_argument(
        "--structured-masking",
        action="store_true",
        default=True,
        help="Paribhāṣā kāraka-aware adaptive masking (H1_MECHANISM)",
    )
    ap.add_argument("--no-structured-masking", dest="structured_masking", action="store_false")
    ap.add_argument(
        "--karaka-budget-match",
        action="store_true",
        help="Rescale the kāraka per-token mask probs so their mean equals the scheduled "
        "rate — matched mask BUDGET vs a uniform control (clean causal contrast, RQ-A). "
        "Off by default to preserve the validated 73.06 recipe.",
    )
    ap.add_argument(
        "--karaka-mode",
        choices=["bpe", "deprel"],
        default="bpe",
        help="Kāraka role assignment mode: 'bpe' (heuristic, legacy) or 'deprel' (real spaCy dependency parser)",
    )
    ap.add_argument(
        "--nhot-mode",
        choices=["heuristic", "real"],
        default="heuristic",
        help="N-hot morphology mode: 'heuristic' (BPE suffix list) or 'real' (Morfessor segmentation)",
    )
    ap.add_argument(
        "--karaka-lookup",
        type=Path,
        default=None,
        help="Path to .npy kāraka role lookup (optional; computed from --karaka-mode if absent)",
    )
    ap.add_argument("--vocab", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--base-dir",
        default="data/corpora/strict_small",
        help="Dir with english_base.txt (+ optional english_base.bin). Use data/corpora/strict for the 100M Small track.",
    )
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
    base_dir = Path(args.base_dir)
    base = _read_lines(base_dir / "english_base.txt")
    base_bin = base_dir / "english_base.bin"
    if base_bin.exists():
        base_tokens = int(len(np.memmap(base_bin, dtype=np.uint16, mode="r")))
    else:
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
        f"freq_alpha={args.freq_alpha} pos_encoding={args.pos_encoding} max_seq={args.max_seq_len}",
        flush=True,
    )

    seq_schedule = [
        (0.0, args.max_seq_len // 4),
        (0.4, args.max_seq_len // 2),
        (0.75, args.max_seq_len),
    ]
    log_freq = _token_log_freq(base + dose_lines, encode, vocab)

    # Vidyut N-hot morpheme-boundary embeddings (H1_MECHANISM)
    nhot_emb: nn.Module | None = None
    if args.nhot_embeddings:
        nhot_matrix = build_nhot_matrix(
            sp, vocab_size=vocab, vidyut_available=False, nhot_mode=args.nhot_mode
        )
        nhot_emb = NhotEmbedding(torch.from_numpy(nhot_matrix).float(), d_model=768)
        print(f"N-hot embeddings: ON (vocab={vocab}, nhot_dim=10, d_model=768)", flush=True)

    model, cfg = build_elc_encoder(
        args.arch,
        vocab_size=vocab,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        mlm_probability=args.mask_start,
        nhot_emb=nhot_emb,
        pos_encoding=args.pos_encoding,
        ffn_type=args.ffn_type,
        norm_type=args.norm_type,
    )
    model = model.to(device)
    if args.compile and device == "cuda":
        # Result-neutral throughput lever (PyTorch 2.x). Guarded: sm_121/Blackwell
        # may not support inductor — fall back to eager rather than abort the run.
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            print("torch.compile: ON", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"torch.compile: FAILED ({e}); continuing eager", flush=True)
    mask_id = cfg.vocab_size - 1
    # build_elc_encoder pins init to manual_seed(0) for reproducible weights (fair
    # cross-arm comparison). Re-seed with args.seed AFTER build so training
    # stochasticity (MLM masking, data order) genuinely varies per seed → valid
    # multi-seed CIs. Without this, every --seed produced identical checkpoints.
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Paribhāṣā kāraka-aware adaptive masking (H1_MECHANISM)
    karaka_lookup: KarakaRoleLookup | None = None
    salience_transfer: SalienceTransfer | None = None
    mask_cfg = StructuredMaskConfig(enabled=False)
    if args.structured_masking:
        if args.karaka_lookup is not None and args.karaka_lookup.exists():
            karaka_lookup = KarakaRoleLookup.from_npy(args.karaka_lookup)
            print(
                f"Kāraka lookup: {args.karaka_lookup} ({len(karaka_lookup._map)} entries)",
                flush=True,
            )
        elif args.karaka_mode == "deprel":
            # Real kāraka lookup from spaCy dependency parsing (H1_MECHANISM, REAL).
            # The lookup is a token→role map over the vocab; a representative sample of
            # sentences covers the vocab fully — parsing all 11.6M (100M track) is ~12h of
            # idle-GPU waste for no quality gain. Cap to a sample for tractable startup.
            _karaka_cap = 400_000
            karaka_sample = base if len(base) <= _karaka_cap else base[:_karaka_cap]
            print(
                f"Building real kāraka lookup (deprel mode) from "
                f"{len(karaka_sample)}/{len(base)} sentences (sampled for vocab coverage)...",
                flush=True,
            )

            from psalm.infrastructure.ml.english_karaka_builder_spacy import (
                build_english_karaka_lookup_spacy,
                load_spacy_model,
            )

            nlp = load_spacy_model("en_core_web_sm")
            karaka_lookup = build_english_karaka_lookup_spacy(nlp, karaka_sample, sp, vocab_size=vocab)

            # Count role distribution
            karta_n = sum(1 for r in karaka_lookup._map.values() if r == "karta")
            karma_n = sum(1 for r in karaka_lookup._map.values() if r == "karma")
            karana_n = sum(1 for r in karaka_lookup._map.values() if r == "karana")
            adhikarana_n = sum(1 for r in karaka_lookup._map.values() if r == "adhikarana")
            apadana_n = sum(1 for r in karaka_lookup._map.values() if r == "apadana")
            sampradana_n = sum(1 for r in karaka_lookup._map.values() if r == "sampradana")
            kriya_n = sum(1 for r in karaka_lookup._map.values() if r == "kriya")
            visesana_n = sum(1 for r in karaka_lookup._map.values() if r == "visesana")
            sep_n = sum(1 for r in karaka_lookup._map.values() if r == "separator")

            print(
                f"Kāraka lookup: deprel (spaCy) ({karta_n} kartā, {karma_n} karma, {karana_n} karaṇa, "
                f"{adhikarana_n} adhikaraṇa, {apadana_n} apādāna, {sampradana_n} sampradāna, "
                f"{kriya_n} kriyā, {visesana_n} viśeṣaṇa, {sep_n} separator)",
                flush=True,
            )
        else:
            # BPE-heuristic kāraka roles: ▁word-starts→kartā(0.50), suffixes→viśeṣaṇa(0.20), rest→separator(0.10)
            karaka_lookup = _build_bpe_karaka_lookup(sp, vocab)
            karta_n = sum(1 for r in karaka_lookup._map.values() if r == "karta")
            visesana_n = sum(1 for r in karaka_lookup._map.values() if r == "visesana")
            sep_n = sum(1 for r in karaka_lookup._map.values() if r == "separator")
            print(
                f"Kāraka lookup: BPE-heuristic ({karta_n} kartā, {visesana_n} viśeṣaṇa, {sep_n} separator)",
                flush=True,
            )
        mask_cfg = StructuredMaskConfig(
            enabled=True,
            mask_prob_start=args.mask_start,
            mask_prob_end=args.mask_end,
        )
        salience_transfer = SalienceTransfer(vocab_size=vocab)
        # Pre-build vectorized prob tensor for O(B*T) GPU gather instead of Python loop
        karaka_lookup.build_vocab_probs(vocab, default_prob=args.mask_start)
        print(f"Structured masking: ON (p={args.mask_start}→{args.mask_end})", flush=True)

    opts: list[torch.optim.Optimizer]
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

    def run_stage(
        lines: list[str],
        n_steps: int,
        offset: int,
        *,
        salience_weights: torch.Tensor | None = None,
        ckpt_interval_tokens: int = 0,
        ckpt_milestones_words: tuple[int, ...] = (),
        words_offset: float = 0.0,
    ) -> tuple[float, float, float]:
        """Returns (last_loss, best_loss, words_processed_this_stage)."""
        it = packer.packed_batches(stream(lines), batch_size=args.batch_size, device=device)
        last = best = float("inf")
        ema = None
        t0 = time.time()
        model.train()
        words_done = words_offset
        tokens_this_stage = 0
        next_ckpt_tokens = ckpt_interval_tokens if ckpt_interval_tokens > 0 else 0
        # BabyLM milestone checkpoints: pending milestones greater than words_offset
        pending_milestones = [m for m in ckpt_milestones_words if m > words_offset]
        milestone_idx = 0
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
                _do_clm = args.objective == "clm" or (args.objective == "hybrid" and gstep % 2 == 1)
                if _do_clm:
                    _, aux = model(batch, objective=HybridObjective.CLM, labels=batch)
                    loss = aux["loss"]
                else:
                    if mask_cfg.enabled and karaka_lookup is not None and karaka_lookup._map:
                        # Paribhāṣā kāraka-aware adaptive masking (H1_MECHANISM, full lookup)
                        prob_tensor = karaka_lookup.mask_probs_for_ids(
                            batch, default_prob=mask_prob
                        )
                        if args.karaka_budget_match:
                            # Preserve total mask BUDGET: rescale per-token probs so their
                            # mean equals the scheduled rate, so the kāraka arm masks the
                            # same expected count as a uniform control — isolating the
                            # *distribution* effect (RQ-A clean causal contrast).
                            _m = prob_tensor.mean().clamp(min=1e-6)
                            prob_tensor = (prob_tensor * (mask_prob / _m)).clamp(0.0, 0.95)
                        masked, loss_mask = make_structured_mlm_mask(
                            batch,
                            mask_id=mask_id,
                            prob_tensor=prob_tensor,
                            exclude={EOS_ID, mask_id},
                        )
                        if salience_transfer is not None:
                            salience_transfer.record_batch(batch, loss_mask)
                    elif salience_weights is not None:
                        # Stage 2: use salience transfer weights from Stage 1
                        masked, loss_mask = make_freq_informed_mlm_mask(
                            batch,
                            mask_id=mask_id,
                            probability=mask_prob,
                            token_log_freq=salience_weights,
                            alpha=args.freq_alpha if args.freq_alpha > 0 else 0.5,
                            exclude={EOS_ID, mask_id},
                        )
                    elif args.freq_alpha > 0:
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
                            batch,
                            mask_id=mask_id,
                            probability=mask_prob,
                            exclude={EOS_ID, mask_id},
                        )
                    _, aux = model(
                        masked, objective=HybridObjective.MLM, labels=batch, mlm_mask=loss_mask
                    )
                    loss = aux["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for o in opts:
                o.step()
            v = float(loss.detach())
            last = v
            best = min(best, v)
            ema = v if ema is None else 0.98 * ema + 0.02 * v
            step_tokens = args.batch_size * cur_seq
            tokens_this_stage += step_tokens
            words_done = (
                words_offset + tokens_this_stage / 1.376
            )  # empirical tok/word for strict-small

            # BabyLM 2026 milestone checkpoints
            if ckpt_interval_tokens > 0 and tokens_this_stage >= next_ckpt_tokens:
                milestone_M = int(words_done / 1_000_000)
                ckpt_dev = out_dir / f"elc_{milestone_M}M.pt"
                save_elc_checkpoint(
                    ckpt_dev,
                    model,
                    mask_id=mask_id,
                    extra={
                        "track": "leaderboard_submission",
                        "checkpoint_words": int(words_done),
                        "step": gstep,
                        "loss": v,
                    },
                )
                print(
                    f"  [CKPT] {ckpt_dev.name} @ {words_done / 1e6:.2f}M words (step {gstep + 1})",
                    flush=True,
                )
                next_ckpt_tokens += ckpt_interval_tokens
            while (
                milestone_idx < len(pending_milestones)
                and words_done >= pending_milestones[milestone_idx]
            ):
                m = pending_milestones[milestone_idx]
                ckpt_dev = out_dir / f"elc_{m // 1_000_000}M.pt"
                if not ckpt_dev.exists():
                    save_elc_checkpoint(
                        ckpt_dev,
                        model,
                        mask_id=mask_id,
                        extra={
                            "track": "leaderboard_submission",
                            "checkpoint_words": m,
                            "step": gstep,
                            "loss": v,
                        },
                    )
                    print(
                        f"  [CKPT-BabyLM] {ckpt_dev.name} @ {words_done / 1e6:.2f}M words",
                        flush=True,
                    )
                milestone_idx += 1

            if (local + 1) % 200 == 0:
                rate = (local + 1) / max(time.time() - t0, 1e-6)
                print(
                    f"  step {gstep + 1}/{total_steps} loss={v:.4f} ema={ema:.4f} "
                    f"lr={lr:.2e} seq={cur_seq} maskp={mask_prob:.3f} {rate:.2f} step/s",
                    flush=True,
                )
        return last, best, tokens_this_stage / 1.376

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    salience_path = out_dir / "salience_weights.npy"

    ckpt_interval_tokens = (
        int(args.checkpoint_interval_words * 1.376) if args.checkpoint_interval_words > 0 else 0
    )
    milestones = _BABYLM_MILESTONES if args.babylm_checkpoints else ()
    if args.babylm_checkpoints:
        print("BabyLM checkpoints: ON (milestones at 1M–10M every 1M, then every 10M)", flush=True)

    t0 = time.time()
    _, b1, words_s1 = run_stage(
        dose_lines,
        stage1_steps,
        0,
        ckpt_interval_tokens=ckpt_interval_tokens,
        ckpt_milestones_words=milestones,
        words_offset=0.0,
    )
    # After Stage 1: save salience weights for transfer to Stage 2
    stage2_salience: torch.Tensor | None = None
    if salience_transfer is not None:
        salience_transfer.save(salience_path)

        w = np.log(np.clip(salience_transfer.salience_weights(), 1e-6, 1.0))
        stage2_salience = torch.tensor(w, dtype=torch.float32)
        print(f"Salience transfer: saved {salience_path} ({vocab} entries)", flush=True)

    last, b2, words_s2 = run_stage(
        base,
        stage2_steps,
        stage1_steps,
        salience_weights=stage2_salience,
        ckpt_interval_tokens=ckpt_interval_tokens,
        ckpt_milestones_words=milestones,
        words_offset=words_s1,
    )
    wall = time.time() - t0
    best = min(b1, b2)

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
            "nhot_embeddings": args.nhot_embeddings,
            "structured_masking": args.structured_masking,
            "freq_alpha": args.freq_alpha,
            "mask_start": args.mask_start,
            "mask_end": args.mask_end,
            "pos_encoding": args.pos_encoding,
            "max_seq_len": args.max_seq_len,
            "english_epochs": args.english_epochs,
            "final_loss": last,
            "best_loss": best,
        },
    )
    total_words = int(words_s1 + words_s2)
    print(
        f"DONE submission: steps={total_steps} words={total_words / 1e6:.1f}M "
        f"final_loss={last:.4f} best_loss={best:.4f} wall={wall / 60:.1f}min -> {ckpt}",
        flush=True,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "track": "leaderboard_submission",
                "dose_arms": args.dose_arms,
                "dose_epochs": args.dose_epochs,
                "english_epochs": args.english_epochs,
                "babylm_compliant": True,
                "total_words_processed": total_words,
                "steps": total_steps,
                "final_loss": last,
                "best_loss": best,
                "wall_seconds": wall,
                "checkpoint": str(ckpt),
                "pos_encoding": args.pos_encoding,
                "nhot_embeddings": args.nhot_embeddings,
                "structured_masking": args.structured_masking,
                "karaka_lookup": str(args.karaka_lookup) if args.karaka_lookup else "bpe_heuristic",
                "seed": args.seed,
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
