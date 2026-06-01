"""H1 COGS re-pilot — pre-registered floor/dose probe on a mechanism-aligned task.

Implements ADR-0014: after the SCAN dose-and-floor pilot showed SCAN's
generalization splits floor at 0% (so binary exact-match cannot host the H1
test), the primary compositional benchmark moves to **COGS**, with the decision
metric and B-vs-C threshold **pre-registered before this script reveals any
comparison** (the file ``docs/decisions/0014-cogs-primary-graded-metric-prereg.md``
is committed first; this script only *reports against* that fixed bar).

Per arm (A/B/C × N seeds), train on COGS ``train`` (in-distribution) and eval on:
  - **lexical-generalization tier** (primary metric Mprim = exact-match), and
  - **structural-generalization tier** (secondary, graded: EM + token-F1 +
    length-binned), reported but NOT gating (predicted to floor).

Dose: matched-epoch (``pre_epochs``) — the capped unique structural set repeated
N× for both B and C. Verdict follows ADR-0014 D3. It does **not** auto-launch the
battery; the human owns that sign-off.

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
from psalm.infrastructure.eval.cogs import load_cogs
from psalm.infrastructure.generators.dyck_source import DyckSentenceSource
from psalm.infrastructure.generators.jsonl_source import JsonlSentenceSource
from psalm.infrastructure.generators.scramble_source import ScramblingGenerator
from psalm.infrastructure.ml.h1_runner import DEFAULT_EVAL_FRACS, EvalSets, H1Runner
from psalm.infrastructure.tokenizer.sentencepiece_tokenizer import SentencePieceTrainer

EOS_ID = 2

# --- Pre-registered constants (ADR-0014 D3). DO NOT tune after seeing results.
FLOOR_MIN = 0.10  # baseline A lexical-tier EM must clear this or the task floors
DELTA_MIN = 0.03  # required B-minus-C lexical-tier EM gap
CI_HALF_WIDTH_GATE = 0.05  # per-arm precision sanity at 3 seeds


def cogs_line(sent: str, lf: str) -> str:
    return f"IN: {sent} OUT: {lf}"


def _ci(samples: list[float]) -> tuple[float, float, float, float]:
    mean, lo, hi = mean_ci(samples)
    return mean, lo, hi, (hi - lo) / 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache/samsadhani_75k.jsonl")
    ap.add_argument("--size", type=float, default=100.0, choices=[60.0, 100.0, 150.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cogs-train", type=int, default=12000)
    ap.add_argument("--lexical-test", type=int, default=200)
    ap.add_argument("--structural-test", type=int, default=100)
    ap.add_argument("--comp-max-new-cap", type=int, default=96)
    ap.add_argument("--vocab", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=3000)  # ADR-0014 D4
    ap.add_argument("--pre-epochs", type=int, default=4)  # ADR-0014 D4
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pre-budget", type=int, default=None)
    ap.add_argument("--arms", default="A,B,C")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--checkpoints", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch

    cuda_ok = torch.cuda.is_available()
    device = args.device if (not args.device.startswith("cuda") or cuda_ok) else "cpu"
    if args.require_cuda and (not cuda_ok or not device.startswith("cuda")):
        raise SystemExit(
            f"--require-cuda set but CUDA unreachable (is_available={cuda_ok}, "
            f"device={device!r}). Use .venv-gpu on the GB10."
        )
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
    dose_frac = (args.pre_epochs * pre_budget) / (args.max_steps * args.batch_size * args.seq_len)
    print(
        f"pre_budget={pre_budget} tokens x{args.pre_epochs} epochs  size={args.size}M  "
        f"seeds={args.seeds}  dose~{dose_frac:.2%} of downstream"
    )

    paninian = JsonlSentenceSource(args.cache)
    dyck = DyckSentenceSource()
    scrambled = ScramblingGenerator(paninian)

    train_pairs = load_cogs("train", limit=args.cogs_train)
    lex_test = load_cogs("gen", tier="lexical", limit=args.lexical_test)
    struct_test = load_cogs("gen", tier="structural", limit=args.structural_test)
    train_lines = [cogs_line(s, lf) for s, lf in train_pairs]
    pan_sample = [s.text for s in paninian.stream(4000, seed=0)]
    dyck_sample = [s.text for s in dyck.stream(1500, seed=0)]
    print(
        f"COGS train={len(train_lines)} lexical_test={len(lex_test)} struct_test={len(struct_test)}"
    )

    tok_dir = Path(".cache/cogs_pilot_tok")
    tok_dir.parent.mkdir(parents=True, exist_ok=True)
    spec = TokenizerSpec(vocab_size=args.vocab, model_type="unigram", sandhi_aware=True)
    tok = SentencePieceTrainer().train(train_lines + pan_sample + dyck_sample, spec, tok_dir)
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
        log_every=250,
    )

    runner = H1Runner(
        assembler=assembler,
        nl_lines=lambda: train_lines,
        encode=tok.encode,
        eos_id=EOS_ID,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        eval_sets=EvalSets(
            compositional=[(f"IN: {s} OUT:", lf) for s, lf in lex_test], minimal_pairs=[]
        ),
        extra_eval_sets={"structural": [(f"IN: {s} OUT:", lf) for s, lf in struct_test]},
        decode=lambda ids: tok.decode(ids),
        append_eos_to_prompt=False,
        pre_budget_tokens=pre_budget,
        pre_epochs=args.pre_epochs,
        eval_fracs=DEFAULT_EVAL_FRACS if args.checkpoints else (),
        comp_max_new_cap=args.comp_max_new_cap,
    )

    matrix = default_h1_matrix(param_count_m=args.size)
    arm_ids = [a.strip() for a in args.arms.split(",") if a.strip()]

    start = time.time()
    lex_em: dict[str, list[float]] = {a: [] for a in arm_ids}
    struct_em: dict[str, list[float]] = {a: [] for a in arm_ids}
    struct_f1: dict[str, list[float]] = {a: [] for a in arm_ids}
    for arm_id in arm_ids:
        arm = matrix.arm(arm_id)
        for seed in range(args.seeds):
            m = runner(arm, seed)
            lex_em[arm_id].append(round(m["compositional_accuracy"], 4))
            struct_em[arm_id].append(round(m.get("structural_em", 0.0), 4))
            struct_f1[arm_id].append(round(m.get("structural_f1", 0.0), 4))
            print(
                f"arm {arm_id} seed {seed}: lex_em={lex_em[arm_id][-1]} "
                f"struct_em={struct_em[arm_id][-1]} struct_f1={struct_f1[arm_id][-1]}"
            )
    wall = time.time() - start

    summary: dict[str, dict[str, float | list[float]]] = {}
    for a in arm_ids:
        mean, lo, hi, hw = _ci(lex_em[a])
        summary[a] = {"mean": mean, "ci": [lo, hi], "half_width": hw}

    # --- Pre-registered verdict (ADR-0014 D3). Only computed if both B and C ran.
    verdict = "INCOMPLETE: need arms A, B, C to evaluate the pre-registered criterion."
    delta = delta_lo = delta_hi = float("nan")
    if all(a in lex_em for a in ("A", "B", "C")):
        a_mean = summary["A"]["mean"]
        diffs = [lex_em["B"][i] - lex_em["C"][i] for i in range(args.seeds)]
        delta, delta_lo, delta_hi = mean_ci(diffs)
        max_hw = max(float(summary[a]["half_width"]) for a in ("A", "B", "C"))
        if a_mean < FLOOR_MIN:
            verdict = (
                f"STILL_FLOORED: baseline A lexical-tier EM {a_mean:.3f} < {FLOOR_MIN}. "
                "Instrument floors; escalate task/budget. DO NOT launch."
            )
        elif delta >= DELTA_MIN and delta_lo > 0 and max_hw < CI_HALF_WIDTH_GATE:
            verdict = (
                f"LAUNCH_RECOMMENDED: B-C={delta:+.3f} (CI {delta_lo:+.3f}..{delta_hi:+.3f}), "
                f"A off-floor ({a_mean:.3f}), max half-width {max_hw:.3f}. "
                "Recommendation only — human sign-off required (ADR-0014 D5)."
            )
        elif delta > 0:
            verdict = (
                f"UNDERPOWERED_OR_SMALL: B-C={delta:+.3f} (CI {delta_lo:+.3f}..{delta_hi:+.3f}), "
                f"max half-width {max_hw:.3f}. Effect <{DELTA_MIN} or CI crosses 0. "
                "Report; recommend generator-expansion dose before deciding."
            )
        else:
            verdict = (
                f"NO_SUPPORT_AT_DOSE: B-C={delta:+.3f}. Pāṇinian prior gives no lexical-tier "
                f"lift over Dyck at this matched-epoch dose (~{dose_frac:.2%})."
            )

    payload = {
        "task": "cogs",
        "adr": "0014",
        "size_m": args.size,
        "seeds": args.seeds,
        "max_steps": args.max_steps,
        "pre_epochs": args.pre_epochs,
        "pre_budget_tokens": pre_budget,
        "dose_fraction": round(dose_frac, 4),
        "vocab": vocab,
        "wall_seconds": round(wall, 1),
        "prereg": {
            "floor_min": FLOOR_MIN,
            "delta_min": DELTA_MIN,
            "ci_hw_gate": CI_HALF_WIDTH_GATE,
        },
        "lexical_em": lex_em,
        "structural_em": struct_em,
        "structural_f1": struct_f1,
        "summary_lexical": summary,
        "delta_B_minus_C": {"mean": delta, "ci": [delta_lo, delta_hi]},
        "verdict": verdict,
    }
    out_path = Path(args.out or f"docs/data/phase2-h1-cogs-pilot-{int(args.size)}m.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print("\n=== COGS PILOT SUMMARY (vs pre-registered ADR-0014 bar) ===")
    for a in arm_ids:
        s = summary[a]
        print(
            f"{a} lexical EM: {s['mean']:.3f} (CI {s['ci'][0]:.3f}-{s['ci'][1]:.3f}, ±{s['half_width']:.3f})"
        )
    print(f"B-C (lexical EM): {delta:+.3f} (CI {delta_lo:+.3f}..{delta_hi:+.3f})")
    print(f"wall: {wall:.0f}s")
    print(f"VERDICT: {verdict}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
