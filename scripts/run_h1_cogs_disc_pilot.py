"""H1 COGS **discrimination** re-pilot (ADR-0015) — off-floor role readout.

Full-LF exact-match floors at proxy scale (SCAN gen 0%; COGS lexical EM ~0.02–0.05
even at 150M / 12k steps — see ``docs/data/phase2-h1-cogs-floorlift-150m`` finding).
ADR-0015 (pre-registered BLIND, committed before this script reveals any B-vs-C
discrimination number) switches the readout from *generating* the logical form to
*discriminating* the correct LF from a minimally **role-corrupted** one (main-verb
agent↔theme swap). The model passes an item when it assigns higher teacher-forced
likelihood to the correct LF (no generation; chance = 50%; off-floor by construction).

Same task (COGS train), same matched-epoch dose (ADR-0014) — only the readout
changes. Verdict follows ADR-0015 D3. Does NOT auto-launch the battery.

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
from psalm.domain.eval.discrimination import CORRUPTIONS
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

# --- Pre-registered constants (ADR-0015 D3). DO NOT tune after seeing results.
LEARNED_MIN = 0.60  # baseline A must clear this (>0.50 chance) or model learned nothing
DELTA_MIN = 0.03  # required B-minus-C discrimination gap
CI_HALF_WIDTH_GATE = 0.05  # per-arm precision sanity at 3 seeds


def cogs_line(sent: str, lf: str) -> str:
    return f"IN: {sent} OUT: {lf}"


def _ci(samples: list[float]) -> tuple[float, float, float, float]:
    mean, lo, hi = mean_ci(samples)
    return mean, lo, hi, (hi - lo) / 2.0


def build_disc_pairs(
    pairs: list[tuple[str, str]], corruption: str = "swap"
) -> list[tuple[str, str]]:
    """(sentence, gold_lf) -> (correct_full, corrupted_full) minimal pairs."""
    fn = CORRUPTIONS[corruption]
    out: list[tuple[str, str]] = []
    for sent, lf in pairs:
        corrupt = fn(lf)
        if corrupt is not None:
            out.append((cogs_line(sent, lf), cogs_line(sent, corrupt)))
    return out


#: Non-canonical-order lexical categories: passives + dative alternations, where
#: surface word order does NOT trivially cue agent/theme, so an agent↔theme swap
#: cannot be detected from position alone (harder than canonical SVO items).
NONCANONICAL_CATEGORIES = frozenset(
    {"active_to_passive", "passive_to_active", "do_dative_to_pp_dative", "pp_dative_to_do_dative"}
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache/samsadhani_75k.jsonl")
    ap.add_argument("--size", type=float, default=100.0, choices=[60.0, 100.0, 150.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cogs-train", type=int, default=12000)
    ap.add_argument("--lexical-test", type=int, default=400)
    ap.add_argument("--corruption", default="swap", choices=sorted(CORRUPTIONS))
    ap.add_argument(
        "--noncanonical",
        action="store_true",
        help="restrict the discrimination set to passive/dative items (surface order misleads)",
    )
    ap.add_argument("--vocab", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--pre-epochs", type=int, default=4)  # ADR-0014 matched-epoch dose
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
        raise SystemExit(
            f"--require-cuda set but CUDA unreachable (is_available={cuda_ok}, device={device!r})."
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
        f"pre_budget={pre_budget} x{args.pre_epochs} epochs  size={args.size}M  "
        f"seeds={args.seeds}  dose~{dose_frac:.2%}"
    )

    paninian = JsonlSentenceSource(args.cache)
    dyck = DyckSentenceSource()
    scrambled = ScramblingGenerator(paninian)

    train_pairs = load_cogs("train", limit=args.cogs_train)
    if args.noncanonical:
        lex_test = load_cogs("gen", categories=NONCANONICAL_CATEGORIES, limit=args.lexical_test)
    else:
        lex_test = load_cogs("gen", tier="lexical", limit=args.lexical_test)
    disc_pairs = build_disc_pairs(lex_test, args.corruption)
    coverage = len(disc_pairs) / len(lex_test) if lex_test else 0.0
    print(f"corruption={args.corruption} noncanonical={args.noncanonical}")
    train_lines = [cogs_line(s, lf) for s, lf in train_pairs]
    pan_sample = [s.text for s in paninian.stream(4000, seed=0)]
    dyck_sample = [s.text for s in dyck.stream(1500, seed=0)]
    print(
        f"COGS train={len(train_lines)} lexical_test={len(lex_test)} "
        f"disc_pairs={len(disc_pairs)} (coverage {coverage:.0%})"
    )

    tok_dir = Path(".cache/cogs_disc_tok")
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
        # Discrimination needs no generation; keep the compositional set empty so
        # no costly greedy decoding runs. The readout is disc_eval_sets only.
        eval_sets=EvalSets(compositional=[], minimal_pairs=[]),
        disc_eval_sets={"lexical": disc_pairs},
        decode=lambda ids: tok.decode(ids),
        append_eos_to_prompt=False,
        pre_budget_tokens=pre_budget,
        pre_epochs=args.pre_epochs,
        eval_fracs=(),
    )

    matrix = default_h1_matrix(param_count_m=args.size)
    arm_ids = [a.strip() for a in args.arms.split(",") if a.strip()]

    start = time.time()
    disc: dict[str, list[float]] = {a: [] for a in arm_ids}
    for arm_id in arm_ids:
        arm = matrix.arm(arm_id)
        for seed in range(args.seeds):
            m = runner(arm, seed)
            disc[arm_id].append(round(m.get("lexical_disc", 0.0), 4))
            print(
                f"arm {arm_id} seed {seed}: disc={disc[arm_id][-1]} (n={int(m.get('lexical_disc_n', 0))})"
            )
    wall = time.time() - start

    summary: dict[str, dict[str, float | list[float]]] = {}
    for a in arm_ids:
        mean, lo, hi, hw = _ci(disc[a])
        summary[a] = {"mean": mean, "ci": [lo, hi], "half_width": hw}

    # --- Pre-registered verdict (ADR-0015 D3).
    verdict = "INCOMPLETE: need arms A, B, C to evaluate the pre-registered criterion."
    delta = delta_lo = delta_hi = float("nan")
    if all(a in disc for a in ("A", "B", "C")):
        a_mean = float(summary["A"]["mean"])
        diffs = [disc["B"][i] - disc["C"][i] for i in range(args.seeds)]
        delta, delta_lo, delta_hi = mean_ci(diffs)
        max_hw = max(float(summary[a]["half_width"]) for a in ("A", "B", "C"))
        if a_mean < LEARNED_MIN:
            verdict = (
                f"NOT_LEARNED_AT_SCALE: baseline A discrimination {a_mean:.3f} < {LEARNED_MIN} "
                "(≈chance). Model learned no discriminable role structure at this scale; "
                "escalate to Phase-3 scale (Option C). Do not reinterpret."
            )
        elif delta >= DELTA_MIN and delta_lo > 0 and max_hw < CI_HALF_WIDTH_GATE:
            verdict = (
                f"LAUNCH_RECOMMENDED: A learned ({a_mean:.3f}), B-C={delta:+.3f} "
                f"(CI {delta_lo:+.3f}..{delta_hi:+.3f}), max half-width {max_hw:.3f}. "
                "Recommendation only — human sign-off required."
            )
        else:
            verdict = (
                f"NO_OR_WEAK_SUPPORT: A learned ({a_mean:.3f}) but B-C={delta:+.3f} "
                f"(CI {delta_lo:+.3f}..{delta_hi:+.3f}), max half-width {max_hw:.3f}. "
                "Pāṇinian prior gives no role-discrimination lift over Dyck at this dose; "
                "consider crystallization dose before deciding."
            )

    payload = {
        "task": "cogs_discrimination",
        "adr": "0015",
        "corruption": args.corruption,
        "noncanonical": args.noncanonical,
        "size_m": args.size,
        "seeds": args.seeds,
        "max_steps": args.max_steps,
        "pre_epochs": args.pre_epochs,
        "pre_budget_tokens": pre_budget,
        "dose_fraction": round(dose_frac, 4),
        "disc_coverage": round(coverage, 3),
        "disc_pairs": len(disc_pairs),
        "vocab": vocab,
        "wall_seconds": round(wall, 1),
        "prereg": {
            "learned_min": LEARNED_MIN,
            "delta_min": DELTA_MIN,
            "ci_hw_gate": CI_HALF_WIDTH_GATE,
        },
        "discrimination": disc,
        "summary": summary,
        "delta_B_minus_C": {"mean": delta, "ci": [delta_lo, delta_hi]},
        "verdict": verdict,
    }
    out_path = Path(args.out or f"docs/data/phase2-h1-cogs-disc-{int(args.size)}m.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print("\n=== COGS DISCRIMINATION PILOT (vs pre-registered ADR-0015 bar) ===")
    for a in arm_ids:
        s = summary[a]
        print(
            f"{a} disc: {s['mean']:.3f} (CI {s['ci'][0]:.3f}-{s['ci'][1]:.3f}, ±{s['half_width']:.3f})"
        )
    print(f"B-C (disc): {delta:+.3f} (CI {delta_lo:+.3f}..{delta_hi:+.3f})")
    print(f"wall: {wall:.0f}s")
    print(f"VERDICT: {verdict}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
