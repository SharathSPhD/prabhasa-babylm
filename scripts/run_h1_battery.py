"""H1 battery on the GB10: arms A-G x N seeds, real data, statistical go/no-go.

This is the scientific entrypoint (the proxy in ``run_h1_real_proxy.py`` is only
a CPU wiring check). It drives the *real* pipeline at the 100-150M scale on the
Blackwell GPU:

  * one shared SentencePiece tokenizer (SCAN-train + Saṃsādhanī cache + Dyck) so
    every arm is scored identically (fairness invariant),
  * each arm pre-pretrains on its structural prior (Pāṇinian / Dyck / none),
    continues on the SCAN-as-LM task ("IN: cmd OUT: actions"), and is scored by
    exact-match on the held-out SCAN split,
  * the orchestrator computes the decisive B-vs-C go/no-go with bootstrap CIs,
    a permutation test, and Holm-Bonferroni correction.

Generation is auto-sized to the longest gold target (see ``H1Runner``), so a
long action sequence can never be silently truncated to a forced-zero score.

Run on the CUDA venv (``.venv-gpu``) with ``--require-cuda``. Use ``--confirm``
for a fast, few-arm single-seed pass that verifies the eval produces non-zero
signal before committing the full multi-hour battery.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from psalm.application.data.assembly import PrePretrainAssembler
from psalm.application.data.tokenizer import TokenizerSpec
from psalm.application.experiments.orchestrator import H1Orchestrator
from psalm.domain.experiments.matrix import default_h1_matrix
from psalm.domain.model.config import preset_for
from psalm.domain.model.training import Precision, TrainConfig
from psalm.infrastructure.eval.scan import load_scan
from psalm.infrastructure.generators.dyck_source import DyckSentenceSource
from psalm.infrastructure.generators.jsonl_source import JsonlSentenceSource
from psalm.infrastructure.ledger.sqlite_ledger import SqliteLedger
from psalm.infrastructure.ml.h1_runner import EvalSets, H1Runner
from psalm.infrastructure.tokenizer.sentencepiece_tokenizer import SentencePieceTrainer

EOS_ID = 2  # SentencePiece default eos


def scan_line(cmd: str, act: str) -> str:
    return f"IN: {cmd} OUT: {act}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache/samsadhani.jsonl")
    ap.add_argument("--size", type=float, default=100.0, choices=[60.0, 100.0, 150.0])
    ap.add_argument("--scan-split", default="length")
    ap.add_argument("--scan-train", type=int, default=12000)
    ap.add_argument("--scan-test", type=int, default=300)
    ap.add_argument("--vocab", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument(
        "--confirm",
        action="store_true",
        help="Fast signal check: arms A,B,C,D only, 1 seed (overrides --seeds).",
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

    seeds = tuple(range(1 if args.confirm else args.seeds))

    paninian = JsonlSentenceSource(args.cache)
    dyck = DyckSentenceSource()
    scan_train = load_scan(args.scan_split, which="train", limit=args.scan_train)
    scan_test = load_scan(args.scan_split, which="test", limit=args.scan_test)
    scan_train_lines = [scan_line(c, a) for c, a in scan_train]

    pan_sample = [s.text for s in paninian.stream(4000, seed=0)]
    dyck_sample = [s.text for s in dyck.stream(1500, seed=0)]
    tok_corpus = scan_train_lines + pan_sample + dyck_sample

    tok_dir = Path(".cache/battery_tok")
    tok_dir.parent.mkdir(parents=True, exist_ok=True)
    spec = TokenizerSpec(vocab_size=args.vocab, model_type="unigram", sandhi_aware=True)
    tok = SentencePieceTrainer().train(tok_corpus, spec, tok_dir)
    vocab = tok.vocab_size
    print(f"tokenizer vocab={vocab} device={device} split={args.scan_split} size={args.size}M")

    assembler = PrePretrainAssembler(paninian=paninian, dyck=dyck)
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
    eval_sets = EvalSets(
        compositional=[(f"IN: {c} OUT:", a) for c, a in scan_test],
        minimal_pairs=[],
    )
    runner = H1Runner(
        assembler=assembler,
        nl_lines=lambda: scan_train_lines,
        encode=tok.encode,
        eos_id=EOS_ID,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        eval_sets=eval_sets,
        decode=lambda ids: tok.decode(ids),
        append_eos_to_prompt=False,
    )

    # Keep the full A-G matrix so the fairness invariants hold (the orchestrator
    # refuses an unfair/partial matrix). --confirm only shrinks seeds + steps for
    # a fast non-zero-signal check; the science uses the full --seeds run.
    matrix = default_h1_matrix(param_count_m=args.size, seeds=seeds)
    fairness = matrix.verify_fairness()
    if fairness:
        raise SystemExit(f"fairness violation(s): {fairness}")

    out_path = Path(
        args.out or f"docs/data/phase2-h1-battery-{args.scan_split}-{int(args.size)}m.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = SqliteLedger(out_path.with_suffix(".db"))
    orch = H1Orchestrator(matrix=matrix, ledger=ledger, runner=runner)

    start = time.time()
    results = orch.run()
    decision = orch.decide()
    wall = time.time() - start

    def accs(arm: str) -> list[float]:
        return [round(r.metric("compositional_accuracy").value, 4) for r in results.get(arm, [])]

    payload = {
        "split": args.scan_split,
        "size_m": args.size,
        "vocab": vocab,
        "seeds": list(seeds),
        "max_steps": args.max_steps,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "confirm": args.confirm,
        "wall_seconds": round(wall, 1),
        "accuracy": {arm_id: accs(arm_id) for arm_id in [a.arm_id for a in matrix.arms]},
        "decision": {
            "go": decision.go,
            "finding": decision.finding,
            "token_savings": decision.token_savings,
            "compositional_gain_points": decision.compositional_gain_points,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"runs: {sum(len(v) for v in results.values())} | wall {wall:.0f}s")
    for arm_id in [a.arm_id for a in matrix.arms]:
        print(f"{arm_id}: {accs(arm_id)}")
    print(f"token_savings      : {decision.token_savings:.3f}")
    print(f"compositional_gain : {decision.compositional_gain_points:.2f} pts")
    print(f"go / finding       : {decision.go} / {decision.finding}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
