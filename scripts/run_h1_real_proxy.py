"""Real-data H1 proxy: Saṃsādhanī/Dyck pre-pretrain -> SCAN task -> exact-match.

Unlike ``run_h1_proxy.py`` (toy char model, fake task), this drives the *real*
pipeline at proxy scale:

  * shared SentencePiece tokenizer trained on the cached Saṃsādhanī corpus +
    SCAN-train + a Dyck sample (fair: identical tokenizer for every arm),
  * pre-pretraining from the cached Pāṇinian sentences (arms B/D) or Dyck (C),
  * the SCAN length-split as the downstream compositional task
    (``IN: cmd OUT: actions``), evaluated by exact-match on the test split,
  * the orchestrator's B-vs-C go/no-go with the statistical gate.

It is still a *proxy* (small model, capped data, CPU-friendly) — a wiring and
sanity check, not the scientific finding. The finding comes from the 100–150M
battery on the GB10. Requires the Saṃsādhanī cache (scripts/cache_samsadhani.py)
and SCAN (auto-downloaded by the loader).
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from psalm.application.data.assembly import PrePretrainAssembler
from psalm.application.data.tokenizer import TokenizerSpec
from psalm.application.experiments.orchestrator import H1Orchestrator
from psalm.domain.experiments.matrix import default_h1_matrix
from psalm.domain.model.config import ModelConfig
from psalm.domain.model.training import Precision, TrainConfig
from psalm.infrastructure.eval.scan import load_scan
from psalm.infrastructure.generators.dyck_source import DyckSentenceSource
from psalm.infrastructure.generators.jsonl_source import JsonlSentenceSource
from psalm.infrastructure.generators.scramble_source import ScramblingGenerator
from psalm.infrastructure.ledger.sqlite_ledger import SqliteLedger
from psalm.infrastructure.ml.h1_runner import EvalSets, H1Runner
from psalm.infrastructure.tokenizer.sentencepiece_tokenizer import SentencePieceTrainer

EOS_ID = 2  # SentencePiece default eos


def scan_line(cmd: str, act: str) -> str:
    return f"IN: {cmd} OUT: {act}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache/samsadhani.jsonl")
    ap.add_argument("--scan-split", default="length")
    ap.add_argument("--scan-train", type=int, default=2000)
    ap.add_argument("--scan-test", type=int, default=200)
    ap.add_argument("--vocab", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--require-cuda",
        action="store_true",
        help="Assert a CUDA device is reachable and fail loud otherwise "
        "(prevents a 'GPU' run from silently degrading to CPU).",
    )
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        device = args.device or ("cuda" if cuda_ok else "cpu")
    except Exception:  # noqa: BLE001
        cuda_ok = False
        device = args.device or "cpu"

    if args.require_cuda:
        if not cuda_ok or not device.startswith("cuda"):
            raise SystemExit(
                f"--require-cuda set but CUDA is not reachable "
                f"(torch.cuda.is_available()={cuda_ok}, device={device!r}). "
                f"Use the CUDA torch venv (.venv-gpu) on the GB10."
            )
        import torch

        print(f"CUDA OK: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")

    paninian = JsonlSentenceSource(args.cache)
    dyck = DyckSentenceSource()

    scan_train = load_scan(args.scan_split, which="train", limit=args.scan_train)
    scan_test = load_scan(args.scan_split, which="test", limit=args.scan_test)
    scan_train_lines = [scan_line(c, a) for c, a in scan_train]

    # Shared tokenizer corpus: SCAN-train + Pāṇinian cache sample + Dyck sample.
    pan_sample = [s.text for s in paninian.stream(3000, seed=0)]
    dyck_sample = [s.text for s in dyck.stream(1000, seed=0)]
    tok_corpus = scan_train_lines + pan_sample + dyck_sample

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        spec = TokenizerSpec(vocab_size=args.vocab, model_type="unigram", sandhi_aware=True)
        tok = SentencePieceTrainer().train(tok_corpus, spec, tmp_path / "tok")
        vocab = tok.vocab_size
        print(f"tokenizer vocab={vocab} device={device}")

        assembler = PrePretrainAssembler(
            paninian=paninian, dyck=dyck, paninian_scrambled=ScramblingGenerator(paninian)
        )
        model_cfg = ModelConfig(
            vocab_size=vocab, d_model=128, n_layers=4, n_heads=4, max_seq_len=96
        )
        train_cfg = TrainConfig(
            max_steps=args.max_steps,
            batch_size=16,
            seq_len=64,
            lr=3e-3,
            warmup_steps=10,
            precision=Precision.FP32 if device == "cpu" else Precision.BF16,
            device=device,
            log_every=40,
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
            pretrain_max_new=48,
            pre_budget_tokens=4000,  # tiny structural budget for the CPU wiring check
        )
        matrix = default_h1_matrix(param_count_m=60.0, seeds=seeds)
        ledger = SqliteLedger(tmp_path / "ledger.db")
        orch = H1Orchestrator(matrix=matrix, ledger=ledger, runner=runner)
        results = orch.run()
        decision = orch.decide()

    def accs(arm: str) -> list[float]:
        return [round(r.metric("compositional_accuracy").value, 3) for r in results[arm]]

    print(f"runs: {sum(len(v) for v in results.values())} (7 arms x {len(seeds)} seeds)")
    print(f"A (no prior)        : {accs('A')}")
    print(f"B (Pāṇinian)        : {accs('B')}")
    print(f"C (Dyck control)    : {accs('C')}")
    print(f"D (Pāṇinian+kāraka) : {accs('D')}")
    print(f"token_savings       : {decision.token_savings:.3f}")
    print(f"compositional_gain  : {decision.compositional_gain_points:.2f} pts")
    print(f"go / finding        : {decision.go} / {decision.finding}")
    print("NOTE: proxy wiring/sanity pass — not the scientific finding.")


if __name__ == "__main__":
    main()
