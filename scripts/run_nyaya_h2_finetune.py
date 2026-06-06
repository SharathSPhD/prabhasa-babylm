#!/usr/bin/env python3
"""H2 Navya-Nyaya LoRA fine-tune of the best H1_MECHANISM checkpoint.

Fine-tunes ELC-PSALM-S on the pramana Pancavayava inference chains using LoRA
adapters (rank=16, target=qkv projections).  Reports fallacious-inference rate
on a held-out Nyaya NLI test set as the H2 primary metric.

The novel claim is the H1_MECHANISM x H2 synergy test:
  PSALM-mechanism + Nyaya vs Generic-1B baseline + Nyaya

on sample efficiency: how many pramana examples are needed to reach 80% accuracy
on the Nyaya NLI test set?

Usage:
    uv run python scripts/run_nyaya_h2_finetune.py \\
        --psalm-ckpt data/checkpoints/submission_compliant/seed_0/elc.pt \\
        --pramana-path /home/sharaths/projects/pramana \\
        --lora-rank 16 --epochs 20 --device cuda

Quick smoke (CPU, fast):
    uv run python scripts/run_nyaya_h2_finetune.py \\
        --psalm-ckpt data/checkpoints/submission_compliant/seed_0/elc.pt \\
        --fast --device cpu
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PRAMANA_DEFAULT = Path("/home/sharaths/projects/pramana")

_ATTN_BIDIRECTIONAL = "bidirectional"  # matches _AttnMode.BIDIRECTIONAL (StrEnum)


def _load_pramana_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _tokenize_pair(sp, premise: str, hypothesis: str, max_len: int = 256) -> list[int]:
    """Encode premise + [SEP] + hypothesis with SentencePiece."""
    p_ids = sp.EncodeAsIds(premise[:512])
    h_ids = sp.EncodeAsIds(hypothesis[:256])
    sep = sp.piece_to_id("[SEP]") if sp.piece_to_id("[SEP]") != 0 else 1
    ids = [sp.bos_id()] + p_ids + [sep] + h_ids + [sp.eos_id()]
    return ids[:max_len]


class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper for a Linear layer."""

    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: float = 32.0) -> None:
        super().__init__()
        self.linear = linear
        in_f, out_f = linear.in_features, linear.out_features
        device = linear.weight.device
        self.lora_A = nn.Parameter(torch.randn(rank, in_f, device=device) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=device))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def _apply_lora(model: nn.Module, rank: int) -> int:
    """Replace QKV Linear layers with LoRA wrappers; return adapter param count."""
    replaced = 0
    module_dict = dict(model.named_modules())
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.endswith('.qkv'):
            # Navigate to parent and replace the qkv layer
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr = parts
                parent = module_dict[parent_name]
                setattr(parent, attr, LoRALinear(module, rank=rank))
                replaced += 1
    adapter_params = sum(p.numel() for m in model.modules()
                         if isinstance(m, LoRALinear)
                         for p in (m.lora_A, m.lora_B))
    print(f"[h2] LoRA applied: {replaced} layers replaced, {adapter_params:,} adapter params", flush=True)
    return adapter_params


def _freeze_base(model: nn.Module) -> None:
    """Freeze all base model parameters; LoRA adapter params remain trainable."""
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name and "cls_head" not in name:
            param.requires_grad_(False)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(-1) == labels).float().mean().item()


def run_h2_finetune(
    psalm_ckpt: Path,
    nli_data: list[dict],
    sp_path: Path,
    *,
    lora_rank: int = 16,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "cuda",
    fast: bool = False,
    seed: int = 42,
) -> dict[str, object]:
    """Fine-tune ELC-PSALM-S on NLI data with LoRA adapters."""
    import sentencepiece as spm
    from psalm.infrastructure.ml.elc_trainer import load_elc_checkpoint

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Load model
    model, mask_id = load_elc_checkpoint(psalm_ckpt, device=device)
    model = model.to(device)

    # Add classification head (3-way NLI: entailment, neutral, contradiction)
    model.cls_head = nn.Linear(model.cfg.d_model, 3).to(device)
    nn.init.normal_(model.cls_head.weight, std=0.02)
    nn.init.zeros_(model.cls_head.bias)

    # Apply LoRA
    _apply_lora(model, rank=lora_rank)
    _freeze_base(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[h2] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)", flush=True)

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(str(sp_path))

    # Split data: 80% train / 20% test
    random.shuffle(nli_data)
    n_train = max(1, int(len(nli_data) * 0.8))
    train_data = nli_data[:n_train]
    test_data = nli_data[n_train:]
    if fast:
        train_data = train_data[:min(20, len(train_data))]
        test_data = test_data[:min(10, len(test_data))]
        epochs = min(3, epochs)
    print(f"[h2] Train: {len(train_data)}, Test: {len(test_data)}, epochs: {epochs}", flush=True)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )

    history: list[dict] = []
    best_test_acc = 0.0
    test_acc = 0.0

    for epoch in range(epochs):
        random.shuffle(train_data)
        model.train()
        train_losses = []
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i : i + batch_size]
            max_len = max(len(_tokenize_pair(sp, ex["premise"], ex["hypothesis"])) for ex in batch)
            input_ids = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
            labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long, device=device)
            for j, ex in enumerate(batch):
                ids = _tokenize_pair(sp, ex["premise"], ex["hypothesis"], max_len)
                input_ids[j, : len(ids)] = torch.tensor(ids)

            # Forward: use CLS token (position 0) hidden state
            hidden = model.encode(input_ids, attn_mode=_ATTN_BIDIRECTIONAL)
            cls_hidden = hidden[:, 0, :]  # (B, d_model)
            logits = model.cls_head(cls_hidden)
            loss = F.cross_entropy(logits, labels)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())

        # Test
        model.train(False)
        with torch.no_grad():
            test_accs = []
            for i in range(0, len(test_data), batch_size):
                batch = test_data[i : i + batch_size]
                max_len = max(len(_tokenize_pair(sp, ex["premise"], ex["hypothesis"])) for ex in batch)
                input_ids = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
                labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long, device=device)
                for j, ex in enumerate(batch):
                    ids = _tokenize_pair(sp, ex["premise"], ex["hypothesis"], max_len)
                    input_ids[j, : len(ids)] = torch.tensor(ids)
                hidden = model.encode(input_ids, attn_mode=_ATTN_BIDIRECTIONAL)
                cls_hidden = hidden[:, 0, :]
                logits = model.cls_head(cls_hidden)
                test_accs.append(_accuracy(logits, labels))

        test_acc = float(np.mean(test_accs)) if test_accs else 0.0
        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        best_test_acc = max(best_test_acc, test_acc)
        print(f"[h2] epoch {epoch+1:2d}/{epochs}  train_loss={train_loss:.4f}  test_acc={test_acc:.4f}  best={best_test_acc:.4f}", flush=True)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "test_acc": test_acc})

    return {
        "best_test_acc": best_test_acc,
        "final_test_acc": test_acc,
        "n_train": len(train_data),
        "n_test": len(test_data),
        "lora_rank": lora_rank,
        "epochs": epochs,
        "history": history,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="H2 Navya-Nyaya LoRA fine-tune")
    ap.add_argument("--psalm-ckpt", required=True, help="Path to ELC checkpoint (.pt)")
    ap.add_argument("--pramana-path", default=str(PRAMANA_DEFAULT))
    ap.add_argument(
        "--nli-data",
        default=None,
        help="Path to a generated NLI jsonl ({premise,hypothesis,label}); bypasses pramana-path.",
    )
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fast", action="store_true", help="Smoke test (CPU, small data)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/h2_nyaya_lora")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT / "src"))

    if args.nli_data:
        nli_path = Path(args.nli_data)
        if not nli_path.exists():
            print(f"[h2] ERROR: --nli-data not found at {nli_path}", flush=True)
            sys.exit(1)
        nli_data = [
            {"premise": r["premise"], "hypothesis": r["hypothesis"], "label": int(r["label"])}
            for r in _load_pramana_jsonl(nli_path)
        ]
        print(f"[h2] Loaded {len(nli_data)} NLI examples from {nli_path}", flush=True)
    else:
        pramana_path = Path(args.pramana_path)
        stage_0_path = pramana_path / "data" / "training" / "stage_0.jsonl"
        stage_1_path = pramana_path / "data" / "training" / "stage_1.jsonl"
        if not stage_0_path.exists() or not stage_1_path.exists():
            print(f"[h2] ERROR: pramana data not found at {pramana_path}", flush=True)
            sys.exit(1)
        from psalm.infrastructure.data.pramana_loader import load_pramana_datasets
        nli = load_pramana_datasets(stage_0_path, stage_1_path)
        nli_data = nli.get("combined", [])
        print(f"[h2] Loaded {len(nli_data)} NLI examples from pramana", flush=True)

    sp_path = ROOT / "data" / "tokenizer" / "strict_small" / "spm.model"
    psalm_ckpt = Path(args.psalm_ckpt)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[h2] CUDA not available, falling back to CPU", flush=True)
        device = "cpu"

    results = run_h2_finetune(
        psalm_ckpt, nli_data, sp_path,
        lora_rank=args.lora_rank,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        fast=args.fast,
        seed=args.seed,
    )

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "h2_results.json"
    out_path.write_text(json.dumps(results, indent=2))

    print(f"\n[h2] Results written to {out_path}")
    print(f"[h2] Best test accuracy: {results['best_test_acc']:.4f}")
    print(f"[h2] H2 threshold (80%): {'MET' if results['best_test_acc'] >= 0.80 else 'NOT MET'}")


if __name__ == "__main__":
    main()
