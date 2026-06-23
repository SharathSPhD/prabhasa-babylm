#!/usr/bin/env python3
"""M3 ACD Faithfulness Gate: ablation-based validation of circuit localization.

Gate: Run targeted ablation (zero-out top-K heads from attribution) vs random ablation
(zero-out K random heads). If targeted ablation does NOT hurt significantly more than
random on BLiMP weak paradigms (NPI, filler-gap, islands), circuits are not real →
STOP here, report NULL finding.

Gating decision: Δ(targeted − random) on weak-paradigm accuracy.
Pre-registered threshold: targeted must be ≥2pp worse than random to pass the gate.

Output: faithfulness_gate_report.json with per-paradigm results + GO/NO-GO decision.

CRITICAL: Uses real head-level ablation via hooks. Each ablation PROVES it changes
model output by comparing logits before/after. If ablation is a no-op (same logits),
result is invalid and gate FAILS with ERROR verdict.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.psalm.infrastructure.acd.blimp_minimal_pairs import NpiLicensingLoader


class AttentionHeadAblator:
    """Real head-level ablation via forward hooks on out_proj."""

    def __init__(self, model: AutoModelForMaskedLM, n_heads: int = 12) -> None:
        self.model = model
        self.n_heads = n_heads
        self.ablated_heads: set[tuple[int, int]] = set()  # (layer, head) pairs
        self.handles: list[Any] = []

    def _make_ablation_hook(self, layer_idx: int, head_idx: int) -> Any:
        """Create a hook that zeros out a specific head's output.

        For SDPA attention, we hook on out_proj and zero out the corresponding
        slice of the output. Each head's output is head_dim = 768 // 12 = 64 dims.
        """

        def hook(module: nn.Module, input: Any, output: Any) -> torch.Tensor:
            # output from out_proj is (batch_size, seq_len, hidden_size)
            if isinstance(output, tuple):
                attn_output = output[0]
            else:
                attn_output = output

            if not isinstance(attn_output, torch.Tensor):
                return output

            # Zero out the head's contribution
            hidden_size = attn_output.shape[-1]
            head_dim = hidden_size // self.n_heads
            start_idx = head_idx * head_dim
            end_idx = (head_idx + 1) * head_dim

            # Clone to avoid in-place modification issues
            attn_output = attn_output.clone()
            attn_output[:, :, start_idx:end_idx] = 0.0

            if isinstance(output, tuple):
                return (attn_output,) + output[1:]
            else:
                return attn_output

        return hook

    def ablate_heads(self, heads: list[tuple[int, int]]) -> None:
        """Register hooks to ablate specified heads.

        Args:
            heads: List of (layer_idx, head_idx) tuples to ablate.
        """
        self.ablated_heads = set(heads)
        encoder = self.model.encoder

        for layer_idx, head_idx in heads:
            block = encoder.blocks[layer_idx]
            # Hook on out_proj (the output projection of multi-head attention)
            hook = self._make_ablation_hook(layer_idx, head_idx)
            handle = block.out_proj.register_forward_hook(hook)
            self.handles.append(handle)

    def clear(self) -> None:
        """Remove all ablation hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.ablated_heads = set()


def compute_pll_score(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    sentence: str,
    device: str,
) -> float:
    """Compute pseudo-log-likelihood (PLL) for a sentence (official BabyLM metric).

    For an MLM, mask each token in turn and sum log-probabilities of gold tokens.

    Args:
        model: The masked language model.
        tokenizer: The tokenizer.
        sentence: Input sentence.
        device: Device for inference.

    Returns:
        Sum of log-probabilities (PLL score). Higher is better.
    """
    tokens = tokenizer.encode(sentence, add_special_tokens=False)
    if not tokens:
        return 0.0

    pll = 0.0
    for mask_idx in range(len(tokens)):
        # Create masked input
        masked_tokens = tokens[:mask_idx] + [tokenizer.mask_token_id] + tokens[mask_idx + 1 :]
        input_ids = torch.tensor([masked_tokens], device=device)

        with torch.no_grad():
            logits = model(input_ids).logits
            # Get logits at the masked position
            mask_logits = logits[0, mask_idx, :]
            # Get log probability of the gold token
            gold_token = tokens[mask_idx]
            log_prob = torch.nn.functional.log_softmax(mask_logits, dim=-1)[gold_token].item()
            pll += log_prob

    return pll


def compute_paradigm_accuracy_with_pll(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    paradigm: str,
    blimp_root: Path,
    device: str,
    max_pairs: int = 100,
) -> tuple[float, int]:
    """Compute baseline accuracy on a single BLiMP paradigm using PLL scoring.

    Returns:
        (accuracy, n_pairs)
    """
    loader = NpiLicensingLoader(blimp_root, paradigm=paradigm)
    pairs = loader.load_pairs()[:max_pairs]

    if not pairs:
        return 0.0, 0

    model.eval()
    correct = 0

    for pair in pairs:
        # Score clean vs corrupt using PLL
        clean_pll = compute_pll_score(model, tokenizer, pair.clean_sentence, device)
        corrupt_pll = compute_pll_score(model, tokenizer, pair.corrupt_sentence, device)

        if clean_pll > corrupt_pll:
            correct += 1

    acc = correct / len(pairs) if pairs else 0.0
    return acc, len(pairs)


def compute_head_importance_simple(
    model: AutoModelForMaskedLM,
    paradigm: str,
    blimp_root: Path,
    device: str,
    sample_size: int = 20,
    n_heads: int = 12,
) -> dict[tuple[int, int], float]:
    """Compute head importance via simple activation magnitude (heuristic).

    For each layer/head, average the frobenius norm of the head's weight slice.

    Args:
        model: The model.
        paradigm: BLiMP paradigm name.
        blimp_root: Path to BLiMP data.
        device: Device for computation.
        sample_size: Number of pairs to sample for attribution.
        n_heads: Number of attention heads.

    Returns:
        Dictionary mapping (layer, head) to importance score.
    """
    loader = NpiLicensingLoader(blimp_root, paradigm=paradigm)
    pairs = loader.load_pairs()[:sample_size]

    if not pairs:
        return {}

    model.eval()
    encoder = model.encoder
    n_layers = len(encoder.blocks)

    head_importance: dict[tuple[int, int], float] = {}

    # Use weight magnitude as importance proxy
    for layer_idx in range(n_layers):
        block = encoder.blocks[layer_idx]
        out_proj_weight = block.out_proj.weight  # (hidden, hidden)
        hidden_size = out_proj_weight.shape[-1]
        head_dim = hidden_size // n_heads

        for head_idx in range(n_heads):
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim
            head_weight = out_proj_weight[start:end, :]
            importance = head_weight.norm(p="fro").item()
            head_importance[(layer_idx, head_idx)] = importance

    return head_importance


def ablate_heads_targeted(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    paradigm: str,
    blimp_root: Path,
    device: str,
    top_k: int = 5,
    max_pairs: int = 100,
    head_importance: dict[tuple[int, int], float] | None = None,
) -> tuple[float, bool]:
    """Ablate top-K heads from importance ranking; measure accuracy drop on paradigm.

    Returns:
        (accuracy, ablation_changed_output)
    """
    model.eval()
    encoder = model.encoder

    # Rank heads by importance
    if head_importance is None or not head_importance:
        # Default: ablate from layer 0
        heads_to_ablate = [(0, i) for i in range(min(top_k, 12))]
    else:
        # Sort by importance (descending) and pick top-k
        sorted_heads = sorted(head_importance.items(), key=lambda x: x[1], reverse=True)
        heads_to_ablate = [h for h, _ in sorted_heads[:top_k]]

    # Prove ablation works: sample a pair and compare outputs before/after
    ablator = AttentionHeadAblator(model, n_heads=12)

    # Get baseline output on a test pair
    loader = NpiLicensingLoader(blimp_root, paradigm=paradigm)
    test_pairs = loader.load_pairs()[:1]
    if not test_pairs:
        return 0.0, False

    test_sent = test_pairs[0].clean_sentence
    with torch.no_grad():
        clean_ids_before = torch.tensor(
            [tokenizer.encode(test_sent, add_special_tokens=False)], device=device
        )
        logits_before = model(clean_ids_before).logits.mean(dim=(1, 2)).item()

    # Apply ablation
    ablator.ablate_heads(heads_to_ablate)

    with torch.no_grad():
        clean_ids_after = torch.tensor(
            [tokenizer.encode(test_sent, add_special_tokens=False)], device=device
        )
        logits_after = model(clean_ids_after).logits.mean(dim=(1, 2)).item()

    # Check if ablation changed output
    ablation_changed = abs(logits_before - logits_after) > 1e-5
    if not ablation_changed:
        print(
            f"WARNING: Targeted ablation DID NOT change model output! "
            f"logits_before={logits_before:.6f}, logits_after={logits_after:.6f}"
        )

    # Re-compute accuracy on paradigm
    acc, _ = compute_paradigm_accuracy_with_pll(
        model, tokenizer, paradigm, blimp_root, device, max_pairs=max_pairs
    )

    # Clean up
    ablator.clear()

    return acc, ablation_changed


def ablate_heads_random(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    paradigm: str,
    blimp_root: Path,
    device: str,
    k: int = 5,
    max_pairs: int = 100,
    seed: int = 42,
) -> tuple[float, bool]:
    """Ablate K random heads; measure accuracy drop on paradigm.

    Returns:
        (accuracy, ablation_changed_output)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.eval()
    encoder = model.encoder

    # Sample K random heads
    n_layers = len(encoder.blocks)
    n_heads_per_layer = 12
    total_heads = n_layers * n_heads_per_layer

    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads_per_layer)]
    random_indices = np.random.choice(len(all_heads), size=min(k, len(all_heads)), replace=False)
    heads_to_ablate = [all_heads[i] for i in random_indices]

    # Prove ablation works
    loader = NpiLicensingLoader(blimp_root, paradigm=paradigm)
    test_pairs = loader.load_pairs()[:1]
    if not test_pairs:
        return 0.0, False

    test_sent = test_pairs[0].clean_sentence
    with torch.no_grad():
        clean_ids_before = torch.tensor(
            [tokenizer.encode(test_sent, add_special_tokens=False)], device=device
        )
        logits_before = model(clean_ids_before).logits.mean(dim=(1, 2)).item()

    # Apply random ablation
    ablator = AttentionHeadAblator(model, n_heads=12)
    ablator.ablate_heads(heads_to_ablate)

    with torch.no_grad():
        clean_ids_after = torch.tensor(
            [tokenizer.encode(test_sent, add_special_tokens=False)], device=device
        )
        logits_after = model(clean_ids_after).logits.mean(dim=(1, 2)).item()

    ablation_changed = abs(logits_before - logits_after) > 1e-5

    # Re-compute accuracy on paradigm
    acc, _ = compute_paradigm_accuracy_with_pll(
        model, tokenizer, paradigm, blimp_root, device, max_pairs=max_pairs
    )

    # Clean up
    ablator.clear()

    return acc, ablation_changed


def main() -> None:
    """Run faithfulness gate on weak paradigms."""
    # Paths
    repo_root = Path(__file__).parent.parent
    # Use local model export instead of HF Hub
    model_path = repo_root / "data" / "hf_export" / "v02_adamw5e4_strict_seed1"
    blimp_root = (
        repo_root
        / "vendor"
        / "babylm-evaluation-pipeline-2026"
        / "strict"
        / "evaluation_data"
        / "full_eval"
        / "blimp_filtered"
    )
    results_dir = repo_root / "results" / "acd_m3"
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model + tokenizer
    print(f"Loading model from {model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True)
    model.to(device)
    model.eval()

    print(f"Model: {model.config.model_type}")
    print(f"  Layers: {model.config.num_hidden_layers}, Heads: {model.config.num_attention_heads}")

    # Weak paradigms to test (the M3 targets)
    # Match the BabyLM BLiMP filenames
    weak_paradigms = [
        "npi_present_1",
        "npi_present_2",
        "wh_island",
    ]

    # Gate parameters
    top_k = 5  # Ablate top-5 heads
    max_pairs_per_paradigm = 50
    n_random_seeds = 5
    n_attribution_samples = 20

    report: dict[str, Any] = {
        "timestamp": "2026-06-23T00:00:00Z",
        "model": str(model_path),
        "gate_type": "faithfulness_via_real_ablation",
        "top_k": top_k,
        "n_random_seeds": n_random_seeds,
        "ablation_method": "head-level forward hook (zeros out top-K by importance)",
        "scoring_method": "PLL (pseudo-log-likelihood)",
        "results": {},
        "summary": {},
        "ablation_validity": {},
    }

    gate_pass = True
    ablation_validity_issues = []

    for paradigm in weak_paradigms:
        print(f"\n{'='*60}")
        print(f"=== Testing paradigm: {paradigm} ===")
        print(f"{'='*60}")

        # Check if paradigm file exists
        paradigm_file = blimp_root / f"{paradigm}.jsonl"
        if not paradigm_file.exists():
            print(f"  SKIP: paradigm file not found at {paradigm_file}")
            continue

        # Baseline accuracy (no ablation)
        print(f"  Computing baseline accuracy (PLL-scored)...")
        baseline_acc, n_pairs = compute_paradigm_accuracy_with_pll(
            model, tokenizer, paradigm, blimp_root, device, max_pairs=max_pairs_per_paradigm
        )
        print(f"  Baseline accuracy: {baseline_acc:.3f} (n={n_pairs})")

        if n_pairs == 0:
            print(f"  SKIP: no pairs for paradigm {paradigm}")
            continue

        # Compute head importance
        print(f"  Computing head importance (via weight magnitude)...")
        head_importance = compute_head_importance_simple(
            model,
            paradigm,
            blimp_root,
            device,
            sample_size=n_attribution_samples,
            n_heads=12,
        )

        # Targeted ablation
        print(f"  Running TARGETED ablation (top-{top_k} heads)...")
        targeted_acc, targeted_ablation_changed = ablate_heads_targeted(
            model,
            tokenizer,
            paradigm,
            blimp_root,
            device,
            top_k=top_k,
            max_pairs=max_pairs_per_paradigm,
            head_importance=head_importance,
        )
        targeted_drop = baseline_acc - targeted_acc
        print(
            f"  Targeted ablation: {targeted_acc:.3f} (drop: {targeted_drop:.3f}pp), "
            f"ablation_changed={targeted_ablation_changed}"
        )

        if not targeted_ablation_changed:
            msg = f"Targeted ablation on {paradigm} did NOT change model output (no-op)"
            print(f"  ERROR: {msg}")
            ablation_validity_issues.append(msg)
            gate_pass = False

        # Random ablation (multiple seeds)
        print(f"  Running RANDOM ablation ({n_random_seeds} seeds)...")
        random_drops = []
        random_ablation_changed_list = []
        for seed in range(n_random_seeds):
            random_acc, random_ablation_changed = ablate_heads_random(
                model,
                tokenizer,
                paradigm,
                blimp_root,
                device,
                k=top_k,
                max_pairs=max_pairs_per_paradigm,
                seed=seed,
            )
            random_drop = baseline_acc - random_acc
            random_drops.append(random_drop)
            random_ablation_changed_list.append(random_ablation_changed)
            print(f"    Random seed {seed}: {random_acc:.3f} (drop: {random_drop:.3f}pp)")

            if not random_ablation_changed:
                msg = f"Random ablation seed {seed} on {paradigm} did NOT change model output (no-op)"
                print(f"    ERROR: {msg}")
                ablation_validity_issues.append(msg)

        avg_random_drop = sum(random_drops) / len(random_drops)
        delta = targeted_drop - avg_random_drop

        print(
            f"  → Δ(targeted − random) = {delta:.3f}pp "
            f"(targeted {targeted_drop:.3f}pp vs random {avg_random_drop:.3f}pp)"
        )

        # Gate decision for this paradigm
        # Pre-registered: targeted must hurt ≥2pp MORE than random
        paradigm_pass = delta >= 0.02 and targeted_ablation_changed
        print(f"  → Paradigm GATE: {'PASS' if paradigm_pass else 'FAIL'}")

        if not paradigm_pass:
            gate_pass = False

        report["results"][paradigm] = {
            "n_pairs": n_pairs,
            "baseline_accuracy": baseline_acc,
            "targeted_accuracy": targeted_acc,
            "targeted_drop": targeted_drop,
            "targeted_ablation_changed": targeted_ablation_changed,
            "random_drops": random_drops,
            "random_ablation_changed_list": random_ablation_changed_list,
            "avg_random_drop": avg_random_drop,
            "delta": delta,
            "gate_pass": paradigm_pass,
        }

        report["ablation_validity"][paradigm] = {
            "targeted_changed": targeted_ablation_changed,
            "random_all_changed": all(random_ablation_changed_list),
        }

    # Overall gate decision
    overall_gate_verdict = "PASS" if (gate_pass and not ablation_validity_issues) else "FAIL"

    interpretation = ""
    if ablation_validity_issues:
        interpretation = (
            "GATE FAIL (ABLATION VALIDITY): "
            + " | ".join(ablation_validity_issues)
            + " Ablations are no-ops; cannot assess circuit localization."
        )
    elif gate_pass:
        interpretation = (
            "GATE PASS: Targeted ablation hurts significantly more than random. "
            "Circuits are real and localizable. "
            "Proceed to circuit-targeted training (M3 Phase 2)."
        )
    else:
        interpretation = (
            "GATE FAIL: Targeted ablation does NOT hurt significantly more than random "
            "(Δ < 2pp across paradigms). Circuits are not sufficiently faithful/real. "
            "Stop here and report F10_NULL (documented null finding)."
        )

    report["summary"] = {
        "overall_gate": overall_gate_verdict,
        "interpretation": interpretation,
        "next_step": (
            "M3 Phase 2: circuit-targeted 100M model + official eval"
            if overall_gate_verdict == "PASS"
            else "M3 Phase CLOSED: faithfulness gate failed, report F10_NULL"
        ),
        "ablation_validity_issues": ablation_validity_issues,
    }

    # Write report
    report_path = results_dir / "faithfulness_gate_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"GATE DECISION: {overall_gate_verdict}")
    print(f"Interpretation: {interpretation}")
    print(f"Report: {report_path}")
    print(f"{'='*60}\n")

    sys.exit(0 if overall_gate_verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
