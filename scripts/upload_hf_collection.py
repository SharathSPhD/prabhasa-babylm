#!/usr/bin/env python3
"""Publish the PSALM artifact collection to the Hugging Face Hub.

Exports each available ELC-PSALM checkpoint to HF format (``AutoModel`` +
``AutoModelForMaskedLM`` via ``trust_remote_code``), writes a model card, and
uploads it under the ``qbz506`` namespace; also uploads the synthetic dose corpora
and the auditable corpus manifest as a dataset. Defaults to a dry run that prints the
plan; pass ``--push`` to actually upload (requires ``huggingface-cli login`` or
``HF_TOKEN``).

    uv run python scripts/upload_hf_collection.py            # dry run
    uv run python scripts/upload_hf_collection.py --push     # upload
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
CKPT = ROOT / "data/checkpoints/strict_small"
TOK = ROOT / "data/tokenizer/strict_small/spm.model"
ARMS_MANIFEST = ROOT / "docs/data/strict-small-arms.json"
DOSE_DIR = ROOT / "data/corpora/strict_small/arms"

ARM_DOSE = {"A": "English (control)", "B": "Pāṇinian", "C": "Dyck (k-shuffle)", "D": "Paribhāṣā"}


def model_card(arm: str) -> str:
    dose = ARM_DOSE.get(arm, arm)
    return f"""---
license: mit
language:
- en
- sa
tags:
- babylm
- elc-bert
- sanskrit
- paninian
- pseudo-log-likelihood
library_name: transformers
pipeline_tag: fill-mask
---

# PSALM ELC-PSALM-S — arm {arm} ({dose} dose)

Small bidirectional ELC-BERT-style encoder trained from scratch under the BabyLM
Strict-Small protocol. This is **ablation arm {arm}**: the stage-one structural dose
is **{dose}**, trimmed to the same token budget as every other arm over a shared
English base, so differences between arms are attributable to dose content under a
fixed budget rather than to data volume.

Trained jointly with masked and causal objectives; minimal pairs are scored by
Salazar-style pseudo-log-likelihood. The export registers both `AutoModel` (base
encoder, returns `last_hidden_state`) and `AutoModelForMaskedLM`, so the official
BabyLM (Super)GLUE fine-tuner can load it directly.

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("qbz506/psalm-arm-{arm.lower()}", trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained("qbz506/psalm-arm-{arm.lower()}", trust_remote_code=True)
```

See the [project site](https://SharathSPhD.github.io/PSALM/) and
[repository](https://github.com/SharathSPhD/PSALM) for the method, the seed-replicated
results, and the scope statement. This checkpoint is part of a controlled scientific
ablation; for the leaderboard-track model see `qbz506/psalm-submission`.
"""


def dataset_card() -> str:
    return """---
license: mit
language:
- sa
- en
tags:
- babylm
- sanskrit
- paninian
- synthetic
- structural-prior
pretty_name: PSALM Strict-Small dose corpora
---

# PSALM Strict-Small dose corpora

The four token-matched structural doses used by the PSALM BabyLM Strict-Small
battery, plus the machine-readable corpus manifest with per-source word/token counts
and SHA-256 hashes that make the ten-million-word budget auditable.

Each dose is trimmed to an identical token budget over a shared English base: `dose_A`
(held-out English control), `dose_B` (Pāṇinian synthetic Sanskrit with gold kāraka
structure), `dose_C` (k-Shuffle Dyck control), and `dose_D` (linearized Navya-Nyāya
Paribhāṣā graphs from the deterministic Śabdabodha pipeline). See the
[repository](https://github.com/SharathSPhD/PSALM) for assembly scripts.
"""


def run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", default="qbz506")
    ap.add_argument("--arms", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--include-submission", action="store_true")
    ap.add_argument("--push", action="store_true", help="actually upload (else dry run)")
    args = ap.parse_args()

    api = None
    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()

    staged: list[tuple[str, str]] = []  # (repo_id, repo_type)

    # Models.
    ckpts = {a: CKPT / f"arm_{a}_seed_0" / "elc.pt" for a in args.arms}
    if args.include_submission:
        ckpts["submission"] = ROOT / "data/checkpoints/submission/elc.pt"
    for arm, ckpt in ckpts.items():
        repo = (
            f"{args.namespace}/psalm-arm-{arm.lower()}"
            if arm in ARM_DOSE
            else f"{args.namespace}/psalm-submission"
        )
        if not ckpt.exists():
            print(f"[skip] {repo}: checkpoint not found ({ckpt})")
            continue
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            print(f"[model] export {ckpt} -> {out}")
            run(
                [
                    PY,
                    "scripts/export_hf_model.py",
                    "--ckpt",
                    str(ckpt),
                    "--tokenizer",
                    str(TOK),
                    "--out",
                    str(out),
                ]
            )
            # Ship the SentencePiece model alongside for the Colab reproduce notebook.
            if TOK.exists():
                (out / "spm.model").write_bytes(TOK.read_bytes())
            (out / "README.md").write_text(
                model_card(arm if arm in ARM_DOSE else "submission"), encoding="utf-8"
            )
            staged.append((repo, "model"))
            if args.push and api is not None:
                api.create_repo(repo, repo_type="model", exist_ok=True)
                api.upload_folder(folder_path=str(out), repo_id=repo, repo_type="model")
                print(f"[pushed] {repo}")

    # Dataset (dose corpora + manifest).
    ds_repo = f"{args.namespace}/psalm-corpora"
    if DOSE_DIR.exists():
        staged.append((ds_repo, "dataset"))
        if args.push and api is not None:
            with tempfile.TemporaryDirectory() as td:
                out = Path(td)
                for f in sorted(DOSE_DIR.glob("dose_*.txt")):
                    (out / f.name).write_bytes(f.read_bytes())
                if ARMS_MANIFEST.exists():
                    (out / "strict-small-arms.json").write_bytes(ARMS_MANIFEST.read_bytes())
                (out / "README.md").write_text(dataset_card(), encoding="utf-8")
                api.create_repo(ds_repo, repo_type="dataset", exist_ok=True)
                api.upload_folder(folder_path=str(out), repo_id=ds_repo, repo_type="dataset")
                print(f"[pushed] {ds_repo}")

    print("\nPlan:" if not args.push else "\nUploaded:")
    for repo, kind in staged:
        print(
            f"  - {kind}: https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}"
        )
    if not args.push:
        print("\n(dry run — re-run with --push after `huggingface-cli login` to upload)")


if __name__ == "__main__":
    main()
