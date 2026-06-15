#!/usr/bin/env python3
"""Upload BabyLM intermediate checkpoints as HF revisions on the model repos."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
TOK = ROOT / "data/tokenizer/strict_small/spm.model"
api = HfApi()
JOBS = [
    ("prabhasa_b_ss_0.1/seed_0", "qbz506/prabhasa-b_ss-0.1"),
    ("prabhasa_b_s/seed_0", "qbz506/prabhasa-b_s"),
]
for ckdir, repo in JOBS:
    d = ROOT / "data/checkpoints" / ckdir
    mids = sorted(d.glob("elc_*M.pt"), key=lambda p: int(re.search(r"elc_(\d+)M", p.name).group(1)))
    print(f"\n=== {repo}: {len(mids)} checkpoints ===", flush=True)
    for ck in mids:
        step = re.search(r"elc_(\d+)M", ck.name).group(1)
        rev = f"step{step}M"
        try:
            existing = api.list_repo_refs(repo).branches
            if any(b.name == rev for b in existing):
                print(f"  {rev}: exists, skip", flush=True)
                continue
        except Exception:
            pass
        with tempfile.TemporaryDirectory() as tmp:
            r = subprocess.run(
                [
                    sys.executable,
                    "scripts/export_hf_model.py",
                    "--ckpt",
                    str(ck),
                    "--tokenizer",
                    str(TOK),
                    "--out",
                    tmp,
                    "--model-name",
                    f"{repo.split('/')[1]}-{rev}",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                print(f"  {rev}: EXPORT FAIL {r.stderr[-200:]}", flush=True)
                continue
            try:
                api.create_branch(repo, branch=rev, exist_ok=True)
                api.upload_folder(
                    folder_path=tmp,
                    repo_id=repo,
                    revision=rev,
                    commit_message=f"intermediate checkpoint @ {step}M words seen",
                )
                print(f"  {rev}: uploaded", flush=True)
            except Exception as e:
                print(f"  {rev}: UPLOAD FAIL {str(e)[-200:]}", flush=True)
print("\nDONE checkpoint upload", flush=True)
