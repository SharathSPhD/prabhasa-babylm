#!/usr/bin/env python3
"""Export an ELC-PSALM ``.pt`` checkpoint to a HF model dir for the official pipeline.

Produces a directory loadable by ``AutoModelForMaskedLM.from_pretrained(...,
trust_remote_code=True)`` and ``AutoProcessor/AutoTokenizer.from_pretrained(...)``
— the exact contract of ``babylm/evaluation-pipeline-2025`` sentence_zero_shot
(mlm backend). Writes:

  * model.safetensors + config.json (with ``auto_map`` to the remote modeling
    files), and self-contained ``modeling_elc_psalm.py`` / ``configuration_elc_psalm.py``
  * a **fast** tokenizer rebuilt from the joint SentencePiece Unigram model whose
    ``encode`` is byte-for-byte id-identical to ``sp.EncodeAsIds`` (validated here),
    with ``mask_token`` bound to SP id ``vocab-1`` (the id the trainer masks with).

Then validates the round-trip: Auto* load + a masked forward + tokenizer id parity.

    uv run python scripts/export_hf_model.py \
        --ckpt data/checkpoints/recipe_v2/arm_A_seed_0/elc.pt \
        --tokenizer data/tokenizer/strict_small/spm.model \
        --out data/hf_export/recipe_v2_armA
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_MODELING_PY = '''"""Remote code for ELC-PSALM (loaded via trust_remote_code).

Exposes the MaskedLM wrapper (zero-shot mlm backend) and the base AutoModel wrapper
(returns last_hidden_state) used by the official (Super)GLUE fine-tuner.
"""
from psalm.infrastructure.ml.hf_export import (
    ElcPsalmForMaskedLM,
    ElcPsalmHFConfig,
    ElcPsalmModel,
)

__all__ = ["ElcPsalmForMaskedLM", "ElcPsalmHFConfig", "ElcPsalmModel"]
'''

_CONFIG_PY = '''"""Remote config for ELC-PSALM (loaded via trust_remote_code)."""
from psalm.infrastructure.ml.hf_export import ElcPsalmHFConfig

__all__ = ["ElcPsalmHFConfig"]
'''


def build_fast_tokenizer(spm_path: str):
    """Fast Unigram tokenizer whose ids match ``sp.EncodeAsIds`` exactly."""
    import sentencepiece as spm
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)
    vocab = [(sp.IdToPiece(i), sp.GetScore(i)) for i in range(sp.GetPieceSize())]
    tk = Tokenizer(models.Unigram(vocab, unk_id=sp.unk_id(), byte_fallback=False))
    tk.normalizer = normalizers.Sequence([normalizers.Nmt(), normalizers.NFKC()])
    tk.pre_tokenizer = pre_tokenizers.Metaspace(replacement="\u2581", prepend_scheme="always")
    tk.decoder = decoders.Metaspace(replacement="\u2581", prepend_scheme="always")

    mask_piece = sp.IdToPiece(sp.GetPieceSize() - 1)  # trainer masks with id vocab-1
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="</s>",
        mask_token=mask_piece,
        model_input_names=["input_ids", "attention_mask"],
    )
    return fast, sp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer/strict_small/spm.model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-name", default="elc-psalm-s")
    args = ap.parse_args()

    import torch
    from torch import nn
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    from psalm.infrastructure.ml.elc_trainer import load_elc_checkpoint
    from psalm.infrastructure.ml.hf_export import ElcPsalmForMaskedLM, ElcPsalmHFConfig

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, mask_id = load_elc_checkpoint(Path(args.ckpt), device="cpu")
    hf_cfg = ElcPsalmHFConfig(elc=model.cfg)
    wrapper = ElcPsalmForMaskedLM(hf_cfg)
    wrapper.encoder.load_state_dict(model.state_dict())
    if wrapper.encoder.cfg.tie_embeddings:
        wrapper.encoder.lm_head.weight = nn.Parameter(wrapper.encoder.tok.weight.detach().clone())
    wrapper.save_pretrained(out)

    # auto_map + flat fields the pipeline / Auto* may read.
    cfg_path = out / "config.json"
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    data.update(
        {
            "architectures": ["ElcPsalmForMaskedLM"],
            "model_type": "elc_psalm",
            "mask_token_id": mask_id,
            "vocab_size": model.cfg.vocab_size,
            "hidden_size": model.cfg.d_model,
            "num_attention_heads": model.cfg.n_heads,
            "num_hidden_layers": model.cfg.n_layers,
            "max_position_embeddings": model.cfg.max_seq_len,
            "auto_map": {
                "AutoConfig": "configuration_elc_psalm.ElcPsalmHFConfig",
                "AutoModel": "modeling_elc_psalm.ElcPsalmModel",
                "AutoModelForMaskedLM": "modeling_elc_psalm.ElcPsalmForMaskedLM",
            },
        }
    )
    cfg_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    (out / "modeling_elc_psalm.py").write_text(_MODELING_PY, encoding="utf-8")
    (out / "configuration_elc_psalm.py").write_text(_CONFIG_PY, encoding="utf-8")

    fast, sp = build_fast_tokenizer(args.tokenizer)
    fast.save_pretrained(out)

    # ---- Validate the round-trip (fail loud) ----
    samples = [
        "The cat sleeps on the mat.",
        "Who should Derek hug after shocking Richard?",
        "colourless green ideas sleep furiously",
    ]
    tok = AutoTokenizer.from_pretrained(out, trust_remote_code=True)
    for s in samples:
        ref = sp.EncodeAsIds(s)
        got = tok(s, add_special_tokens=False)["input_ids"]
        if ref != got:
            raise SystemExit(f"TOKENIZER MISMATCH on {s!r}:\n  spm={ref}\n  hf ={got}")
    assert tok.mask_token_id == mask_id, (tok.mask_token_id, mask_id)

    reloaded = AutoModelForMaskedLM.from_pretrained(out, trust_remote_code=True)
    reloaded.eval()
    ids = torch.tensor([sp.EncodeAsIds(samples[0])], dtype=torch.long)
    ids[0, 2] = mask_id
    with torch.no_grad():
        logits = reloaded(input_ids=ids).logits
    assert logits.shape == (1, ids.shape[1], model.cfg.vocab_size), logits.shape

    # AutoModel base path (used by the official (Super)GLUE fine-tuner).
    from transformers import AutoConfig, AutoModel

    base = AutoModel.from_pretrained(out, trust_remote_code=True)
    base.eval()
    auto_cfg = AutoConfig.from_pretrained(out, trust_remote_code=True)
    assert int(auto_cfg.hidden_size) == int(model.cfg.d_model), auto_cfg.hidden_size
    attn = torch.ones_like(ids)
    with torch.no_grad():
        hidden = base(input_ids=ids, attention_mask=attn).last_hidden_state
    assert hidden.shape == (1, ids.shape[1], model.cfg.d_model), hidden.shape

    print(f"OK HF export -> {out}")
    print(f"  tokenizer id-identical to SP; mask_token_id={tok.mask_token_id}")
    print(f"  AutoModelForMaskedLM loads (trust_remote_code); logits {tuple(logits.shape)}")
    print(f"  AutoModel base loads; hidden_size={auto_cfg.hidden_size}; last_hidden_state {tuple(hidden.shape)}")


if __name__ == "__main__":
    main()
