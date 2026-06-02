"""HuggingFace export scaffold for ELC-PSALM → official BabyLM ``mlm`` backend.

Full tokenizer round-trip and Hub upload are TODO; smoke export writes a local
``PreTrainedModel``-compatible directory for ``invoke_official_zero_shot``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from psalm.domain.model.elc_config import ElcPsalmConfig
from psalm.infrastructure.ml.elc_psalm import ElcPsalmEncoder, _AttnMode
from psalm.infrastructure.ml.elc_trainer import load_elc_checkpoint

try:
    import torch
    from torch import nn
    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.modeling_outputs import MaskedLMOutput
except ImportError:  # pragma: no cover - optional without psalm[ml]
    PretrainedConfig = object  # type: ignore[misc, assignment]
    PreTrainedModel = object  # type: ignore[misc, assignment]
    MaskedLMOutput = object  # type: ignore[misc, assignment]
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


class ElcPsalmHFConfig(PretrainedConfig):
    """HF config shim mapping to :class:`ElcPsalmConfig`."""

    model_type = "elc_psalm"

    def __init__(self, elc: dict[str, Any] | ElcPsalmConfig | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if isinstance(elc, ElcPsalmConfig):
            self.elc: dict[str, Any] = elc.model_dump()
        elif isinstance(elc, dict):
            self.elc = elc
        else:
            self.elc = ElcPsalmConfig(
                vocab_size=128, d_model=32, n_layers=2, n_heads=4
            ).model_dump()


class ElcPsalmForMaskedLM(PreTrainedModel):
    """Thin ``PreTrainedModel`` wrapper over :class:`ElcPsalmEncoder` for BabyLM MLM eval."""

    config_class = ElcPsalmHFConfig

    def __init__(self, config: ElcPsalmHFConfig) -> None:
        super().__init__(config)
        elc = ElcPsalmConfig.model_validate(config.elc)
        self.encoder = ElcPsalmEncoder(elc)
        self.mask_token_id = elc.vocab_size - 1

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> MaskedLMOutput:
        del attention_mask, kwargs
        if input_ids is None:
            raise ValueError("input_ids required")
        hidden = self.encoder.encode(input_ids, attn_mode=_AttnMode.BIDIRECTIONAL)
        logits = self.encoder.lm_head(hidden)
        loss = None
        if labels is not None:
            import torch.nn.functional as F

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return MaskedLMOutput(loss=loss, logits=logits)


def export_elc_checkpoint_to_hf(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    model_name: str = "psalm-elc-psalm-smoke",
) -> Path:
    """Write a minimal HF model directory from a ``.pt`` ELC checkpoint (smoke / scaffold)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model, mask_id = load_elc_checkpoint(checkpoint_path, device="cpu")
    hf_cfg = ElcPsalmHFConfig(elc=model.cfg)
    wrapper = ElcPsalmForMaskedLM(hf_cfg)
    wrapper.encoder.load_state_dict(model.state_dict())
    if wrapper.encoder.cfg.tie_embeddings:
        wrapper.encoder.lm_head.weight = nn.Parameter(wrapper.encoder.tok.weight.detach().clone())
    wrapper.save_pretrained(output_dir)

    config_extra = {
        "architectures": ["ElcPsalmForMaskedLM"],
        "model_type": "elc_psalm",
        "mask_token_id": mask_id,
    }
    cfg_path = output_dir / "config.json"
    if cfg_path.exists():
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        data.update(config_extra)
        cfg_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    readme = output_dir / "README_EXPORT.md"
    readme.write_text(
        f"# {model_name}\n\n"
        "Smoke HF export from PSALM ELC-PSALM checkpoint.\n"
        "Official BabyLM eval: ``invoke_official_zero_shot(model_path=..., backend=mlm)``.\n"
        "TODO: joint SentencePiece tokenizer + AutoProcessor for full BLiMP/EWoK.\n",
        encoding="utf-8",
    )
    return output_dir
