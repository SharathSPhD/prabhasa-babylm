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
    from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput
except ImportError:  # pragma: no cover - optional without psalm[ml]
    PretrainedConfig = object  # type: ignore[misc, assignment]
    PreTrainedModel = object  # type: ignore[misc, assignment]
    MaskedLMOutput = object  # type: ignore[misc, assignment]
    BaseModelOutput = object  # type: ignore[misc, assignment]
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def _make_nhot_placeholder(elc: ElcPsalmConfig) -> Any:
    """Return a zero-weight NhotEmbedding placeholder; state_dict loading fills real weights."""
    from psalm.infrastructure.ml.nhot_embeddings import NHOT_DIM, NhotEmbedding

    return NhotEmbedding(torch.zeros(elc.vocab_size, NHOT_DIM), elc.d_model)


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
        # Flat fields the official pipeline / AutoConfig consumers read. The GLUE
        # fine-tuner (evaluation_pipeline/finetune/classifier_model.py) reads
        # ``hidden_size`` off AutoConfig to size its classification head.
        self.hidden_size = int(self.elc["d_model"])
        self.num_attention_heads = int(self.elc["n_heads"])
        self.num_hidden_layers = int(self.elc["n_layers"])
        self.vocab_size = int(self.elc["vocab_size"])


class ElcPsalmForMaskedLM(PreTrainedModel):
    """Thin ``PreTrainedModel`` wrapper over :class:`ElcPsalmEncoder` for BabyLM MLM eval."""

    config_class = ElcPsalmHFConfig
    _tied_weights_keys: dict[str, str] = {}

    def __init__(self, config: ElcPsalmHFConfig) -> None:
        config.tie_word_embeddings = False
        super().__init__(config)
        elc = ElcPsalmConfig.model_validate(config.elc)
        nhot_emb = _make_nhot_placeholder(elc) if elc.nhot_embeddings else None
        self.encoder = ElcPsalmEncoder(elc, nhot_emb=nhot_emb)
        self.mask_token_id = elc.vocab_size - 1
        self.post_init()

    def get_input_embeddings(self) -> Any:
        return self.encoder.tok

    def set_input_embeddings(self, value: Any) -> None:
        self.encoder.tok = value

    def get_output_embeddings(self) -> Any:
        return self.encoder.lm_head

    def set_output_embeddings(self, value: Any) -> None:
        self.encoder.lm_head = value

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> MaskedLMOutput:
        del kwargs
        if input_ids is None:
            raise ValueError("input_ids required")
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask.bool()
        hidden = self.encoder.encode(
            input_ids,
            attn_mode=_AttnMode.BIDIRECTIONAL,
            key_padding_mask=key_padding_mask,
        )
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


class ElcPsalmModel(PreTrainedModel):
    """Base ``AutoModel`` wrapper returning ``last_hidden_state``.

    The official BabyLM (Super)GLUE fine-tuner loads the model via
    ``AutoModel.from_pretrained(..., trust_remote_code=True)`` and attaches its own
    classification head, reading ``last_hidden_state`` (CLS/token-0 pooling). This class
    shares the exact ``encoder.*`` parameter layout of :class:`ElcPsalmForMaskedLM`, so a
    single exported checkpoint loads into either wrapper.
    """

    config_class = ElcPsalmHFConfig
    _tied_weights_keys: dict[str, str] = {}

    def __init__(self, config: ElcPsalmHFConfig) -> None:
        config.tie_word_embeddings = False
        super().__init__(config)
        elc = ElcPsalmConfig.model_validate(config.elc)
        nhot_emb = _make_nhot_placeholder(elc) if elc.nhot_embeddings else None
        self.encoder = ElcPsalmEncoder(elc, nhot_emb=nhot_emb)
        self.post_init()

    def get_input_embeddings(self) -> Any:
        return self.encoder.tok

    def set_input_embeddings(self, value: Any) -> None:
        self.encoder.tok = value

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutput:
        del kwargs
        if input_ids is None:
            raise ValueError("input_ids required")
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask.bool()
        hidden = self.encoder.encode(
            input_ids,
            attn_mode=_AttnMode.BIDIRECTIONAL,
            key_padding_mask=key_padding_mask,
        )
        return BaseModelOutput(last_hidden_state=hidden)


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
    # Swap in the already-loaded encoder; avoids a redundant state_dict round-trip
    # and correctly propagates N-hot weights when nhot_embeddings=True.
    wrapper.encoder = model
    if model.cfg.tie_embeddings:
        model.lm_head.weight = nn.Parameter(model.tok.weight.detach().clone())
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
