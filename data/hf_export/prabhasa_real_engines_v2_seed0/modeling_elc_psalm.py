"""Remote code for ELC-PSALM (loaded via trust_remote_code).

Exposes the MaskedLM wrapper (zero-shot mlm backend) and the base AutoModel wrapper
(returns last_hidden_state) used by the official (Super)GLUE fine-tuner.
"""

from psalm.infrastructure.ml.hf_export import (
    ElcPsalmForMaskedLM,
    ElcPsalmHFConfig,
    ElcPsalmModel,
)

__all__ = ["ElcPsalmForMaskedLM", "ElcPsalmHFConfig", "ElcPsalmModel"]
