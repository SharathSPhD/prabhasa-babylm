"""ML infrastructure for Navya-Nyāya fine-tuning integration with ELC-PSALM.

This module wraps the domain-pure Pañcāvayava scaffold with PyTorch/transformers
integration for the official (Super)GLUE fine-tuning pipeline.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from psalm.application.nyaya_scaffold import NyayaFineTuneConfig, apply_nyaya_transform


class NyayaClassificationHead(nn.Module):
    """Classification head on top of ELC-PSALM [CLS] representation (H2).

    A simple 2-layer MLP that takes the pooled [CLS] token from the encoder
    and produces class logits. Used for fine-tuning on classification tasks.

    Attributes:
        d_model: Hidden dimension of the encoder
        num_labels: Number of output classes
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        num_labels: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the classification head.

        Args:
            d_model: Hidden dimension of the encoder
            num_labels: Number of output classes
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()
        self.d_model = d_model
        self.num_labels = num_labels
        self.dropout_rate = dropout

        # 2-layer MLP: d_model -> d_model -> num_labels
        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch_size, d_model] tensor (typically [CLS] representation)

        Returns:
            [batch_size, num_labels] logits
        """
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def apply_nyaya_transforms_to_dataset(
    dataset: Any,
    *,
    task_name: str,
    text_key: str = "nyaya_text",
) -> Any:
    """Apply Nyāya transform to a HuggingFace dataset.

    This is a wrapper for integration with the existing evaluation_pipeline.
    It applies the task-specific Pañcāvayava transformation to each batch.

    Args:
        dataset: HuggingFace Dataset object
        task_name: Task name (rte, boolq, mnli, etc.)
        text_key: Output key for Nyāya text (default: "nyaya_text")

    Returns:
        Transformed dataset with 'nyaya_text' column added
    """
    def transform_fn(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        return apply_nyaya_transform(batch, task_name)

    # Apply the transform to the entire dataset
    return dataset.map(transform_fn, batched=True, desc=f"Applying Nyāya transform to {task_name}")


__all__ = [
    "NyayaClassificationHead",
    "NyayaFineTuneConfig",
    "apply_nyaya_transforms_to_dataset",
]
