"""Tests for Navya-Nyāya Pañcāvayava fine-tuning scaffold (H2).

Domain unit tests for the 5-component reasoning structure and NLI transformation.
"""

from __future__ import annotations

import pytest

from psalm.application.nyaya_scaffold import (
    NLILabel,
    NyayaFineTuneConfig,
    NyayaInferenceUnit,
    apply_nyaya_transform,
    transform_nli_dataset,
)
from psalm.infrastructure.ml.nyaya_finetune import NyayaClassificationHead


class TestNyayaInferenceUnit:
    """Tests for the 5-component Pañcāvayava inference unit."""

    def test_from_nli_pair_builds_unit(self) -> None:
        """Test building a unit from premise/hypothesis."""
        unit = NyayaInferenceUnit.from_nli_pair(
            premise="The cat is on the mat.",
            hypothesis="The cat is on furniture.",
            label="entailment",
        )
        assert unit.pratijña == "The cat is on furniture."
        assert unit.hetu == "The cat is on the mat."
        assert "We observe:" in unit.upanaya
        assert unit.label == NLILabel.ENTAILMENT

    def test_to_text_contains_all_markers(self) -> None:
        """Test linearization includes all 5 component markers."""
        unit = NyayaInferenceUnit.from_nli_pair(
            premise="Evidence here",
            hypothesis="Conclusion here",
            label="contradiction",
        )
        text = unit.to_text()
        assert "[P]" in text
        assert "[H]" in text
        assert "[U1]" in text
        assert "[U2]" in text
        assert "[N]" in text
        # Verify separator is used
        assert "[SEP]" in text

    def test_to_text_separator_format(self) -> None:
        """Test that components are properly separated."""
        unit = NyayaInferenceUnit.from_nli_pair(
            premise="Premise A",
            hypothesis="Hypothesis B",
            label="neutral",
        )
        text = unit.to_text()
        parts = text.split(" [SEP] ")
        assert len(parts) == 5, f"Expected 5 parts separated by [SEP], got {len(parts)}"
        assert parts[0].startswith("[P]")
        assert parts[1].startswith("[H]")
        assert parts[2].startswith("[U1]")
        assert parts[3].startswith("[U2]")
        assert parts[4].startswith("[N]")

    def test_label_none_when_not_provided(self) -> None:
        """Test that label can be None."""
        unit = NyayaInferenceUnit.from_nli_pair(
            premise="Some premise",
            hypothesis="Some hypothesis",
            label=None,
        )
        assert unit.label is None

    def test_invalid_label_ignored(self) -> None:
        """Test that invalid label strings are silently ignored."""
        unit = NyayaInferenceUnit.from_nli_pair(
            premise="Premise",
            hypothesis="Hypothesis",
            label="invalid_label_xyz",
        )
        assert unit.label is None

    def test_nli_label_enum(self) -> None:
        """Test NLILabel enum values."""
        assert NLILabel.ENTAILMENT == "entailment"
        assert NLILabel.CONTRADICTION == "contradiction"
        assert NLILabel.NEUTRAL == "neutral"


class TestTransformNLIDataset:
    """Tests for HuggingFace dataset transformation."""

    def test_transform_batch_with_default_keys(self) -> None:
        """Test transforming a batch with default column names."""
        batch = {
            "premise": ["The cat is here.", "Dogs bark."],
            "hypothesis": ["The cat exists.", "Animals make noise."],
            "label": [0, 0],  # entailment
        }
        result = transform_nli_dataset(batch)
        assert "nyaya_text" in result
        assert len(result["nyaya_text"]) == 2
        assert all("[P]" in text and "[H]" in text for text in result["nyaya_text"])

    def test_transform_preserves_original_fields(self) -> None:
        """Test that original fields are preserved."""
        batch = {
            "premise": ["Premise A"],
            "hypothesis": ["Hypothesis B"],
            "label": [1],
        }
        result = transform_nli_dataset(batch)
        assert result["premise"] == batch["premise"]
        assert result["hypothesis"] == batch["hypothesis"]
        assert result["label"] == batch["label"]

    def test_transform_with_custom_label_map(self) -> None:
        """Test transformation with custom label mapping."""
        batch = {
            "premise": ["P1"],
            "hypothesis": ["H1"],
            "label": [10],
        }
        custom_map = {10: "custom_label", 11: "other"}
        result = transform_nli_dataset(batch, label_map=custom_map)
        assert "nyaya_text" in result
        # Labels are uppercased in the output
        assert "[CUSTOM_LABEL]" in result["nyaya_text"][0]

    def test_transform_with_missing_labels(self) -> None:
        """Test transformation when label key is missing."""
        batch = {
            "premise": ["Premise A"],
            "hypothesis": ["Hypothesis B"],
        }
        result = transform_nli_dataset(batch, label_key="missing_label")
        assert "nyaya_text" in result
        assert len(result["nyaya_text"]) == 1

    def test_transform_with_custom_keys(self) -> None:
        """Test transformation with custom column names."""
        batch = {
            "text_a": ["The sky is blue."],
            "text_b": ["It is colored."],
            "label_col": [0],
        }
        result = transform_nli_dataset(
            batch,
            premise_key="text_a",
            hypothesis_key="text_b",
            label_key="label_col",
        )
        assert "nyaya_text" in result
        assert "[P]" in result["nyaya_text"][0]

    def test_transform_batch_size_consistency(self) -> None:
        """Test that output batch size matches input."""
        batch = {
            "premise": ["P1", "P2", "P3"],
            "hypothesis": ["H1", "H2", "H3"],
            "label": [0, 1, 2],
        }
        result = transform_nli_dataset(batch)
        assert len(result["nyaya_text"]) == 3

    def test_transform_default_label_map(self) -> None:
        """Test default MNLI label mapping (0=ent, 1=neutral, 2=contra)."""
        batch = {
            "premise": ["P1", "P2", "P3"],
            "hypothesis": ["H1", "H2", "H3"],
            "label": [0, 1, 2],
        }
        result = transform_nli_dataset(batch)
        # Labels are uppercased in the output
        assert "[ENTAILMENT]" in result["nyaya_text"][0]
        assert "[NEUTRAL]" in result["nyaya_text"][1]
        assert "[CONTRADICTION]" in result["nyaya_text"][2]


class TestNyayaFineTuneConfig:
    """Tests for H2 fine-tune configuration."""

    def test_default_config(self) -> None:
        """Test default fine-tune config."""
        cfg = NyayaFineTuneConfig()
        assert cfg.enabled is False
        assert cfg.tasks == ["rte", "boolq"]
        assert cfg.max_seq_len == 512

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        cfg = NyayaFineTuneConfig(enabled=True, tasks=["mnli"], max_seq_len=256)
        assert cfg.enabled is True
        assert cfg.tasks == ["mnli"]
        assert cfg.max_seq_len == 256

    def test_config_with_all_tasks(self) -> None:
        """Test config accepting all task types."""
        cfg = NyayaFineTuneConfig(tasks=["rte", "boolq", "mnli"])
        assert len(cfg.tasks) == 3


class TestApplyNyayaTransform:
    """Tests for applying task-specific transformations."""

    def test_apply_transform_rte(self) -> None:
        """Test RTE transformation (standard premise/hypothesis)."""
        batch = {
            "premise": ["Text A"],
            "hypothesis": ["Text B"],
            "label": [0],
        }
        result = apply_nyaya_transform(batch, "rte")
        assert "nyaya_text" in result

    def test_apply_transform_boolq(self) -> None:
        """Test BoolQ transformation (passage/question)."""
        batch = {
            "passage": ["The cat sat on the mat."],
            "question": ["Did the cat sit?"],
            "label": [1],  # True
        }
        result = apply_nyaya_transform(batch, "boolq")
        assert "nyaya_text" in result

    def test_apply_transform_mnli(self) -> None:
        """Test MNLI transformation (premise/hypothesis)."""
        batch = {
            "premise": ["Premise text"],
            "hypothesis": ["Hypothesis text"],
            "gold_label": [0],  # entailment
        }
        result = apply_nyaya_transform(batch, "mnli")
        assert "nyaya_text" in result


class TestNyayaClassificationHead:
    """Tests for the classification head."""

    @pytest.mark.parametrize("num_labels,expected_out", [(2, 2), (3, 3)])
    def test_forward_pass_shape(self, num_labels: int, expected_out: int) -> None:
        """Test that output shape matches num_labels."""
        pytest.importorskip("torch")
        import torch

        head = NyayaClassificationHead(d_model=256, num_labels=num_labels, dropout=0.1)
        x = torch.randn(4, 256)  # batch_size=4, d_model=256
        out = head(x)
        assert out.shape == (4, num_labels)

    def test_head_with_zero_dropout(self) -> None:
        """Test that head works with zero dropout."""
        pytest.importorskip("torch")
        import torch

        head = NyayaClassificationHead(d_model=128, num_labels=2, dropout=0.0)
        x = torch.randn(2, 128)
        out = head(x)
        assert out.shape == (2, 2)

    def test_head_initialization(self) -> None:
        """Test head initialization."""
        pytest.importorskip("torch")
        head = NyayaClassificationHead(d_model=512, num_labels=3, dropout=0.2)
        assert head.d_model == 512
        assert head.num_labels == 3
