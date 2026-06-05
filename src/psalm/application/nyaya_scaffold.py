"""Navya-Nyāya Pañcāvayava reasoning scaffold for fine-tuning (H2).

Wraps NLI/BoolQ/classification tasks with the 5-component inference structure,
giving the encoder structured chain-of-thought evidence at fine-tune time.

The Pañcāvayava (5-component) inference from Navya-Nyāya (classical Indian logic):
  1. Pratijñā (Proposition): The hypothesis to be proven
  2. Hetu (Reason): Evidence supporting the hypothesis
  3. Udāharaṇa (Example): General rule or pattern
  4. Upanaya (Application): How the rule applies to this case
  5. Nigamana (Conclusion): The final inference/label
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class NLILabel(StrEnum):
    """Standard NLI labels for 3-way classification."""

    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


SEPARATOR = " [SEP] "


@dataclass
class NyayaInferenceUnit:
    """A complete 5-component Pañcāvayava inference unit.

    Attributes:
        pratijña: Proposition/hypothesis
        hetu: Reason/evidence (usually the premise)
        udaharana: General rule template
        upanaya: Application of rule to case
        nigamana: Conclusion/label
        label: Optional NLI label
    """

    pratijña: str
    hetu: str
    udaharana: str
    upanaya: str
    nigamana: str
    label: NLILabel | None = None

    def to_text(self) -> str:
        """Linearize the inference unit for tokenization.

        Format: [P] proposition [SEP] [H] reason [SEP] [U1] example [SEP] [U2] application [SEP] [N] conclusion
        """
        parts = [
            f"[P] {self.pratijña}",
            f"[H] {self.hetu}",
            f"[U1] {self.udaharana}",
            f"[U2] {self.upanaya}",
            f"[N] {self.nigamana}",
        ]
        return SEPARATOR.join(parts)

    @classmethod
    def from_nli_pair(
        cls,
        premise: str,
        hypothesis: str,
        *,
        label: str | None = None,
        udaharana_template: str = "If {premise_pattern}, then {hypothesis_pattern}",
    ) -> NyayaInferenceUnit:
        """Build a Pañcāvayava unit from an NLI premise/hypothesis pair.

        Args:
            premise: The supporting evidence (Hetu)
            hypothesis: The statement to be proven (Pratijñā)
            label: Optional NLI label (entailment/contradiction/neutral)
            udaharana_template: Template for the general rule

        Returns:
            A complete NyayaInferenceUnit
        """
        # Upanaya: the premise IS observed
        upanaya = f"We observe: {premise}"
        # Udāharaṇa: soft general rule template
        udaharana = f"In general: {udaharana_template.format(premise_pattern='such evidence exists', hypothesis_pattern='the conclusion follows')}"
        nigamana_text = f"Therefore: {hypothesis}"
        if label:
            nigamana_text += f" [{label.upper()}]"

        # Validate and convert label if provided
        label_obj = None
        if label and label in {e.value for e in NLILabel}:
            label_obj = NLILabel(label)

        return cls(
            pratijña=hypothesis,
            hetu=premise,
            udaharana=udaharana,
            upanaya=upanaya,
            nigamana=nigamana_text,
            label=label_obj,
        )


def transform_nli_dataset(
    examples: dict[str, list[Any]],
    *,
    premise_key: str = "premise",
    hypothesis_key: str = "hypothesis",
    label_key: str = "label",
    label_map: dict[int, str] | None = None,
) -> dict[str, list[Any]]:
    """HuggingFace datasets.map() compatible transform.

    Converts an NLI batch into Pañcāvayava-formatted text.
    Returns the original keys plus 'nyaya_text'.

    Args:
        examples: Batch dict with premise/hypothesis/label keys
        premise_key: Name of the premise column
        hypothesis_key: Name of the hypothesis column
        label_key: Name of the label column
        label_map: Mapping from label index to label name (default: MNLI mapping)

    Returns:
        Original dict with added 'nyaya_text' key
    """
    # Default MNLI label mapping: 0=entailment, 1=neutral, 2=contradiction
    label_map = label_map or {0: "entailment", 1: "neutral", 2: "contradiction"}

    nyaya_texts = []
    n_examples = len(examples[premise_key])

    for i in range(n_examples):
        premise = examples[premise_key][i]
        hypothesis = examples[hypothesis_key][i]
        # Handle missing label key gracefully
        raw_label = examples.get(label_key, [None] * n_examples)[i]
        label_str = None
        if raw_label is not None:
            label_str = label_map.get(int(raw_label), "unknown")

        unit = NyayaInferenceUnit.from_nli_pair(premise, hypothesis, label=label_str)
        nyaya_texts.append(unit.to_text())

    return {**examples, "nyaya_text": nyaya_texts}


def apply_nyaya_transform(
    batch: dict[str, list[Any]],
    task_name: str,
) -> dict[str, list[Any]]:
    """Apply task-specific Nyāya transformation.

    Routes to the correct column mapping for each (Super)GLUE task.

    Args:
        batch: Input batch dict
        task_name: Name of the task (rte, boolq, mnli, etc.)

    Returns:
        Batch dict with 'nyaya_text' added
    """
    task_name = task_name.lower()

    if task_name == "rte":
        # RTE: premise/hypothesis, label in {0: entailment, 1: not_entailment}
        # Map to our standard: 0=entailment, 1=neutral (treat non-entailment as neutral)
        label_map = {0: "entailment", 1: "neutral"}
        return transform_nli_dataset(batch, label_map=label_map)

    elif task_name == "boolq":
        # BoolQ: passage/question, label in {True, False}
        # Treat passage as premise, question as hypothesis
        # Map: True -> entailment, False -> contradiction
        batch_renamed = {
            "premise": batch.get("passage", []),
            "hypothesis": batch.get("question", []),
            "label": [1 if x else 0 for x in batch.get("label", [])],
        }
        label_map = {1: "entailment", 0: "contradiction"}
        return transform_nli_dataset(batch_renamed, label_map=label_map)

    elif task_name == "mnli":
        # MNLI: premise/hypothesis, gold_label in {0: entailment, 1: neutral, 2: contradiction}
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        return transform_nli_dataset(batch, label_key="gold_label", label_map=label_map)

    else:
        # Fallback: assume standard premise/hypothesis
        return transform_nli_dataset(batch)


@dataclass
class NyayaFineTuneConfig:
    """Configuration for H2 Nyāya fine-tuning.

    Attributes:
        enabled: Whether to apply Nyāya scaffold (default: False)
        tasks: List of tasks to apply to (default: ["rte", "boolq"])
        max_seq_len: Maximum sequence length (default: 512)
    """

    enabled: bool = False
    tasks: list[str] = field(default_factory=lambda: ["rte", "boolq"])
    max_seq_len: int = 512


# For torch-based ML infrastructure (infrastructure/ml/nyaya_finetune.py)
__all__ = [
    "NLILabel",
    "NyayaInferenceUnit",
    "transform_nli_dataset",
    "apply_nyaya_transform",
    "NyayaFineTuneConfig",
    "SEPARATOR",
]
