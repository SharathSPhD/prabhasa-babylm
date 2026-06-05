"""Tests for converting pramana instruction-following format to NLI format.

The pramana format (instruction + output with Nyāya structure) is converted
to standard NLI triplets (premise/hypothesis/label) expected by transform_nli_dataset().

Test coverage:
1. Converter outputs correct NLI format (premise/hypothesis/label structure)
2. Label mapping from hetvabhāsa section (fallacy detection → entailment/contradiction/neutral)
3. All 75 pramana examples convert without error
4. Premise/hypothesis extraction from pratijñā/hetu
5. Edge cases (missing sections, malformed JSON, empty strings)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psalm.domain.data.pramana_converter import (
    PramanaConversionError,
    convert_pramana_example,
    extract_label_from_hetvabhasa,
)
from psalm.infrastructure.data.pramana_loader import convert_pramana_jsonl


class TestExtractLabelFromHetvabhasa:
    """Test fallacy detection and label determination."""

    def test_no_fallacy_detected_yields_entailment(self) -> None:
        """If hetvabhāsa reports no fallacy, label should be entailment (0)."""
        hetvabhasa_text = """
        savyabhichara: none_detected
        viruddha: none_detected
        prakaranasama: none_detected
        """
        label = extract_label_from_hetvabhasa(hetvabhasa_text)
        assert label == 0, "No fallacy should map to entailment"

    def test_fallacy_detected_yields_contradiction(self) -> None:
        """If hetvabhāsa reports a fallacy (viruddha/savyabhichara), label should be contradiction (2)."""
        hetvabhasa_text = """
        savyabhichara: detected - inconclusive reasoning
        viruddha: none_detected
        """
        label = extract_label_from_hetvabhasa(hetvabhasa_text)
        assert label == 2, "Detected fallacy should map to contradiction"

    def test_empty_hetvabhasa_yields_neutral(self) -> None:
        """If hetvabhāsa section is missing or empty, default to neutral (1)."""
        label = extract_label_from_hetvabhasa("")
        assert label == 1, "Missing/empty hetvabhāsa should default to neutral"

    def test_case_insensitive_fallacy_detection(self) -> None:
        """Fallacy detection should be case-insensitive."""
        hetvabhasa_text = "SAVYABHICHARA: DETECTED"
        label = extract_label_from_hetvabhasa(hetvabhasa_text)
        assert label == 2


class TestConvertPramanaExample:
    """Test conversion of individual pramana examples."""

    def test_basic_conversion_structure(self) -> None:
        """Converted example must contain premise, hypothesis, label keys."""
        pramana_ex = {
            "instruction": "Pet assignment problem",
            "input": "",
            "output": """
## Pancha Avayava
**Pratijna (Thesis)**: Alice has the fish.
**Hetu (Reason)**: Evidence here.
## Hetvabhasa (Fallacy Check)
savyabhichara: none_detected
            """,
        }
        result = convert_pramana_example(pramana_ex)

        assert isinstance(result, dict)
        assert "premise" in result
        assert "hypothesis" in result
        assert "label" in result
        assert isinstance(result["label"], int)
        assert result["label"] in [0, 1, 2]

    def test_extracts_pratijña_as_hypothesis(self) -> None:
        """Pratijñā section should become hypothesis."""
        pramana_ex = {
            "instruction": "Test",
            "input": "",
            "output": """
## Pancha Avayava
**Pratijna (Thesis)**: Alice has the fish.
**Hetu (Reason)**: Bob has the dog.
## Hetvabhasa
none_detected
            """,
        }
        result = convert_pramana_example(pramana_ex)
        assert "Alice has the fish" in result["hypothesis"]

    def test_extracts_hetu_as_premise(self) -> None:
        """Hetu section should become premise."""
        pramana_ex = {
            "instruction": "Test",
            "input": "",
            "output": """
## Pancha Avayava
**Pratijna (Thesis)**: Alice has the fish.
**Hetu (Reason)**: Bob has the dog and Carol doesn't have the cat.
## Hetvabhasa
none_detected
            """,
        }
        result = convert_pramana_example(pramana_ex)
        assert "Bob has the dog" in result["premise"] or "Carol" in result["premise"]

    def test_malformed_example_raises_error(self) -> None:
        """Missing required fields should raise PramanaConversionError."""
        with pytest.raises(PramanaConversionError):
            convert_pramana_example({"instruction": "only has instruction"})

    def test_missing_pancha_avayava_raises_error(self) -> None:
        """Output without Pancha Avayava section should raise error."""
        pramana_ex = {
            "instruction": "Test",
            "input": "",
            "output": "Some text without Pancha Avayava section",
        }
        with pytest.raises(PramanaConversionError):
            convert_pramana_example(pramana_ex)

    def test_label_from_hetvabhasa_applied(self) -> None:
        """Label should come from hetvabhāsa fallacy detection."""
        pramana_ex = {
            "instruction": "Test",
            "input": "",
            "output": """
## Pancha Avayava
**Pratijna**: Thesis.
**Hetu**: Reason.
## Hetvabhasa
savyabhichara: none_detected
viruddha: none_detected
            """,
        }
        result = convert_pramana_example(pramana_ex)
        assert result["label"] == 0  # no fallacy → entailment


class TestConvertPramanaJsonl:
    """Test batch conversion of JSONL files."""

    def test_converts_multiple_examples(self, tmp_path: Path) -> None:
        """Convert multiple pramana examples from JSONL."""
        jsonl_file = tmp_path / "test.jsonl"
        examples = [
            {
                "instruction": "Ex1",
                "input": "",
                "output": "## Pancha Avayava\n**Pratijna (Thesis)**: A.\n**Hetu (Reason)**: B.\n## Hetvabhasa\nnone_detected",
            },
            {
                "instruction": "Ex2",
                "input": "",
                "output": "## Pancha Avayava\n**Pratijna (Thesis)**: C.\n**Hetu (Reason)**: D.\n## Hetvabhasa\nnone_detected",
            },
        ]
        with open(jsonl_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        results = convert_pramana_jsonl(jsonl_file)
        assert len(results) == 2
        assert all("premise" in r and "hypothesis" in r and "label" in r for r in results)

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        """Malformed lines should be skipped with warning, not crash."""
        jsonl_file = tmp_path / "malformed.jsonl"
        with open(jsonl_file, "w") as f:
            # Write a good example
            good_ex = {
                "instruction": "good",
                "input": "",
                "output": "## Pancha Avayava\n**Pratijna (Thesis)**: A.\n**Hetu (Reason)**: B.\n## Hetvabhasa\nnone_detected",
            }
            f.write(json.dumps(good_ex) + "\n")
            # Write bad examples
            f.write("not valid json\n")
            f.write('{"incomplete": "dict"}\n')

        results = convert_pramana_jsonl(jsonl_file)
        # Should skip the bad lines but still convert the good one
        assert len(results) == 1
        assert "premise" in results[0]

    def test_returns_list_of_dicts(self, tmp_path: Path) -> None:
        """Output should be list of NLI-format dicts."""
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, "w") as f:
            ex = {
                "instruction": "test",
                "input": "",
                "output": "## Pancha Avayava\n**Pratijna (Thesis)**: X.\n**Hetu (Reason)**: Y.\n## Hetvabhasa\nnone_detected",
            }
            f.write(json.dumps(ex) + "\n")

        results = convert_pramana_jsonl(jsonl_file)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, dict) for r in results)


class TestPramanaFileIntegration:
    """Integration tests with actual pramana files (if available)."""

    def test_stage_0_converts_successfully(self) -> None:
        """All 20 examples from pramana/data/training/stage_0.jsonl should convert."""
        stage_0_path = Path("/home/sharaths/projects/pramana/data/training/stage_0.jsonl")
        if not stage_0_path.exists():
            pytest.skip("pramana/stage_0.jsonl not found")

        results = convert_pramana_jsonl(stage_0_path)
        assert len(results) == 20
        assert all("premise" in r for r in results)
        assert all("hypothesis" in r for r in results)
        assert all("label" in r for r in results)
        assert all(r["label"] in [0, 1, 2] for r in results)

    def test_stage_1_converts_successfully(self) -> None:
        """All 55 examples from pramana/data/training/stage_1.jsonl should convert."""
        stage_1_path = Path("/home/sharaths/projects/pramana/data/training/stage_1.jsonl")
        if not stage_1_path.exists():
            pytest.skip("pramana/stage_1.jsonl not found")

        results = convert_pramana_jsonl(stage_1_path)
        assert len(results) == 55
        assert all("premise" in r for r in results)
        assert all("hypothesis" in r for r in results)
        assert all("label" in r for r in results)

    def test_combined_stage_0_and_1_totals_75(self) -> None:
        """Together, stage_0 + stage_1 = 75 examples."""
        stage_0_path = Path("/home/sharaths/projects/pramana/data/training/stage_0.jsonl")
        stage_1_path = Path("/home/sharaths/projects/pramana/data/training/stage_1.jsonl")

        if not stage_0_path.exists() or not stage_1_path.exists():
            pytest.skip("pramana files not found")

        results_0 = convert_pramana_jsonl(stage_0_path)
        results_1 = convert_pramana_jsonl(stage_1_path)
        assert len(results_0) + len(results_1) == 75
