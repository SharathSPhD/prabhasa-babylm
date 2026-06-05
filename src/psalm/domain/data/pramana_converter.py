"""Domain layer: convert pramana Nyāya format to NLI triplets.

This module implements pure data transformation logic (no I/O, no torch).
The pramana format (instruction + Pancha Avayava + hetvabhāsa structure) is
converted to standard NLI format (premise/hypothesis/label) expected by PSALM's
transform_nli_dataset() function.

Conversion rules:
- Pratijñā (Thesis) → hypothesis (what is to be proven)
- Hetu (Reason) → premise (evidence supporting the hypothesis)
- Hetvabhāsa (Fallacy check):
    - "none_detected" (valid inference) → label 0 (entailment)
    - Fallacy detected (savyabhichara/viruddha/etc.) → label 2 (contradiction)
    - Missing/unclear → label 1 (neutral)

This maintains hexagonal architecture: domain stays pure, I/O happens in
infrastructure layer (pramana_loader.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class PramanaConversionError(Exception):
    """Raised when pramana example cannot be converted to NLI format."""

    pass


@dataclass
class NLIExample:
    """A converted NLI example ready for fine-tuning."""

    premise: str
    hypothesis: str
    label: int  # 0=entailment, 1=neutral, 2=contradiction


def extract_label_from_hetvabhasa(hetvabhasa_section: str) -> int:
    """Determine NLI label from hetvabhāsa (fallacy check) section.

    Args:
        hetvabhasa_section: The Hetvabhasa section text from pramana output

    Returns:
        int: NLI label: 0=entailment (valid), 1=neutral (unclear), 2=contradiction (fallacy)

    Logic:
    - If hetvabhāsa explicitly states "none_detected" for all fallacy types,
      the reasoning is valid → entailment (0)
    - If any fallacy is detected (savyabhichara, viruddha, prakaranasama, etc.),
      the inference is fallacious → contradiction (2)
    - If hetvabhāsa is missing or ambiguous, default to neutral (1)
    """
    if not hetvabhasa_section or not hetvabhasa_section.strip():
        # Missing hetvabhāsa section defaults to neutral
        return 1

    hetvabhasa_lower = hetvabhasa_section.lower()

    # Check for detection of any fallacy type (look for patterns like "X: detected")
    # First, look for explicit detection markers
    fallacy_types = [
        "savyabhichara",
        "viruddha",
        "prakaranasama",
        "sadhyasama",
        "kalaatita",
    ]

    # Scan for "fallacy_type: detected" patterns (case-insensitive)
    detected_any_fallacy = False
    for fallacy in fallacy_types:
        # Look for pattern like "savyabhichara: detected" (more specific)
        if re.search(rf"{fallacy}\s*:\s*detected", hetvabhasa_lower):
            detected_any_fallacy = True
            break

    if detected_any_fallacy:
        return 2  # Fallacy detected → contradiction

    # Check for "none_detected" patterns indicating valid reasoning
    # A proper hetvabhāsa check should have "none_detected" statements
    none_detected_count = hetvabhasa_lower.count("none_detected")

    # If we find 2+ "none_detected" markers, treat as valid inference
    if none_detected_count >= 2:
        return 0  # Valid reasoning → entailment

    # If we have fallacy type markers but no clear none_detected or detected,
    # or if we have very few none_detected but no explicit detection either,
    # default to neutral
    return 1


def extract_section(output_text: str, section_name: str, partial_match: bool = True) -> str:
    """Extract a section from pramana output by heading.

    Args:
        output_text: Full pramana output text
        section_name: Section heading to find (e.g., "Pancha Avayava", "Hetvabhasa")
        partial_match: If True, match section_name as substring (for "Pancha Avayava (5-Member Syllogism)")

    Returns:
        str: The section content (or empty string if not found)

    Searches for markdown-style headers (## Section Name) and extracts text until
    the next main header (##) or end of file. Subsections (###, ####, etc.) are included.
    """
    # Look for ## Section Name (case-insensitive)
    # Pattern: ## optional whitespace, optional text before section_name, section_name, optional text until newline
    # Then capture everything until next ## (main section header)
    if partial_match:
        # Allow partial matching: "## ... Pancha Avayava ... \n content until next ##"
        pattern = rf"^##\s*[^#]*{re.escape(section_name)}[^\n]*$\n(.*?)(?=\n##\s|\Z)"
    else:
        pattern = rf"^##\s*{re.escape(section_name)}[^\n]*$\n(.*?)(?=\n##\s|\Z)"

    match = re.search(pattern, output_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_field_from_markdown(section_text: str, field_name: str) -> str:
    """Extract a field value from markdown-formatted section.

    Args:
        section_text: Section text containing **Field Name**: value
        field_name: Field to extract (e.g., "Pratijna", "Hetu")

    Returns:
        str: The field value (text after the markdown bold), or empty string

    Handles patterns like:
    - **Pratijna (Thesis)**: Alice has the fish.
    - **Hetu (Reason)**: Bob has the dog.
    """
    # Pattern: **Field (anything)**: value
    pattern = rf"\*\*{re.escape(field_name)}[^*]*\*\*:\s*([^\n]+)"
    match = re.search(pattern, section_text, re.IGNORECASE)
    if match:
        value = match.group(1).strip()
        # Strip any trailing markdown like bullet points or additional formatting
        value = re.sub(r"\s*[-*].*$", "", value)
        return value
    return ""


def convert_pramana_example(pramana_dict: dict[str, str]) -> dict[str, str | int]:
    """Convert a single pramana example to NLI format.

    Args:
        pramana_dict: A pramana example with keys: instruction, input, output

    Returns:
        dict: NLI format with keys: premise, hypothesis, label

    Raises:
        PramanaConversionError: If required fields are missing or format is invalid

    The pramana format has:
    - ## Pancha Avayava (5-Member Syllogism) section with subsections for each syllogism
    - Each syllogism has **Pratijna (Thesis)**: ... and **Hetu (Reason)**: ...
    - ## Hetvabhasa (Fallacy Check) section with fallacy detection results

    We extract from the first syllogism and use hetvabhāsa for the label.
    """
    # Validate input structure
    if not isinstance(pramana_dict, dict):
        raise PramanaConversionError("pramana_dict must be a dictionary")

    required_keys = ["instruction", "input", "output"]
    missing = [k for k in required_keys if k not in pramana_dict]
    if missing:
        raise PramanaConversionError(f"Missing required keys: {missing}")

    output_text = pramana_dict.get("output", "")
    if not output_text or not isinstance(output_text, str):
        raise PramanaConversionError("output must be a non-empty string")

    # Extract sections (using partial matching for variant section names)
    pancha_avayava = extract_section(output_text, "Pancha Avayava", partial_match=True)
    if not pancha_avayava:
        raise PramanaConversionError("Output does not contain 'Pancha Avayava' section")

    hetvabhasa = extract_section(output_text, "Hetvabhasa", partial_match=True)

    # Extract hypothesis (pratijñā/thesis) - look for first occurrence
    hypothesis = extract_field_from_markdown(pancha_avayava, "Pratijna")
    if not hypothesis:
        raise PramanaConversionError(
            "Could not extract Pratijna (Thesis) from Pancha Avayava section"
        )

    # Extract premise (hetu/reason)
    premise = extract_field_from_markdown(pancha_avayava, "Hetu")
    if not premise:
        raise PramanaConversionError("Could not extract Hetu (Reason) from Pancha Avayava section")

    # Determine label from hetvabhāsa
    label = extract_label_from_hetvabhasa(hetvabhasa)

    return {
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label,
    }


__all__ = [
    "PramanaConversionError",
    "NLIExample",
    "extract_label_from_hetvabhasa",
    "extract_section",
    "extract_field_from_markdown",
    "convert_pramana_example",
]
