"""Morphological analysis infrastructure for English and Sanskrit.

This module provides real morphological segmentation and feature extraction,
replacing heuristic suffix/prefix lists with principled morpheme analyzers.
"""

from __future__ import annotations

from psalm.infrastructure.morphology.english import (
    EnglishMorphemeAnalyzer,
    Morpheme,
    segment_english_token,
)
from psalm.infrastructure.morphology.sanskrit import (
    SanskritMorpheme,
    SanskritMorphemeAnalyzer,
    analyze_sanskrit_token,
)

__all__ = [
    "EnglishMorphemeAnalyzer",
    "Morpheme",
    "segment_english_token",
    "SanskritMorphemeAnalyzer",
    "SanskritMorpheme",
    "analyze_sanskrit_token",
]
