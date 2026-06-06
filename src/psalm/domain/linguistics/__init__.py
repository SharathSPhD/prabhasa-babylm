"""Linguistic engines for kāraka role assignment using real dependency parsing.

This module provides REAL (not heuristic) analyzers for English syntactic structure,
replacing the BPE-word-initial heuristic used in the initial H1_MECHANISM implementation.

Entry point: `english_karaka_real` module for spaCy-based kāraka assignment.
"""

from .english_karaka_real import (
    TokenRole,
    assign_karaka_roles_spacy,
    parse_and_assign,
    roles_to_dict,
)

__all__ = [
    "TokenRole",
    "assign_karaka_roles_spacy",
    "parse_and_assign",
    "roles_to_dict",
]
