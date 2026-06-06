"""Real English kāraka assignment via spaCy dependency parsing.

Maps spaCy Universal Dependency relations to Pāṇinian kāraka roles:
- nsubj (nominal subject) → kartā (agent)
- dobj/obj (direct object) → karma (patient)
- iobj/dative (indirect object) → sampradāna (recipient)
- obl with case=Instr / prep=with/by → karaṇa (instrument)
- obl (oblique) / prep=in/on/at → adhikaraṇa (locus)
- prep=from → apādāna (source/ablative)
- amod/advmod → viśeṣaṇa (modifier)
- det/aux/punct/conj → separator (function words, conjunctions)
- ROOT/VERB → kriyā (action; treated as low-mask or contextual)

FIDELITY: Uses statistical dependency parser (spaCy en_core_web_sm).
Real parsing, not surface heuristics. Limitations: parser errors on
complex/ambiguous sentences; no semantic role labeling beyond UD structure.
"""

from __future__ import annotations

from dataclasses import dataclass

from spacy.tokens import Doc, Token


@dataclass(frozen=True)
class TokenRole:
    """A token and its assigned kāraka role."""

    token: str
    role: str


def _dep_to_karaka(token: Token) -> str:
    """Map a spaCy token's UD dependency to a kāraka role.

    Args:
        token: spaCy Token with dep_ attribute

    Returns:
        kāraka role name (kartā, karma, karaṇa, etc.) or 'unknown'
    """
    dep = token.dep_
    pos = token.pos_
    text = token.text.lower()
    head_text = token.head.text.lower() if token.head else ""

    # Verbal predicates
    if pos == "VERB" or dep == "ROOT":
        return "kriya"  # Action/verb (use ASCII for compatibility)

    # Core arguments
    if dep in {"nsubj", "nsubjpass"}:
        return "karta"  # Agent/subject

    if dep in {"dobj", "obj"}:
        return "karma"  # Patient/object (direct object of main verb)

    if dep in {"iobj", "dative"}:
        return "sampradana"  # Indirect object/recipient

    # Oblique arguments: map by preposition or case
    # pobj (prepositional object) is handled here, mapped via its preposition
    if dep in {"obl", "pobj"} or dep.startswith("obl:"):
        # Check the head preposition to determine role
        if head_text in {"with", "by"}:
            return "karana"
        # Check for locative markers
        if head_text in {"in", "on", "at", "inside", "outside", "around", "near", "under", "above"}:
            return "adhikarana"
        # Check for ablative markers
        if head_text == "from":
            return "apadana"
        # Check for recipient markers
        if head_text in {"to", "for"}:
            return "sampradana"
        # Check this token itself (in case it's the preposition)
        if text in {"with", "by"}:
            return "karana"
        if text in {"in", "on", "at", "inside", "outside", "around", "near", "under", "above"}:
            return "adhikarana"
        if text == "from":
            return "apadana"
        if text in {"to", "for"}:
            return "sampradana"
        # Default oblique → locative
        return "adhikarana"

    # Prepositions: explicit mapping (prep dep)
    if dep in {"prep"} or pos == "ADP":
        if text in {"with", "by"}:
            return "karana"
        if text in {"in", "on", "at", "inside", "outside", "around", "near", "under", "above"}:
            return "adhikarana"
        if text == "from":
            return "apadana"
        if text in {"to", "for"}:
            return "sampradana"
        return "separator"

    # Modifiers
    if dep in {"amod", "advmod"}:
        return "visesana"

    # Function words and particles
    if dep in {"det", "aux", "punct", "conj", "cc"}:
        return "separator"

    # Catch-all: function words by POS
    if pos in {"DET", "AUX", "CCONJ", "SCONJ", "PART"}:
        return "separator"

    # Unknown
    return "unknown"


def assign_karaka_roles_spacy(doc: Doc) -> list[TokenRole]:
    """Assign kāraka roles using spaCy dependency parse.

    Args:
        doc: spaCy Doc (already parsed)

    Returns:
        List of (token, role) pairs
    """
    roles: list[TokenRole] = []
    for token in doc:
        role = _dep_to_karaka(token)
        roles.append(TokenRole(token.text, role))
    return roles


def parse_and_assign(sentence: str, nlp) -> list[TokenRole]:
    """Parse a sentence with spaCy and assign kāraka roles.

    Args:
        sentence: Raw English text
        nlp: spaCy Language model (e.g., spacy.load('en_core_web_sm'))

    Returns:
        List of (token, role) pairs
    """
    doc = nlp(sentence)
    return assign_karaka_roles_spacy(doc)


def roles_to_dict(token_roles: list[TokenRole]) -> dict[str, str]:
    """Convert list of TokenRole to dict for lookup."""
    return {tr.token: tr.role for tr in token_roles}
