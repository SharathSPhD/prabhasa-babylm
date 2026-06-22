"""English dependency relations → Pāṇinian kāraka role mapper (M2b).

Maps spaCy universal dependency relations to kāraka roles using morphosyntactic
evidence. Role assignments include per-token confidence scores; low-confidence
tokens fall back to the default heuristic masking probability.

Caches role lookups at data-load time for efficiency.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    import spacy
except ImportError as e:
    raise ImportError(
        "spacy is required for M2b (deprel mapper). Install via: pip install spacy"
    ) from e


# Mapping from universal deprel to kāraka role with confidence base
# (confidence can be further adjusted per token based on morphology)
DEPREL_TO_KARAKA: dict[str, tuple[str, float]] = {
    "nsubj": ("karta", 0.95),  # nominal subject → kartā (agent)
    "nsubjpass": ("karma", 0.90),  # passive subject → karma (patient in passive)
    "obj": ("karma", 0.90),  # direct object → karma (patient)
    "dobj": ("karma", 0.90),  # legacy: direct object
    "iobj": ("sampradana", 0.85),  # indirect object → sampradāna (recipient)
    "obl": ("adhikarana", 0.70),  # oblique → adhikaraṇa (locus, context-dependent)
    "obl:tmod": ("adhikarana", 0.80),  # temporal oblique → locus (time is a frame)
    "obl:loc": ("adhikarana", 0.95),  # locative oblique → adhikaraṇa (location)
    "obl:agent": ("karta", 0.85),  # agent oblique (passive) → kartā
    "nmod": ("visesana", 0.60),  # nominal modifier → viśeṣaṇa (adjective/descriptor)
    "amod": ("visesana", 0.90),  # adjectival modifier → viśeṣaṇa
    "acl": ("visesana", 0.70),  # adjectival clause → viśeṣaṇa
    "advmod": ("kriya", 0.65),  # adverbial modifier → kriyā (modifies action)
    "case": ("separator", 0.80),  # case-marking → separator (postpositions)
    "aux": ("kriya", 0.75),  # auxiliary → kriyā (auxiliary verb)
    "cop": ("kriya", 0.70),  # copula → kriyā
    "root": ("kriya", 0.95),  # root verb → kriyā (main action)
    "conj": ("kriya", 0.75),  # conjunction → kriyā (coordinated action)
    "compound": ("visesana", 0.70),  # compound → viśeṣaṇa
    "dep": ("unknown", 0.30),  # unspecified → unknown
    "punct": ("separator", 0.90),  # punctuation → separator
    "det": ("separator", 0.85),  # determiner → separator
    "cc": ("separator", 0.70),  # coordinating conjunction → separator
    "mark": ("separator", 0.70),  # subordinating conjunction → separator
}

# Fallback for unmapped deprels
DEFAULT_DEPREL_ROLE = ("unknown", 0.30)


class EnglishDeprelKarakaMapper:
    """Maps English spaCy dependency relations to Pāṇinian kāraka roles.

    Uses spaCy's pre-trained en_core_web_sm model to parse sentences and
    extract universal dependency relations. Each token is assigned a kāraka
    role and confidence score.

    Confidence thresholds allow low-confidence assignments to fall back to
    default masking probabilities.
    """

    def __init__(
        self, model_name: str = "en_core_web_sm", confidence_threshold: float = 0.50
    ) -> None:
        """Initialize the mapper with a spaCy model.

        Args:
            model_name: spaCy model name (must be pre-installed).
            confidence_threshold: Tokens with confidence < this are marked "unknown".
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError as e:
            raise OSError(
                f"spaCy model '{model_name}' not found. Install with: "
                f"python -m spacy download {model_name}"
            ) from e

        self.confidence_threshold = confidence_threshold
        self._cache: dict[str, list[dict[str, Any]]] = {}

    def get_token_roles(
        self,
        doc: Any,  # spacy.Doc
        use_deprel: bool = True,
    ) -> list[dict[str, Any]]:
        """Extract kāraka roles for all tokens in a parsed document.

        Args:
            doc: A parsed spaCy Doc object.
            use_deprel: If False, return "unknown" for all tokens (regression guard).

        Returns:
            List of dicts, one per token: {"role": str, "confidence": float, "deprel": str}
        """
        if not use_deprel:
            return [{"role": "unknown", "confidence": 0.0, "deprel": token.dep_} for token in doc]

        # Check cache
        doc_text = doc.text
        if doc_text in self._cache:
            return self._cache[doc_text]

        result = []
        for token in doc:
            deprel = token.dep_
            role, base_conf = DEPREL_TO_KARAKA.get(deprel, DEFAULT_DEPREL_ROLE)

            # Adjust confidence based on threshold
            if base_conf < self.confidence_threshold:
                role = "unknown"
                confidence = 0.0
            else:
                confidence = base_conf

            result.append(
                {
                    "role": role,
                    "confidence": confidence,
                    "deprel": deprel,
                }
            )

        # Cache the result
        self._cache[doc_text] = result
        return result

    def get_vocab_mask_probs(
        self,
        vocab_size: int,
        default_prob: float = 0.30,
    ) -> torch.Tensor:
        """Build a vocab-size tensor of masking probabilities.

        This is called once per training run to pre-build a vectorized
        lookup table. Unused tokens default to default_prob.

        Args:
            vocab_size: Size of the vocabulary.
            default_prob: Default masking probability for unknown roles.

        Returns:
            Tensor (vocab_size,) of float32 probabilities in [0.0, 1.0].
        """
        # For real deprel mapping, we don't have a pre-built vocab lookup
        # (roles are per-token-instance, not per-token-id).
        # Return all-default as the base; during forward pass, we'll
        # compute actual probs from parsed deprels.
        return torch.full((vocab_size,), fill_value=default_prob, dtype=torch.float32)

    def get_mask_probs_for_ids(
        self,
        ids: torch.Tensor,  # (B, T)
        probs: torch.Tensor,  # (vocab_size,) pre-built vocab probs
        default_prob: float = 0.30,
    ) -> torch.Tensor:
        """Extract masking probabilities for a batch of token IDs.

        This is a placeholder that gathers from a pre-built vocab tensor.
        For real deprel-based masking, the forward pass should instead:
        1. Decode token IDs to text.
        2. Parse with spaCy.
        3. Extract deprel per token.
        4. Compute role-based probs.

        For now, this returns the pre-built vocab lookup (regression guard).

        Args:
            ids: Token IDs (B, T).
            probs: Pre-built vocab probability tensor (vocab_size,).
            default_prob: Default probability for fallback.

        Returns:
            Tensor (B, T) of probabilities.
        """
        # Simple gather from vocab tensor
        return probs.to(ids.device)[ids.long()]
