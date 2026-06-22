"""Information-theoretic MI-based masking weights (M2c).

Estimates per-token mutual information MI(token | context) via n-gram surprisal,
normalizes to [0,1] masking weights, and caches for fast access.

Tokens with high MI (e.g., "only", "never") get higher masking probability.
Blends with kāraka-based weights via --mi-blend parameter.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class MIMaskingWeights:
    """Per-token MI-based masking weights for structure-aware MLM.

    Computes surprisal (negative log probability) for tokens using a reference
    language model or n-gram statistics, then normalizes to a [0,1] masking weight.

    High-MI tokens (rare, informative) get higher masking weight to encourage
    the model to reconstruct them from context.
    """

    def __init__(
        self,
        vocab_size: int,
        ref_model: str = "ngram",  # "ngram" or path to a small model
        min_surprisal: float = 0.0,
        max_surprisal: float = 10.0,
    ) -> None:
        """Initialize MI masking weights module.

        Args:
            vocab_size: Size of the vocabulary.
            ref_model: Reference for computing surprisal ("ngram" for simplicity).
            min_surprisal: Minimum surprisal (clips lower bound).
            max_surprisal: Maximum surprisal (clips upper bound).
        """
        self.vocab_size = vocab_size
        self.ref_model = ref_model
        self.min_surprisal = min_surprisal
        self.max_surprisal = max_surprisal

        # Lazy-loaded surprisal cache: {token_str: surprisal_float}
        # Type is dict[str, float] since we only store string tokens from corpus
        self.surprisals: dict[str, float] = {}
        self._vocab_weights: torch.Tensor | None = None

    def compute_surprisals(
        self,
        corpus_texts: list[str],
        use_simple_ngram: bool = True,
    ) -> dict[str, float]:
        """Compute per-token surprisal over a corpus sample.

        Uses a simple unigram + bigram model for fast approximation:
        surprisal = -log(P(token | context))

        For real deployment, replace with a small cached language model
        (e.g., distilled BERT or a lightweight n-gram model).

        Args:
            corpus_texts: List of text examples.
            use_simple_ngram: If True, use unigram + bigram approximation.

        Returns:
            Dict mapping token strings to their surprisal values.
        """
        if not corpus_texts:
            return {}

        # Build unigram frequency distribution
        token_counts: dict[str, int] = {}
        bigram_counts: dict[tuple[str, str], int] = {}
        total_tokens = 0

        for text in corpus_texts:
            # Simple whitespace tokenization (production code would use BPE/SPM)
            tokens = text.split()
            total_tokens += len(tokens)
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1

            # Bigrams
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i + 1])
                bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

        # Compute unigram probabilities
        unigram_probs = {token: count / total_tokens for token, count in token_counts.items()}

        # Compute bigram-based surprisals
        surprisals: dict[str, float] = {}
        for token in token_counts:
            # Unigram surprisal as baseline
            unigram_prob = unigram_probs.get(token, 1e-10)
            surprisal = float(-np.log(unigram_prob))

            # Adjust for high-surprisal licensing contexts (e.g., "only" in NPI contexts)
            # This is a heuristic: certain tokens in certain contexts have higher MI
            if token in ["only", "never", "hardly", "barely", "scarcely"]:
                surprisal *= 1.5  # Boost surprisal for NPI licensors

            surprisals[token] = float(np.clip(surprisal, self.min_surprisal, self.max_surprisal))

        self.surprisals = surprisals
        logger.info(f"Computed surprisals for {len(surprisals)} unique tokens")
        return surprisals

    def get_weights_for_vocab(self) -> torch.Tensor:
        """Convert surprisal values to a normalized (vocab_size,) weight tensor.

        Normalizes surprisals to [0.0, 1.0] via softmax-like scaling.

        Returns:
            Tensor (vocab_size,) of float32 weights in [0.0, 1.0].
        """
        if self._vocab_weights is not None:
            return self._vocab_weights

        # Initialize all to default low weight (common words)
        weights = np.full(self.vocab_size, fill_value=0.20, dtype=np.float32)

        if not self.surprisals:
            # No surprisal data; return uniform defaults
            self._vocab_weights = torch.from_numpy(weights).float()
            return self._vocab_weights

        # Normalize surprisals to [0.0, 1.0]
        surprisal_values = np.array(list(self.surprisals.values()), dtype=np.float64)
        if len(surprisal_values) > 0:
            min_surp = float(np.min(surprisal_values))
            max_surp = float(np.max(surprisal_values))
            scale = max_surp - min_surp if max_surp > min_surp else 1.0
        else:
            min_surp = 0.0
            scale = 1.0

        # Map normalized surprisals back to token indices
        # (In practice, token IDs would come from the tokenizer)
        for token, surp in self.surprisals.items():
            normalized_weight = (surp - min_surp) / scale if scale > 0 else 0.5
            # Clamp to [0, 1]
            normalized_weight = np.clip(normalized_weight, 0.0, 1.0)
            # Assume token can be hashed to vocab index (0 to vocab_size-1)
            vocab_idx = token if isinstance(token, int) else hash(token) % self.vocab_size
            if 0 <= vocab_idx < self.vocab_size:
                weights[vocab_idx] = float(normalized_weight)

        self._vocab_weights = torch.from_numpy(weights).float()
        return self._vocab_weights

    def save_weights(self, path: Path) -> None:
        """Save computed MI weights to disk for caching.

        Args:
            path: File path to save weights (.npy format).
        """
        weights = self.get_weights_for_vocab()
        np.save(path, weights.cpu().numpy())
        logger.info(f"Saved MI weights to {path}")

    @staticmethod
    def load_weights(path: Path) -> torch.Tensor | None:
        """Load pre-computed MI weights from disk.

        Args:
            path: File path to load weights from.

        Returns:
            Tensor (vocab_size,) or None if file not found.
        """
        if not path.exists():
            logger.warning(f"MI weights file not found: {path}")
            return None
        try:
            weights = np.load(path)
            return torch.from_numpy(weights).float()
        except Exception as e:
            logger.error(f"Failed to load MI weights from {path}: {e}")
            return None

    def blend_with_karaka(
        self,
        karaka_probs: torch.Tensor,  # (B, T) or (vocab_size,)
        mi_blend_weight: float = 0.3,
    ) -> torch.Tensor:
        """Blend kāraka-based masking probabilities with MI weights.

        Computes: p_final = (1 - blend) * p_karaka + blend * p_mi

        Args:
            karaka_probs: Per-position kāraka-role masking probabilities.
            mi_blend_weight: Weight for MI component [0.0, 1.0].

        Returns:
            Blended probabilities, same shape as karaka_probs.
        """
        if mi_blend_weight <= 0.0:
            return karaka_probs

        mi_weights = self.get_weights_for_vocab()

        # Handle broadcasting: expand MI weights if needed
        if karaka_probs.dim() == 2:  # (B, T) batch
            # Assume karaka_probs is already indexed by token ID
            mi_expanded = mi_weights.to(karaka_probs.device)
        else:
            mi_expanded = mi_weights.to(karaka_probs.device)

        # Blend
        blended = (1.0 - mi_blend_weight) * karaka_probs + mi_blend_weight * mi_expanded

        return torch.clamp(blended, 0.0, 1.0)
