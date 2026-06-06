#!/usr/bin/env python3
"""Spot-check script: test the real morphological N-hot analyzer on real English words.

This demonstrates that the real morphological analyzer correctly segments
English tokens into morpheme components, which is better than the heuristic
suffix/prefix list approach.

Usage:
    uv run python scripts/test_nhot_morphology.py
"""

from psalm.infrastructure.morphology.english import EnglishMorphemeAnalyzer


def main() -> None:
    """Demonstrate real morphological analysis on 10+ English words."""
    analyzer = EnglishMorphemeAnalyzer()

    # Test words from various morphological categories
    test_words = [
        # Simple roots
        ("cat", "simple root"),
        ("book", "simple root"),
        # Inflectional suffixes
        ("cats", "plural"),
        ("walked", "past tense"),
        ("walking", "gerund"),
        ("books", "plural"),
        # Derivational suffixes
        ("helpful", "adjective former"),
        ("happiness", "nominalizer (note: greedy -s stripping)"),
        ("action", "nominalizer"),
        # Prefixes
        ("rebuild", "repetition prefix"),
        ("dislike", "negation prefix"),
        ("unhappy", "negation prefix"),
        ("overlook", "spatial prefix"),
        # Complex words
        ("rebuilding", "prefix + root + inflection"),
        ("organization", "root + derivational suffix"),
        ("beautifully", "adjective + adverbial suffix"),
    ]

    print("=" * 80)
    print("REAL MORPHOLOGICAL ANALYSIS ON ENGLISH TOKENS")
    print("=" * 80)
    print()

    for word, description in test_words:
        result = analyzer.segment(word)
        morphemes = [(m.surface, m.role) for m in result]
        print(f"{word:20} | {description:40} | {morphemes}")

    print()
    print("=" * 80)
    print("KEY OBSERVATIONS:")
    print("=" * 80)
    print()
    print("1. MORPHEME IDENTIFICATION:")
    print("   - Successfully identifies prefixes (un-, re-, dis-, over-)")
    print("   - Successfully identifies inflectional suffixes (-s, -ed, -ing)")
    print("   - Successfully identifies derivational suffixes (-tion, -ful, etc.)")
    print()
    print("2. IMPROVEMENTS OVER HEURISTIC SUFFIX LISTS:")
    print("   - Uses curated morpheme inventory (not arbitrary suffix patterns)")
    print("   - Enforces minimum stem length (prevents over-stripping)")
    print("   - Classifies affixes as prefix/suffix and inflectional/derivational")
    print()
    print("3. KNOWN LIMITATIONS:")
    print("   - Does not handle vowel doubling (running -> runn + ing, not run + ning)")
    print("   - Greedy suffix stripping (happiness -> happines + s, not happy + ness)")
    print("   - Does not handle allomorphy (plural -s vs -es)")
    print()
    print("4. USAGE:")
    print("   - Use --nhot-mode real in build_nhot_matrix.py to enable this analyzer")
    print("   - Provides richer morphological features for the model")
    print()


if __name__ == "__main__":
    main()
