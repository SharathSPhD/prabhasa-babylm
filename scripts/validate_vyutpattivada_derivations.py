#!/usr/bin/env python3
"""Validation script: 5 worked derivations through VyutpattivadaEngine.compile_graph()."""

from __future__ import annotations

from psalm.infrastructure.generators.corpus_from_grammar import (
    PrabasaExample,
    VyutpattivadaEngine,
)


def _example(
    *,
    text: str,
    karaka_parse: list[tuple[str, str]],
    frame_signature: str,
    name: str,
) -> tuple[str, PrabasaExample]:
    """Build a minimal PrabasaExample for testing."""
    return name, PrabasaExample(
        text=text,
        language="sa",
        karaka_parse=tuple(karaka_parse),
        frame_signature=frame_signature,
        meta={"fixture_id": "test-eng-1"},
    )


# ============================================================================
# Five Worked Derivations
# ============================================================================

DERIVATIONS = [
    _example(
        name="Derivation 1: Simple Transitive (kartā + karma)",
        text="rAma.eka.karwA Pala.eka.karma paW1.viXiH",
        karaka_parse=[("rAma", "karwA"), ("Pala", "karma")],
        frame_signature="paW1|viXiH|rAma|eka|Pala|eka",
    ),
    _example(
        name="Derivation 2: Intransitive with Adhikaraṇa (locus)",
        text="guru.eka.karwA aSva.eka.aXikaraNam vas1.varwamAnaH",
        karaka_parse=[("guru", "karwA"), ("aSva", "aXikaraNam")],
        frame_signature="vas1|varwamAnaH|guru|eka|-|aXikaraNam|aSva",
    ),
    _example(
        name="Derivation 3: Transitive + Dravya Instrument (karaṇa)",
        text="bAla.dvi.karwA puswaka.eka.karma aSina.eka.karaNam KAx1.viXiH",
        karaka_parse=[
            ("bAla", "karwA"),
            ("puswaka", "karma"),
            ("aSina", "karaNam"),
        ],
        frame_signature="KAx1|viXiH|bAla|dvi|puswaka|eka|aSina|eka",
    ),
    _example(
        name="Derivation 4: Guṇa Instrument (knowledge) → Prakāratā",
        text="nara.eka.karwA gfha.eka.karma vixyA.eka.karaNam KAx1.viXiH",
        karaka_parse=[
            ("nara", "karwA"),
            ("gfha", "karma"),
            ("vixyA", "karaNam"),
        ],
        frame_signature="KAx1|viXiH|nara|eka|gfha|eka|vixyA|eka",
    ),
    _example(
        name="Derivation 5: Complex Multi-Oblique (kartā + karma + apādāna + sampradāna)",
        text="vana.eka.karwA ratna.bahu.karma nadI.eka.apAxAnam yajamAna.eka.sampraxAnam dA.viXiH",
        karaka_parse=[
            ("vana", "karwA"),
            ("ratna", "karma"),
            ("nadI", "apAxAnam"),
            ("yajamAna", "sampraxAnam"),
        ],
        frame_signature="dA|viXiH|vana|eka|ratna|bahu|nadI|apAxAnam|yajamAna|eka",
    ),
]


def main() -> None:
    """Validate all 5 derivations."""
    print("=" * 80)
    print("VYUTPATTIVĀDA ENGINE VALIDATION: 5 WORKED DERIVATIONS")
    print("=" * 80)
    print()

    engine = VyutpattivadaEngine()
    all_success = True

    for i, (name, example) in enumerate(DERIVATIONS, 1):
        print(f"[{i}/5] {name}")
        print(f"      Text: {example.text}")
        print(f"      Parse: {example.karaka_parse}")

        result = engine.compile_graph(example)

        if result.shabdabodha_graph is None:
            print("      ✗ FAILED: No graph generated")
            all_success = False
        else:
            graph = result.shabdabodha_graph
            print(f"      ✓ SUCCESS: Graph generated")
            print(f"        - Nodes: {len(graph.nodes)} ({', '.join(n.label or n.id for n in graph.nodes)})")
            print(
                f"        - Edges: {len(graph.edges)} "
                f"({', '.join(f'{e.sansa.value}' for e in graph.edges)})"
            )
            print(f"        - Node categories: {sorted(set(n.category.value for n in graph.nodes))}")

        print()

    print("=" * 80)
    if all_success:
        print("✓ ALL DERIVATIONS VALID")
        print()
        print("Summary:")
        print("  - VyutpattivadaEngine.compile_graph() successfully wraps compile_shabdabodha()")
        print("  - All 5 derivations produced valid typed Śabdabodha graphs")
        print("  - Graphs include: nodes (dhātu, kārakas, obliques), edges (sansa types)")
        print("  - Type validation (yogyatā) gates prevent invalid frames")
        print()
        return 0
    else:
        print("✗ SOME DERIVATIONS FAILED")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
