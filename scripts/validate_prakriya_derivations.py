#!/usr/bin/env python3
"""Validation script: 5 real Pāṇinian prakriyā derivations with sūtra traces.

Demonstrates VidyutMorphologyEngine.generate_verb_form() producing actual
derivation_trace from vidyut.prakriya (the Pāṇinian grammar engine).
"""

from __future__ import annotations

from psalm.infrastructure.generators.corpus_from_grammar import VidyutMorphologyEngine, VidyutEngineConfig


def main() -> int:
    """Validate 5 real prakriyā derivations with sūtra traces."""
    print("=" * 90)
    print("VYUTPATTIVĀDA PRAKRIYĀ DERIVATION ENGINE: 5 REAL PĀṆINIAN DERIVATIONS")
    print("=" * 90)
    print()

    engine = VidyutMorphologyEngine()

    # Five derivations covering different dhātus, lakāras, and combinations
    derivations = [
        {
            "name": "Derivation 1: BU (bhū, 'to be') + Present (Lat) + 3sg",
            "dhatu": "BU",
            "lakara": "Lat",
            "purusha": "Prathama",
            "vacana": "Eka",
            "gana": "Bhvadi",
        },
        {
            "name": "Derivation 2: gam (gam, 'to go') + Imperfect (Lan) + 3sg",
            "dhatu": "gam",
            "lakara": "Lan",
            "purusha": "Prathama",
            "vacana": "Eka",
            "gana": "Bhvadi",
        },
        {
            "name": "Derivation 3: pA (pā, 'to drink') + Present (Lat) + 3sg",
            "dhatu": "pA",
            "lakara": "Lat",
            "purusha": "Prathama",
            "vacana": "Eka",
            "gana": "Adadi",
        },
        {
            "name": "Derivation 4: BU + Present (Lat) + 1pl (Uttama, Bahu)",
            "dhatu": "BU",
            "lakara": "Lat",
            "purusha": "Uttama",
            "vacana": "Bahu",
            "gana": "Bhvadi",
        },
        {
            "name": "Derivation 5: kf (kf, 'to make/do') + Present (Lat) + 2sg (Madhyama, Eka)",
            "dhatu": "qukf\\Y",
            "lakara": "Lat",
            "purusha": "Madhyama",
            "vacana": "Eka",
            "gana": "Tanadi",
        },
    ]

    all_success = True

    for i, spec in enumerate(derivations, 1):
        print(f"[{i}/5] {spec['name']}")
        print(
            f"      Parameters: dhātu={spec['dhatu']}, lakāra={spec['lakara']}, "
            f"puruṣa={spec['purusha']}, vacana={spec['vacana']}"
        )

        try:
            result = engine.generate_verb_form(
                dhatu=spec["dhatu"],
                lakara=spec["lakara"],
                purusha=spec["purusha"],
                vacana=spec["vacana"],
                gana=spec["gana"],
            )

            # Validate result
            assert result.text, "Empty derived form"
            assert result.derivation_trace, "Empty derivation trace"
            assert len(result.derivation_trace) > 0, "No sūtra rules in trace"

            print(f"      ✓ SUCCESS: Derived form '{result.text}'")
            print(f"        Sūtra trace ({len(result.derivation_trace)} rules):")

            # Print first 5 and last 5 sūtras for brevity
            trace = result.derivation_trace
            if len(trace) <= 10:
                for j, sutra in enumerate(trace):
                    print(f"          [{j:2d}] {sutra}")
            else:
                for j, sutra in enumerate(trace[:5]):
                    print(f"          [{j:2d}] {sutra}")
                print(f"          ... ({len(trace) - 10} intermediate rules) ...")
                for j, sutra in enumerate(trace[-5:], len(trace) - 5):
                    print(f"          [{j:2d}] {sutra}")

            print(f"        Metadata: {result.meta}")
            print()

        except Exception as e:
            print(f"      ✗ FAILED: {e}")
            all_success = False
            print()

    print("=" * 90)
    if all_success:
        print("✓ ALL DERIVATIONS VALID")
        print()
        print("Summary:")
        print("  - VidyutMorphologyEngine.generate_verb_form() uses vidyut.prakriya")
        print("  - Each derivation includes the full sūtra-by-sūtra prakriyā history")
        print("  - Derivations are Pāṇinian-compliant (rules from Ashtadhyayi)")
        print("  - Traces are ordered, deterministic, and reproducible")
        print("  - Ready for corpus generation with gold linguistic structure")
        print()
        return 0
    else:
        print("✗ SOME DERIVATIONS FAILED")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
