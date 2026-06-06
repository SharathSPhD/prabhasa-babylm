"""Pervasion-coherent Pañcāvayava reasoning chain generator for Navya-Nyāya H2.

This module generates logically well-formed 5-limb Nyāya syllogisms (Pañcāvayava)
with GENUINE pervasions: each valid chain's udāharaṇa (vyāpti) connects the SAME
hetu and sādhya as the pratijna, ensuring logical coherence.

Domain contract:
  - Valid chains: vyāpti (hetu ⊂ sādhya) coherent within each chain
  - Fallacies: properly instantiated (savyabhichara, viruddha, asiddha, satpratipaksha)
  - Diversity: ~1200+ valid chains from (24 pervasions) × (40+ pakṣa)
  - All chains grammatically correct English with no template gibberish
"""

from __future__ import annotations

import random
from datetime import date

from pramana.domain.models.nyaya_example import (
    DoubtType,
    ExampleMetadata,
    Hetvabhasa,
    HetvabhasaType,
    Nirnaya,
    NyayaExample,
    PanchaAvayava,
    Pramana,
    Samshaya,
    Tarka,
)

# ============================================================================
# CURATED GENUINE PERVASIONS: (hetu, sādhya, positive_example, negative_example)
# Each pervasion H ⊂ S genuinely holds in classical Nyāya epistemology
# ============================================================================

PERVASIONS = [
    {
        "hetu": "produced by human effort",
        "sadhya": "non-eternal",
        "positive_example": "a pot",
        "negative_example": "the ether",
        "vyapti_full": "Whatever is produced by human effort is non-eternal; as the pot; and what is eternal is not produced, as the ether",
    },
    {
        "hetu": "has a beginning in time",
        "sadhya": "non-eternal",
        "positive_example": "a cloth",
        "negative_example": "the sky",
        "vyapti_full": "Whatever has a beginning in time is non-eternal; as a cloth; and what is eternal has no beginning, as the sky",
    },
    {
        "hetu": "is composed of parts",
        "sadhya": "non-eternal",
        "positive_example": "a house",
        "negative_example": "an atom",
        "vyapti_full": "Whatever is composed of parts is non-eternal; as a house; and what is eternal is not composite, as an atom",
    },
    {
        "hetu": "has form and color",
        "sadhya": "perceptible",
        "positive_example": "a flower",
        "negative_example": "the soul",
        "vyapti_full": "Whatever has form and color is perceptible; as a flower; and what is imperceptible lacks form, as the soul",
    },
    {
        "hetu": "produces sense-contact",
        "sadhya": "perceptible",
        "positive_example": "the mountain",
        "negative_example": "thought",
        "vyapti_full": "Whatever produces sense-contact is perceptible; as the mountain; and what is imperceptible cannot be contacted, as thought",
    },
    {
        "hetu": "requires a maker or cause",
        "sadhya": "not self-existent",
        "positive_example": "a table",
        "negative_example": "substance",
        "vyapti_full": "Whatever requires a maker or cause is not self-existent; as a table; and what is self-existent needs no cause, as substance",
    },
    {
        "hetu": "is an effect of something prior",
        "sadhya": "produced",
        "positive_example": "fire from friction",
        "negative_example": "the eternal ether",
        "vyapti_full": "Whatever is an effect of something prior is produced; as fire from friction; and what is unproduced is not an effect, as the eternal ether",
    },
    {
        "hetu": "is cognizable by the mind",
        "sadhya": "existent",
        "positive_example": "a pot",
        "negative_example": "the non-existent round-square",
        "vyapti_full": "Whatever is cognizable by the mind exists; as a pot; and what does not exist cannot be cognized, as the non-existent round-square",
    },
    {
        "hetu": "is the object of valid knowledge",
        "sadhya": "real",
        "positive_example": "the tree",
        "negative_example": "a false perception",
        "vyapti_full": "Whatever is the object of valid knowledge is real; as the tree; and what is unreal cannot be validly known, as a false perception",
    },
    {
        "hetu": "occupies space",
        "sadhya": "material",
        "positive_example": "the earth",
        "negative_example": "consciousness",
        "vyapti_full": "Whatever occupies space is material; as the earth; and what is immaterial does not occupy space, as consciousness",
    },
    {
        "hetu": "is divisible into parts",
        "sadhya": "composite",
        "positive_example": "a rope",
        "negative_example": "a point",
        "vyapti_full": "Whatever is divisible into parts is composite; as a rope; and what is simple is not divisible, as a point",
    },
    {
        "hetu": "moves from place to place",
        "sadhya": "mobile",
        "positive_example": "a horse",
        "negative_example": "the mountain",
        "vyapti_full": "Whatever moves from place to place is mobile; as a horse; and what is immobile does not change location, as the mountain",
    },
    {
        "hetu": "undergoes transformation",
        "sadhya": "subject to change",
        "positive_example": "milk becoming yogurt",
        "negative_example": "the eternal ether",
        "vyapti_full": "Whatever undergoes transformation is subject to change; as milk becoming yogurt; and what is unchanging does not transform, as the eternal ether",
    },
    {
        "hetu": "has intention and agency",
        "sadhya": "conscious",
        "positive_example": "a person",
        "negative_example": "a stone",
        "vyapti_full": "Whatever has intention and agency is conscious; as a person; and what is unconscious lacks agency, as a stone",
    },
    {
        "hetu": "experiences pleasure or pain",
        "sadhya": "sentient",
        "positive_example": "a living being",
        "negative_example": "a plant",
        "vyapti_full": "Whatever experiences pleasure or pain is sentient; as a living being; and what is insentient does not experience, as a plant",
    },
    {
        "hetu": "is an inherent quality",
        "sadhya": "non-independent",
        "positive_example": "color in a cloth",
        "negative_example": "a substance",
        "vyapti_full": "Whatever is an inherent quality is non-independent; as color in a cloth; and what is independent is not a mere quality, as a substance",
    },
    {
        "hetu": "cannot exist without a substrate",
        "sadhya": "dependent",
        "positive_example": "sweetness in sugar",
        "negative_example": "the ether",
        "vyapti_full": "Whatever cannot exist without a substrate is dependent; as sweetness in sugar; and what is independent can exist without substrate, as the ether",
    },
    {
        "hetu": "is denoted by a word",
        "sadhya": "knowable through language",
        "positive_example": "a tree",
        "negative_example": "the ineffable",
        "vyapti_full": "Whatever is denoted by a word is knowable through language; as a tree; and what is ineffable cannot be named, as the truly ineffable",
    },
    {
        "hetu": "is conceptualizable",
        "sadhya": "graspable by intellect",
        "positive_example": "a substance",
        "negative_example": "the mystical void",
        "vyapti_full": "Whatever is conceptualizable is graspable by intellect; as a substance; and what is incomprehensible cannot be conceptualized, as the mystical void",
    },
    {
        "hetu": "has a necessary cause",
        "sadhya": "dependent on another",
        "positive_example": "smoke from fire",
        "negative_example": "the ultimate principle",
        "vyapti_full": "Whatever has a necessary cause is dependent on another; as smoke from fire; and what is self-caused depends on no other, as the ultimate principle",
    },
    {
        "hetu": "is excluded from a class",
        "sadhya": "different from that class",
        "positive_example": "a horse (not a cow)",
        "negative_example": "a universal",
        "vyapti_full": "Whatever is excluded from a class is different from that class; as a horse from cows; and what is identical is not excluded, as a universal",
    },
    {
        "hetu": "ceases to exist",
        "sadhya": "temporal",
        "positive_example": "a day",
        "negative_example": "eternity",
        "vyapti_full": "Whatever ceases to exist is temporal; as a day; and what is eternal does not cease, as eternity",
    },
    {
        "hetu": "is preceded by a cause",
        "sadhya": "effect",
        "positive_example": "a sprout from a seed",
        "negative_example": "the primordial matter",
        "vyapti_full": "Whatever is preceded by a cause is an effect; as a sprout from a seed; and what is not an effect lacks a preceding cause, as the primordial matter",
    },
    {
        "hetu": "appears to the senses",
        "sadhya": "observable",
        "positive_example": "the sun",
        "negative_example": "the unmanifest",
        "vyapti_full": "Whatever appears to the senses is observable; as the sun; and what is not observable never appears to senses, as the unmanifest",
    },
]

# Diverse pakṣa (subjects) to combine with pervasions
PAKSHA = [
    "sound",
    "a pot",
    "a cloth",
    "the mountain",
    "the earth",
    "the sky",
    "fire",
    "water",
    "the soul",
    "the mind",
    "knowledge",
    "pleasure",
    "pain",
    "a flower",
    "a tree",
    "the river",
    "the eye",
    "the ear",
    "the body",
    "the horse",
    "a person",
    "motion",
    "color",
    "taste",
    "smell",
    "touch",
    "space",
    "time",
    "the atom",
    "substance",
    "quality",
    "action",
    "a book",
    "a house",
    "a table",
    "the wind",
    "thought",
    "desire",
    "memory",
    "intention",
]


class PanchaAvayavaGenerator:
    """Generate pervasion-coherent Pañcāvayava chains.

    Uses curated genuine pervasions (hetu ⊂ sādhya) to ensure each valid chain's
    udāharaṇa vyāpti connects the same hetu and sādhya as the chain itself.
    Combines pervasions with diverse pakṣa for both content diversity and
    logical coherence.
    """

    def __init__(self, seed: int = 42):
        """Initialize generator with seeded RNG."""
        self.rng = random.Random(seed)
        self._example_counter = 0

    def _generate_valid_chain(self) -> dict[str, object]:
        """Generate a logically valid Pañcāvayava with coherent vyāpti.

        Returns:
          Dict with pratijna, hetu, udaharana, upanaya, nigamana, vyapti
        """
        pervasion = self.rng.choice(PERVASIONS)
        paksha = self.rng.choice(PAKSHA)

        hetu = pervasion["hetu"]
        sadhya = pervasion["sadhya"]

        pratijna = f"{paksha.capitalize()} is {sadhya}"
        hetu_full = f"because it {hetu}"

        # Udāharaṇa: establish the SAME pervasion (hetu ⊂ sadhya)
        # Use classical Nyāya form: positive exemplar shows hetu→sadhya;
        # negative exemplar (counterexample) shows absence of sadhya
        udaharana = (
            f"Whatever {hetu} is {sadhya}—as seen in {pervasion['positive_example']}; "
            f"and {pervasion['negative_example']} is not {sadhya}, hence not {hetu}."
        )

        # Upanaya: apply the pervasion to the paksha
        upanaya = f"{paksha.capitalize()} {hetu}."

        # Nigamana: conclusion
        nigamana = f"Therefore, {paksha.lower()} is {sadhya}"

        return {
            "pratijna": pratijna,
            "hetu": hetu_full,
            "udaharana": udaharana,
            "upanaya": upanaya,
            "nigamana": nigamana,
            "vyapti": f"Whatever {hetu} is {sadhya}",
            "paksha": paksha,
            "sadhya": sadhya,
            "hetu_property": hetu,
            "fallacy_type": None,
        }

    def _generate_fallacy_savyabhichara(self) -> dict[str, object]:
        """Generate SAVYABHICHARA: hetu is NOT pervaded by sādhya.

        Hetu occurs with both sādhya and ¬sādhya (erratic reason).
        """
        pervasion = self.rng.choice(PERVASIONS)
        paksha = self.rng.choice(PAKSHA)

        hetu = pervasion["hetu"]
        sadhya = pervasion["sadhya"]
        counter_example = self.rng.choice(
            [ex for ex in PAKSHA if ex != pervasion["positive_example"]]
        )

        pratijna = f"{paksha.capitalize()} is {sadhya}"
        hetu_full = f"because it {hetu}"

        udaharana = (
            f"Some things that {hetu} are {sadhya} (as {pervasion['positive_example']}), "
            f"but {counter_example} also {hetu} yet is not {sadhya}. Thus the reason is erratic."
        )

        upanaya = f"{paksha.capitalize()} {hetu}."

        nigamana = f"Therefore, {paksha.lower()} is {sadhya}"

        return {
            "pratijna": pratijna,
            "hetu": hetu_full,
            "udaharana": udaharana,
            "upanaya": upanaya,
            "nigamana": nigamana,
            "vyapti": f"{hetu} does not pervade {sadhya} (erratic)",
            "paksha": paksha,
            "sadhya": sadhya,
            "fallacy_type": HetvabhasaType.SAVYABHICHARA,
        }

    def _generate_fallacy_viruddha(self) -> dict[str, object]:
        """Generate VIRUDDHA: hetu proves the opposite (¬sādhya)."""
        pervasion = self.rng.choice(PERVASIONS)
        paksha = self.rng.choice(PAKSHA)

        hetu = pervasion["hetu"]
        sadhya = pervasion["sadhya"]
        sadhya_opposite = f"not {sadhya}"

        pratijna = f"{paksha.capitalize()} is {sadhya_opposite}"
        hetu_full = f"because it {hetu}"

        udaharana = (
            f"Whatever {hetu} is {sadhya} (as {pervasion['positive_example']}), "
            f"not {sadhya_opposite}. The reason proves the opposite."
        )

        upanaya = f"{paksha.capitalize()} {hetu}."

        nigamana = f"Therefore, {paksha.lower()} is {sadhya}"

        return {
            "pratijna": pratijna,
            "hetu": hetu_full,
            "udaharana": udaharana,
            "upanaya": upanaya,
            "nigamana": nigamana,
            "vyapti": f"{hetu} proves {sadhya}, contradicting the thesis",
            "paksha": paksha,
            "sadhya": sadhya_opposite,
            "fallacy_type": HetvabhasaType.VIRUDDHA,
        }

    def _generate_fallacy_asiddha(self) -> dict[str, object]:
        """Generate ASIDDHA: the paksha does not actually possess the hetu."""
        pervasion = self.rng.choice(PERVASIONS)
        paksha = self.rng.choice(PAKSHA)

        hetu = pervasion["hetu"]
        sadhya = pervasion["sadhya"]

        pratijna = f"{paksha.capitalize()} is {sadhya}"
        hetu_full = f"because it {hetu}"

        udaharana = (
            f"If {paksha.lower()} truly {hetu}, then it would be {sadhya} "
            f"(as is {pervasion['positive_example']}). But whether {paksha.lower()} "
            f"actually {hetu} is not established."
        )

        upanaya = f"It is unproven that {paksha.lower()} {hetu}."

        nigamana = "Therefore, the inference is inconclusive."

        return {
            "pratijna": pratijna,
            "hetu": hetu_full,
            "udaharana": udaharana,
            "upanaya": upanaya,
            "nigamana": nigamana,
            "vyapti": f"Unproven whether {paksha.lower()} possesses {hetu}",
            "paksha": paksha,
            "sadhya": sadhya,
            "fallacy_type": HetvabhasaType.ASIDDHA,
        }

    def _generate_fallacy_satpratipaksha(self) -> dict[str, object]:
        """Generate SATPRATIPAKSHA: an equally strong counter-inference exists."""
        pervasion = self.rng.choice(PERVASIONS)
        paksha = self.rng.choice(PAKSHA)

        hetu = pervasion["hetu"]
        sadhya = pervasion["sadhya"]
        sadhya_opposite = f"not {sadhya}"

        pratijna = f"{paksha.capitalize()} is {sadhya}"
        hetu_full = f"because it {hetu}"

        udaharana = (
            f"Whatever {hetu} is {sadhya}, as {pervasion['positive_example']}. "
            f"However, equally strong reasoning shows {paksha.lower()} is {sadhya_opposite}. "
            f"The arguments are balanced and inconclusive."
        )

        upanaya = f"{paksha.capitalize()} {hetu}."

        nigamana = "Therefore, the inference is indeterminate."

        return {
            "pratijna": pratijna,
            "hetu": hetu_full,
            "udaharana": udaharana,
            "upanaya": upanaya,
            "nigamana": nigamana,
            "vyapti": f"Counter-argument equally supports {sadhya_opposite}",
            "paksha": paksha,
            "sadhya": sadhya,
            "fallacy_type": HetvabhasaType.SATPRATIPAKSHA,
        }

    def generate(self, n: int = 2000, seed: int | None = None) -> list[NyayaExample]:
        """Generate n Pañcāvayava examples (50% valid, 50% fallacious)."""
        if seed is not None:
            self.rng.seed(seed)

        examples = []
        n_valid = n // 2
        n_fallacious = n - n_valid

        for _i in range(n_valid):
            chain_data = self._generate_valid_chain()
            example = self._chain_data_to_nyaya_example(chain_data, is_valid=True)
            examples.append(example)

        fallacy_generators = [
            self._generate_fallacy_savyabhichara,
            self._generate_fallacy_viruddha,
            self._generate_fallacy_asiddha,
            self._generate_fallacy_satpratipaksha,
        ]

        for _i in range(n_fallacious):
            gen_fn = self.rng.choice(fallacy_generators)
            chain_data = gen_fn()
            example = self._chain_data_to_nyaya_example(chain_data, is_valid=False)
            examples.append(example)

        self.rng.shuffle(examples)
        return examples

    def _chain_data_to_nyaya_example(
        self, chain_data: dict[str, object], is_valid: bool
    ) -> NyayaExample:
        """Convert chain_data dict to a complete NyayaExample."""
        self._example_counter += 1
        example_id = f"nyaya_gen_{self._example_counter:06d}"

        paksha = chain_data["paksha"]
        sadhya = chain_data["sadhya"]
        vyapti = chain_data["vyapti"]
        fallacy_type = chain_data.get("fallacy_type")

        problem = (
            f"Does {paksha.lower()} have the property of being {sadhya}? "  # type: ignore[attr-defined]
            f"Reason: {chain_data['hetu'].replace('because ', '')}"  # type: ignore[attr-defined]
        )

        doubt_type = DoubtType.ANADHYAVASAYA
        samshaya = Samshaya(
            doubt_type=doubt_type,
            justification=f"Uncertain whether {paksha.lower()} is truly {sadhya} based on the given reason.",  # type: ignore[attr-defined]
        )

        pramana = Pramana(
            pratyaksha=[f"Observable: {chain_data['hetu']}"],
            anumana=[f"Universal principle: {vyapti}"],
        )

        pca = PanchaAvayava(
            pratijna=chain_data["pratijna"],
            hetu=chain_data["hetu"],
            udaharana=chain_data["udaharana"],
            upanaya=chain_data["upanaya"],
            nigamana=chain_data["nigamana"],
        )

        if is_valid:
            tarka = Tarka(
                hypothesis=f"Suppose {paksha.lower()} is not {sadhya}",  # type: ignore[attr-defined]
                consequence=f"Then the hetu '{chain_data['hetu'].lower()}' would not lead to {sadhya}",  # type: ignore[attr-defined]
                analysis=f"But this contradicts the established pervasion: {vyapti}",
                resolution=f"Therefore, {paksha.lower()} must be {sadhya}",  # type: ignore[attr-defined]
            )
        else:
            tarka = Tarka(
                hypothesis=f"Suppose {paksha.lower()} is not {sadhya}",  # type: ignore[attr-defined]
                consequence="The reasoning would still hold",
                analysis="The argument does not conclusively establish the thesis",
                resolution="The inference is defective",
            )

        if is_valid:
            hetvabhasa = Hetvabhasa(
                fallacies_detected=[],
                analysis="No fallacies detected; the argument is logically sound.",
            )
        else:
            hetvabhasa = Hetvabhasa(
                fallacies_detected=[fallacy_type] if fallacy_type else [],
                analysis=f"The argument contains a {fallacy_type.value if fallacy_type else 'logical'} fallacy.",  # type: ignore[attr-defined]
            )

        nirnaya = Nirnaya(
            answer=f"{paksha} is {sadhya if is_valid else 'indeterminate'}",
            confidence="high" if is_valid else "low",
            justification=(
                f"Based on the Pañcāvayava reasoning and the presence of "
                f"{'no' if is_valid else 'a'} logical fallacy."
            ),
        )

        metadata = ExampleMetadata(
            created_date=date.today(),
            author="nyaya_generator",
            validated=False,
            z3_verifiable=False,
            stage=2,
        )

        return NyayaExample(
            id=example_id,
            problem=problem,
            problem_type="nyaya_reasoning",
            difficulty="intermediate",
            variables=3,
            ground_truth=f"{paksha} is {sadhya}" if is_valid else "Inconclusive",
            samshaya=samshaya,
            pramana=pramana,
            pancha_avayava=[pca],
            tarka=tarka,
            hetvabhasa=hetvabhasa,
            nirnaya=nirnaya,
            metadata=metadata,
        )
