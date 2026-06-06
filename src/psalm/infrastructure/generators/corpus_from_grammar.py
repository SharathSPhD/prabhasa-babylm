"""PRABHĀSA: Corpus-from-Grammar generation pipeline (Phase 3).

Composes five deterministic linguistic engines to generate unbounded,
structurally-and-semantically-labeled Sanskrit training corpora:

1. Vidyut — morphology & derivation (existing, `vidyut_source.py`)
2. Paribhāṣā — rule-ordered derivation (new, `ParibhashaRuleInterpreter`)
3. Śabdabodha — kāraka-aware sentence assembly (extends `shabdabodha_compile`)
4. Vyutpattivāda — compositional semantic graphs (new, `VyutpattivadaEngine`)
5. Navya-Nyāya — validity labels & augmentation (stub, Phase 4)

Output: `PrabasaExample` records with five gold labels, exported to
`paribhasha_aligned_v1` JSONL via `to_aligned_record()`.

This is the scalability moat: grammar-guaranteed quality, unbounded generation,
deterministic reproducibility. See `docs/memory/corpus_from_grammar_design.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from psalm.application.data.ports import AnnotatedSentence
from psalm.infrastructure.generators.paribhasha.types import ShabdabodhaGraph


class FrameType(StrEnum):
    """Kāraka frame categories (classical Pāṇinian classification)."""

    AKARMAKA = "akarmaka"  # Intransitive: kartā only
    TRANSITIVE = "transitive"  # Transitive: kartā + karma
    MULTI_OBLIQUE = "multi_oblique"  # Multi-slot: kartā + karma + obliques


class MorphemeType(StrEnum):
    """Morphological segment classification (Pāṇinian)."""

    ROOT = "root"  # Dhātu
    AFFIX = "affix"  # Pratyaya
    DESINENCE = "desinence"  # Tiṅ or Sup ending
    AUGMENT = "augment"  # Augment (ā-)


class RuleClass(StrEnum):
    """Paribhāṣā rule classification (rule types)."""

    SUBSTITUTION = "substitution"  # ādeśa
    COMBINATION = "combination"  # Agama
    SUPPRESSION = "suppression"  # Lopa
    MARKER = "marker"  # Anubandha


class ValidityLabel(StrEnum):
    """Navya-Nyāya semantic validity (tristate)."""

    VALID = "valid"  # Type-safe, no fallacy
    MARGINAL = "marginal"  # Type-safe, mild inconsistency
    FALLACIOUS = "fallacious"  # Violates yogyatā or vyāpti


@dataclass(frozen=True)
class MorphemeSegment:
    """One morpheme in a derivation stream."""

    text: str  # Segment text (e.g., "bhu", "v", "a")
    morpheme_type: MorphemeType  # Classification
    supra_index: int | None = None  # Optional Pāṇinian supra-numeral marker


@dataclass(frozen=True)
class MorphemeBoundary:
    """Character-level boundary in surface text."""

    start: int  # Start character offset
    end: int  # End character offset
    boundary_type: str  # "morpheme" | "sandhi_junction" | "word"


@dataclass(frozen=True)
class DerivationStep:
    """One rule application in the derivation."""

    sūtra: str  # Rule id (e.g., "2.4.82")
    rule_class: RuleClass  # Substitution, combination, suppression, marker
    paribhasha_class: str | None = None  # e.g., "sthānivat" (stability)
    semantic_effect: str | None = None  # Prose description


@dataclass(frozen=True)
class PrabasaExample:
    """Complete PRABHĀSA corpus record from all five engines.

    Attributes:
        text: Surface Sanskrit sentence (WX or IAST).
        language: ISO 639 code ("sa" for Sanskrit).

        morpheme_boundaries: Character-level boundaries (Vidyut).
        morpheme_stream: Ordered morpheme segments (Vidyut).
        derivation_trace: Pāṇinian sūtra ids (Vidyut).

        paribhasha_rule_classes: Sūtra → rule class mapping (Paribhāṣā).

        karaka_parse: Token–role pairs, surface order (Śabdabodha).
        frame_signature: "dhatu|gana|frame_type" (Śabdabodha).

        shabdabodha_graph: Typed semantic graph (Vyutpattivāda).

        nyaya_validity_label: "valid" | "marginal" | "fallacious" (Navya-Nyāya).
        nyaya_semantic_augmentation: Vyāpti, fallacy labels (Navya-Nyāya, optional).

        meta: Auxiliary metadata (seed, source, etc.).
    """

    # Surface text
    text: str
    language: str = "sa"

    # Vidyut: morphology (gold from grammar generator)
    morpheme_boundaries: list[MorphemeBoundary] = field(default_factory=list)
    morpheme_stream: list[MorphemeSegment] = field(default_factory=list)
    derivation_trace: tuple[str, ...] = ()  # Sūtra ids in order

    # Paribhāṣā: rule interpretation
    paribhasha_rule_classes: dict[str, RuleClass] = field(default_factory=dict)

    # Śabdabodha: sentence structure & kāraka
    karaka_parse: tuple[tuple[str, str], ...] = ()  # (token, role) pairs
    frame_signature: str = ""  # "dhatu|gana|frame_type"

    # Vyutpattivāda: semantic graph (typed)
    shabdabodha_graph: ShabdabodhaGraph | None = None

    # Navya-Nyāya: validity & augmentation
    nyaya_validity_label: ValidityLabel | None = None
    nyaya_semantic_augmentation: dict[str, Any] = field(default_factory=dict)

    # Metadata
    meta: dict[str, Any] = field(default_factory=dict)

    def to_aligned_record(self) -> dict[str, Any]:
        """Export to paribhasha_aligned_v1 schema (see docs/contracts/aligned-pair-schema.json).

        Raises:
            ValueError: if required fields (text, karaka_parse, shabdabodha_graph) are missing.

        Returns:
            JSON-serializable dict conforming to paribhasha_aligned_v1.
        """
        raise NotImplementedError(
            "to_aligned_record is a stub. Implementation depends on "
            "pramana.infrastructure.storage.paribhasha_export module."
        )


# ============================================================================
# Pipeline Stages (stubs for Phase 3 implementation)
# ============================================================================


@dataclass(frozen=True)
class VidyutEngineConfig:
    """Configuration for Vidyut morphology generator."""

    dhatus: tuple[tuple[str, str], ...] = (
        ("BU", "Bhvadi"),
        ("gam", "Bhvadi"),
    )
    lakaras: tuple[str, ...] = ("Lat", "Lit")
    include_derivation: bool = True


class VidyutMorphologyEngine:
    """Morphology generator using vidyut.prakriya (Pāṇinian derivation engine)."""

    def __init__(self, config: VidyutEngineConfig | None = None) -> None:
        """Initialize with config; lazily loads vidyut on first call."""
        self.config = config or VidyutEngineConfig()
        self._vyakarana = None

    def _get_vyakarana(self) -> object:
        """Lazily load Vyakarana instance."""
        if self._vyakarana is None:
            try:
                from vidyut.prakriya import Vyakarana
            except ImportError as exc:
                raise RuntimeError(
                    "vidyut is not installed. Install it: `uv pip install vidyut`."
                ) from exc
            self._vyakarana = Vyakarana()  # type: ignore[no-untyped-call,assignment]
        return self._vyakarana

    def generate_verb_form(
        self, dhatu: str, lakara: str, purusha: str, vacana: str, gana: str = "Bhvadi"
    ) -> PrabasaExample:
        """Generate one tiṅanta (finite verb) with full Pāṇinian derivation trace.

        Uses vidyut.prakriya to derive the form and capture the ordered sūtra-by-sūtra
        rule applications (prakriyā history) as the derivation_trace.

        Args:
            dhatu: Verbal root in SLP1 (e.g., "BU").
            lakara: Tense/mood (e.g., "Lat" for present).
            purusha: Person (e.g., "Prathama" for 3rd).
            vacana: Number (e.g., "Eka" for singular).
            gana: Verbal class (e.g., "Bhvadi" for class 1).

        Returns:
            PrabasaExample with:
            - text: Derived verb form (SLP1)
            - derivation_trace: Tuple of sūtra codes (e.g., ("3.2.123", "7.3.84", ...))
            - meta: dhatu, gana, lakara, purusha, vacana for reference

        Raises:
            ValueError: if derivation fails or produces no results.
        """
        from vidyut.prakriya import Dhatu, Gana, Lakara, Pada, Prayoga, Purusha, Vacana  # type: ignore[attr-defined] # noqa: I001

        vyakarana = self._get_vyakarana()

        # Build the Dhatu from root and gana
        try:
            dhatu_obj = Dhatu.mula(dhatu, getattr(Gana, gana))  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise ValueError(f"Unknown gana: {gana}") from exc

        # Build the Pada (tinanta request)
        try:
            pada = Pada.Tinanta(
                dhatu=dhatu_obj,
                prayoga=Prayoga.Kartari,
                lakara=getattr(Lakara, lakara),
                purusha=getattr(Purusha, purusha),
                vacana=getattr(Vacana, vacana),
            )
        except AttributeError as exc:
            raise ValueError(f"Invalid tense/person/number: {lakara}/{purusha}/{vacana}") from exc

        # Derive using Vyakarana (Pāṇinian grammar engine)
        results = vyakarana.derive(pada)  # type: ignore[attr-defined]
        if not results:
            raise ValueError(f"No derivation for {dhatu} {lakara} {purusha} {vacana}")

        prakriya = results[0]
        text = str(prakriya.text)
        if not text:
            raise ValueError(f"Empty derivation for {dhatu} {lakara} {purusha} {vacana}")

        # Extract sūtra trace from prakriya history
        derivation_trace = tuple(str(step.code) for step in prakriya.history)

        return PrabasaExample(
            text=text,
            language="sa",
            derivation_trace=derivation_trace,
            meta={
                "dhatu": dhatu,
                "gana": gana,
                "lakara": lakara,
                "purusha": purusha,
                "vacana": vacana,
            },
        )


@dataclass(frozen=True)
class ParibhashaInterpreterConfig:
    """Configuration for Paribhāṣā rule-ordering interpreter."""

    enable_rule_validation: bool = True
    paribhasha_version: str = "classical"  # Classical vs. extended


class ParibhashaRuleInterpreter:
    """Validates and interprets Pāṇinian rule-application order (Paribhāṣā)."""

    def __init__(self, config: ParibhashaInterpreterConfig | None = None) -> None:
        """Initialize with Paribhāṣā precedence rules."""
        self.config = config or ParibhashaInterpreterConfig()

    def interpret_derivation(self, example: PrabasaExample) -> PrabasaExample:
        """Enrich derivation with Paribhāṣā rule types & classes.

        Args:
            example: PrabasaExample with derivation_trace populated by Vidyut.

        Returns:
            Same example with paribhasha_rule_classes populated.

        Raises:
            NotImplementedError: Stub for Phase 3.
        """
        raise NotImplementedError(
            "ParibhashaRuleInterpreter.interpret_derivation is a stub. "
            "Implementation maps each sūtra to its Paribhāṣā class "
            "(ādeśa, āgama, lopa, etc.) and validates ordering via "
            "classical precedence rules."
        )


@dataclass(frozen=True)
class ShabdabodhaAssemblerConfig:
    """Configuration for Śabdabodha sentence assembly."""

    frame_inventory: frozenset[str] = frozenset({"akarmaka", "transitive"})
    nominal_stem_pool: frozenset[str] = frozenset(("bAla", "nara", "vana", "grham"))
    max_oblique_slots: int = 3


class ShabdabodhaAssembler:
    """Assembles parsed sentences with kāraka frames (Śabdabodha)."""

    def __init__(self, config: ShabdabodhaAssemblerConfig | None = None) -> None:
        """Initialize with frame inventory & nominal pool."""
        self.config = config or ShabdabodhaAssemblerConfig()

    def assemble_sentence(
        self, morph_example: PrabasaExample, frame_type: FrameType
    ) -> PrabasaExample:
        """Generate nominals & assemble full kāraka-parsed sentence.

        Args:
            morph_example: Example with verb morphology populated.
            frame_type: Frame type (akarmaka, transitive, multi_oblique).

        Returns:
            PrabasaExample with karaka_parse and frame_signature populated.

        Raises:
            NotImplementedError: Stub for Phase 3.
            ValueError: if frame_type not in inventory.
        """
        raise NotImplementedError(
            "ShabdabodhaAssembler.assemble_sentence is a stub. "
            "Implementation selects nominal stems from pool, "
            "generates vacana agreement, and linearizes to surface order."
        )


@dataclass(frozen=True)
class VyutpattivadaEngineConfig:
    """Configuration for Vyutpattivāda semantic graph generation."""

    padartha_lexicon_path: str | None = None  # Override default lexicon
    enable_type_validation: bool = True
    skip_invalid_graphs: bool = True


class VyutpattivadaEngine:
    """Maps parsed sentences to typed Śabdabodha graphs (Vyutpattivāda)."""

    # Classical Vyutpattivāda rule ids (coverage tracking)
    RULE_DHATU_KRIYA: Final = "VYU-001-dhatu-kriya"
    RULE_KARW_SAMYOGATA: Final = "VYU-002-karwA-samyogata"
    RULE_KARMA_VISAYATA: Final = "VYU-003-karma-visayata"
    RULE_KARANAM: Final = "VYU-004-karaNam"
    RULE_ADHIKARANAM: Final = "VYU-005-aXikaraNam-samyogata"
    RULE_APADANA: Final = "VYU-006-apAxAnam-samyogata"
    RULE_SAMPRADANA: Final = "VYU-007-sampraxAnam-samyogata"
    RULE_SANKHYA_VISESANA: Final = "VYU-008-saMKyA-visesana"
    RULE_PADARTHA_GUNA: Final = "VYU-009-padartha-guna"

    def __init__(self, config: VyutpattivadaEngineConfig | None = None) -> None:
        """Initialize with type rules & validation strategy."""
        self.config = config or VyutpattivadaEngineConfig()
        self._skip_log: dict[str, int] = {}

    def compile_graph(self, parsed_example: PrabasaExample) -> PrabasaExample:
        """Apply Vyutpattivāda rules to generate typed semantic graph.

        Wraps :func:`compile_shabdabodha` from the :mod:`shabdabodha` rule engine,
        mapping kāraka parse + frame signature into a type-valid Śabdabodha graph
        (ADR-0034 D3). Returns the input example with shabdabodha_graph populated,
        or the same example unchanged if the graph cannot be built.

        Args:
            parsed_example: Example with text, karaka_parse, and frame_signature populated.

        Returns:
            Same example with shabdabodha_graph populated if successful, else unchanged.

        Raises:
            ValueError: if enable_type_validation=True and graph violates constraints.
        """
        from psalm.infrastructure.generators.paribhasha.shabdabodha import (
            ShabdabodhaSkip,
            ShabdabodhaSuccess,
            compile_shabdabodha,
        )

        if not parsed_example.karaka_parse:
            return parsed_example

        sentence = AnnotatedSentence(
            text=parsed_example.text,
            karaka_parse=parsed_example.karaka_parse,
            meta={"frame_signature": parsed_example.frame_signature} | parsed_example.meta,
        )

        outcome = compile_shabdabodha(sentence)

        # Log skip reasons for coverage analysis
        if isinstance(outcome, ShabdabodhaSkip):
            self._skip_log[outcome.rule_id] = self._skip_log.get(outcome.rule_id, 0) + 1
            if self.config.skip_invalid_graphs:
                return parsed_example
            raise ValueError(f"Graph compilation skipped: {outcome.reason} ({outcome.rule_id})")

        # outcome is ShabdabodhaSuccess
        assert isinstance(outcome, ShabdabodhaSuccess)
        if self.config.enable_type_validation:
            from psalm.infrastructure.generators.paribhasha.relations import validate_graph

            try:
                validate_graph(outcome.graph)
            except Exception as exc:
                if self.config.skip_invalid_graphs:
                    return parsed_example
                raise ValueError(f"Graph validation failed: {exc}") from exc

        return PrabasaExample(
            text=parsed_example.text,
            language=parsed_example.language,
            morpheme_boundaries=parsed_example.morpheme_boundaries,
            morpheme_stream=parsed_example.morpheme_stream,
            derivation_trace=parsed_example.derivation_trace,
            paribhasha_rule_classes=parsed_example.paribhasha_rule_classes,
            karaka_parse=parsed_example.karaka_parse,
            frame_signature=parsed_example.frame_signature,
            shabdabodha_graph=outcome.graph,
            nyaya_validity_label=parsed_example.nyaya_validity_label,
            nyaya_semantic_augmentation=parsed_example.nyaya_semantic_augmentation,
            meta=parsed_example.meta,
        )


@dataclass(frozen=True)
class NavyaNyayaAugmenterConfig:
    """Configuration for Navya-Nyāya semantic augmentation."""

    enable_vyapti_labels: bool = False  # Phase 4
    enable_hetvabhasa_detection: bool = False  # Phase 4
    fallacy_taxonomy_path: str | None = None


class NavyaNyayaAugmenter:
    """Augments semantic graphs with Navya-Nyāya validity labels (Phase 4)."""

    def __init__(self, config: NavyaNyayaAugmenterConfig | None = None) -> None:
        """Initialize with Navya-Nyāya rules (stub for Phase 3)."""
        self.config = config or NavyaNyayaAugmenterConfig()

    def augment_example(self, vyutpattivada_example: PrabasaExample) -> PrabasaExample:
        """Compute validity labels and fallacy annotations (stub).

        Args:
            vyutpattivada_example: Example with shabdabodha_graph populated.

        Returns:
            Same example with nyaya_validity_label and nyaya_semantic_augmentation populated.

        Raises:
            NotImplementedError: Stub for Phase 3 (deferred to Phase 4).
        """
        # Phase 3: placeholder — return VALID label for all graphs
        if vyutpattivada_example.shabdabodha_graph is None:
            return vyutpattivada_example

        updated = PrabasaExample(
            text=vyutpattivada_example.text,
            language=vyutpattivada_example.language,
            morpheme_boundaries=vyutpattivada_example.morpheme_boundaries,
            morpheme_stream=vyutpattivada_example.morpheme_stream,
            derivation_trace=vyutpattivada_example.derivation_trace,
            paribhasha_rule_classes=vyutpattivada_example.paribhasha_rule_classes,
            karaka_parse=vyutpattivada_example.karaka_parse,
            frame_signature=vyutpattivada_example.frame_signature,
            shabdabodha_graph=vyutpattivada_example.shabdabodha_graph,
            nyaya_validity_label=ValidityLabel.VALID,  # Placeholder
            nyaya_semantic_augmentation={},
            meta=vyutpattivada_example.meta,
        )
        return updated


# ============================================================================
# Full Pipeline Orchestration
# ============================================================================


@dataclass(frozen=True)
class CorpusFromGrammarConfig:
    """Configuration for the complete PRABHĀSA pipeline."""

    n_examples: int = 1000
    seed: int = 0

    vidyut_config: VidyutEngineConfig | None = None
    paribhasha_config: ParibhashaInterpreterConfig | None = None
    shabdabodha_config: ShabdabodhaAssemblerConfig | None = None
    vyutpattivada_config: VyutpattivadaEngineConfig | None = None
    navya_nyaya_config: NavyaNyayaAugmenterConfig | None = None

    include_paribhasha: bool = True
    include_vyutpattivada: bool = True
    include_navya_nyaya: bool = False  # Phase 4


class CorpusFromGrammarGenerator:
    """Orchestrates the full PRABHĀSA pipeline.

    Composes Vidyut → Paribhāṣā → Śabdabodha → Vyutpattivāda → Navya-Nyāya
    to generate unbounded, grammar-coherent Sanskrit training corpora.
    """

    def __init__(self, config: CorpusFromGrammarConfig | None = None) -> None:
        """Initialize all five engines."""
        self.config = config or CorpusFromGrammarConfig()
        self.vidyut = VidyutMorphologyEngine(self.config.vidyut_config)
        self.paribhasha = ParibhashaRuleInterpreter(self.config.paribhasha_config)
        self.shabdabodha = ShabdabodhaAssembler(self.config.shabdabodha_config)
        self.vyutpattivada = VyutpattivadaEngine(self.config.vyutpattivada_config)
        self.navya_nyaya = NavyaNyayaAugmenter(self.config.navya_nyaya_config)

    def generate_corpus(self) -> list[PrabasaExample]:
        """Generate a complete PRABHĀSA corpus.

        Returns:
            List of PrabasaExample records, each with all five labels populated.

        Raises:
            NotImplementedError: Stub for Phase 3.
        """
        raise NotImplementedError(
            "CorpusFromGrammarGenerator.generate_corpus is a stub. "
            "Implementation orchestrates the pipeline: "
            "Vidyut → Paribhāṣā → Śabdabodha → Vyutpattivāda → Navya-Nyāya, "
            "tracking coverage and skips along the way."
        )

    def measure_coverage(self, corpus: list[PrabasaExample]) -> dict[str, Any]:
        """Compute coverage ledger (see ADR-0019 M1–M3).

        Args:
            corpus: Generated corpus.

        Returns:
            Coverage report: total examples, valid graphs, skips by reason.

        Raises:
            NotImplementedError: Stub for Phase 3.
        """
        raise NotImplementedError(
            "CorpusFromGrammarGenerator.measure_coverage is a stub. "
            "Implementation counts successes & skips, groups by frame type, "
            "and produces a versioned ledger for publication."
        )


# ============================================================================
# Export & Integration
# ============================================================================


def to_aligned_record(example: PrabasaExample) -> dict[str, Any]:
    """Convert PrabasaExample to paribhasha_aligned_v1 JSON record.

    Args:
        example: PRABHĀSA example with all labels.

    Returns:
        JSON-serializable dict conforming to aligned-pair schema.

    Raises:
        NotImplementedError: Stub for Phase 3.
        ValueError: if required fields missing.
    """
    raise NotImplementedError(
        "to_aligned_record is a stub. Implementation serializes "
        "shabdabodha_graph, karaka_parse, and meta to paribhasha_aligned_v1 format."
    )


def validate_aligned_record(record: dict[str, Any]) -> None:
    """Validate record against paribhasha_aligned_v1 JSON schema.

    Args:
        record: JSON record to validate.

    Raises:
        jsonschema.ValidationError: if record does not conform to schema.
    """
    raise NotImplementedError(
        "validate_aligned_record is a stub. Implementation uses "
        "jsonschema to validate against docs/contracts/aligned-pair-schema.json."
    )
