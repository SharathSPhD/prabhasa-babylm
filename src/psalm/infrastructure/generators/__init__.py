"""Phase-1–3 synthetic generators (Pāṇinian, Dyck, Paribhāṣā, PRABHĀSA)."""

from psalm.infrastructure.generators.corpus_from_grammar import (
    CorpusFromGrammarConfig,
    CorpusFromGrammarGenerator,
    FrameType,
    NavyaNyayaAugmenter,
    NavyaNyayaAugmenterConfig,
    ParibhashaInterpreterConfig,
    ParibhashaRuleInterpreter,
    PrabasaExample,
    ShabdabodhaAssembler,
    ShabdabodhaAssemblerConfig,
    ValidityLabel,
    VidyutEngineConfig,
    VidyutMorphologyEngine,
    VyutpattivadaEngine,
    VyutpattivadaEngineConfig,
    to_aligned_record,
    validate_aligned_record,
)
from psalm.infrastructure.generators.paribhasha import (
    ParibhashaGenerator,
    ParibhashaGeneratorConfig,
    ShabdabodhaGraph,
    Stratum,
    render_graph,
    validate_graph,
)
from psalm.infrastructure.generators.paribhasha_source import ParibhashaSentenceSource

__all__ = [
    # PRABHĀSA pipeline (Phase 3)
    "CorpusFromGrammarConfig",
    "CorpusFromGrammarGenerator",
    "FrameType",
    "NavyaNyayaAugmenter",
    "NavyaNyayaAugmenterConfig",
    "ParibhashaRuleInterpreter",
    "ParibhashaInterpreterConfig",
    "PrabasaExample",
    "ShabdabodhaAssembler",
    "ShabdabodhaAssemblerConfig",
    "ValidityLabel",
    "VidyutEngineConfig",
    "VidyutMorphologyEngine",
    "VyutpattivadaEngine",
    "VyutpattivadaEngineConfig",
    "to_aligned_record",
    "validate_aligned_record",
    # Paribhāṣā generators (Phase 1–2)
    "ParibhashaGenerator",
    "ParibhashaGeneratorConfig",
    "ParibhashaSentenceSource",
    "ShabdabodhaGraph",
    "Stratum",
    "render_graph",
    "validate_graph",
]
