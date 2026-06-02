"""KB triple → kāraka meaning structure (reverse Saṃsādhanī input)."""

from __future__ import annotations

from psalm.domain.data.karaka_frames import VERIFIED_OBLIQUE_FRAMES, KarakaFrame
from psalm.infrastructure.crystallization.domain_triples import KbTriple
from psalm.infrastructure.crystallization.entity_lexicon import EntityEntry, EntityLexicon
from psalm.infrastructure.crystallization.relation_map import ROLE_DHATUS, property_info
from psalm.infrastructure.crystallization.stem_validation import (
    alias_entry_to_validated_stem,
    validated_stem_set,
)

# Lakāra / number grid for combinatorial crystallization (matches karaka_frames).
LAKARAS: tuple[str, ...] = (
    "varwamAnaH",
    "anaxyawanaBUwaH",
    "sAmAnyaBaviRyakAlaH",
    "viXiH",
)
NUMBERS: tuple[str, ...] = ("eka", "xvi", "bahu")


def _noun(idx: int, ent: EntityEntry, number: str, karaka: str, head: int) -> dict[str, object]:
    return {
        "id": idx,
        "pos": "noun",
        "stem": ent.stem_wx,
        "gender": ent.gender,
        "number": number,
        "karaka": karaka,
        "head": head,
    }


def _verb(idx: int, dhatu: str, lakara: str) -> dict[str, object]:
    return {
        "id": idx,
        "pos": "verb",
        "dhatu": dhatu,
        "prayoga": "karwari",
        "lakara": lakara,
    }


def _resolve_entries(
    sub: EntityEntry, obj: EntityEntry, *, require_generator: bool
) -> tuple[EntityEntry, EntityEntry] | None:
    if not require_generator:
        return sub, obj
    validated = validated_stem_set()
    if not validated:
        return sub, obj
    sub_a = alias_entry_to_validated_stem(sub, validated)
    obj_a = alias_entry_to_validated_stem(obj, validated)
    if sub_a is None or obj_a is None:
        return None
    return sub_a, obj_a


def _pick_oblique_dhatu(role: str, dhatu_index: int) -> str | None:
    """Choose a dhātu verified for ``role`` in ``VERIFIED_OBLIQUE_FRAMES``."""
    verified = VERIFIED_OBLIQUE_FRAMES.get(role)
    if not verified:
        return None
    dhatu, _with_karma = verified[dhatu_index % len(verified)]
    return dhatu


def build_frame_from_triple(
    triple: KbTriple,
    lexicon: EntityLexicon,
    *,
    lakara: str = "varwamAnaH",
    subject_number: str = "eka",
    object_number: str = "eka",
    dhatu_index: int = 0,
    require_generator_ready: bool = True,
) -> KarakaFrame | None:
    """Compose one meaning structure from a mapped entity–entity triple."""
    info = property_info(triple.property_id)
    if info is None:
        return None
    sub = lexicon.lookup(triple.subject_qid)
    obj = lexicon.lookup(triple.object_qid)
    if sub is None or obj is None:
        return None

    resolved = _resolve_entries(sub, obj, require_generator=require_generator_ready)
    if resolved is None:
        return None
    sub, obj = resolved

    cls = info["cls"]
    role = info["role"]
    if cls == "unmappable":
        return None

    if cls == "copular":
        # Predicative: kartā + karma + as1 (empirically accepted by Saṃsādhanī).
        dhatu = "as1"
        verb_id = 3
        words: list[dict[str, object]] = [
            _noun(1, sub, subject_number, "karwA", verb_id),
            _noun(2, obj, object_number, "karma", verb_id),
            _verb(verb_id, dhatu, lakara),
        ]
        sig = (
            triple.property_id,
            dhatu,
            lakara,
            sub.stem_wx,
            subject_number,
            obj.stem_wx,
            object_number,
            "copular",
        )
        return KarakaFrame({"words": words}, sig)

    # kāraka-action: object fills the relation role; subject is kartā unless role is karma-only.
    if role in VERIFIED_OBLIQUE_FRAMES:
        action_dhatu = _pick_oblique_dhatu(role, dhatu_index)
        if action_dhatu is None:
            return None
    else:
        dhatus = ROLE_DHATUS.get(role, ("as1",))
        action_dhatu = dhatus[dhatu_index % len(dhatus)]
    if role == "karwA":
        # Object is agent — swap kartā to object for this property family.
        verb_id = 3
        words = [
            _noun(1, obj, object_number, "karwA", verb_id),
            _noun(2, sub, subject_number, "karma", verb_id),
            _verb(verb_id, action_dhatu, lakara),
        ]
        sig = (
            triple.property_id,
            action_dhatu,
            lakara,
            obj.stem_wx,
            object_number,
            sub.stem_wx,
            subject_number,
            role,
        )
        return KarakaFrame({"words": words}, sig)

    if role in ("aXikaraNam", "apAxAnam", "karaNam", "sampraxAnam"):
        verb_id = 3
        words = [
            _noun(1, sub, subject_number, "karwA", verb_id),
            _noun(2, obj, object_number, role, verb_id),
            _verb(verb_id, action_dhatu, lakara),
        ]
        sig = (
            triple.property_id,
            action_dhatu,
            lakara,
            sub.stem_wx,
            subject_number,
            obj.stem_wx,
            object_number,
            role,
        )
        return KarakaFrame({"words": words}, sig)

    # karma and default transitive
    verb_id = 3
    words = [
        _noun(1, sub, subject_number, "karwA", verb_id),
        _noun(2, obj, object_number, "karma", verb_id),
        _verb(verb_id, action_dhatu, lakara),
    ]
    sig = (
        triple.property_id,
        action_dhatu,
        lakara,
        sub.stem_wx,
        subject_number,
        obj.stem_wx,
        object_number,
        role,
    )
    return KarakaFrame({"words": words}, sig)


def enumerate_frames_for_triples(
    triples: list[KbTriple],
    lexicon: EntityLexicon,
    *,
    max_frames: int | None = None,
    require_generator_ready: bool = False,
) -> list[KarakaFrame]:
    """Expand triples across lakāra/number/dhātu grid until ``max_frames`` signatures.

    Default ``require_generator_ready=False`` counts distinct meaning structures
    from the seed lexicon stems (dose-ceiling proof). Set ``True`` when probing
    live Saṃsādhanī acceptance (grammaticality sample).
    """
    seen: set[tuple[str, ...]] = set()
    out: list[KarakaFrame] = []
    for triple in triples:
        for lakara in LAKARAS:
            for sub_num in NUMBERS:
                for obj_num in NUMBERS:
                    for d_idx in range(3):
                        frame = build_frame_from_triple(
                            triple,
                            lexicon,
                            lakara=lakara,
                            subject_number=sub_num,
                            object_number=obj_num,
                            dhatu_index=d_idx,
                            require_generator_ready=require_generator_ready,
                        )
                        if frame is None or frame.signature in seen:
                            continue
                        seen.add(frame.signature)
                        out.append(frame)
                        if max_frames is not None and len(out) >= max_frames:
                            return out
    return out
