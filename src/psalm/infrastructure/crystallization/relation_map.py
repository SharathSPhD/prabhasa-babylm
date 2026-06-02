"""Wikidata property → kāraka-frame class map (crystallization Attack-2 layer).

Curated from the Phase-2 design spike; shared so Milestone 1 and
``scripts/crystallization_spike.py`` stay aligned without duplicating semantics.
"""

from __future__ import annotations

from typing import Literal

FrameClass = Literal["karaka_action", "copular", "unmappable"]

PROPERTY_MAP: dict[str, dict[str, str]] = {
    "P31": {"label": "instance of", "cls": "copular", "role": "predicate"},
    "P279": {"label": "subclass of", "cls": "copular", "role": "predicate"},
    "P276": {"label": "location", "cls": "karaka_action", "role": "aXikaraNam"},
    "P131": {"label": "located in admin entity", "cls": "karaka_action", "role": "aXikaraNam"},
    "P706": {"label": "located on terrain", "cls": "karaka_action", "role": "aXikaraNam"},
    "P170": {"label": "creator", "cls": "karaka_action", "role": "karwA"},
    "P176": {"label": "manufacturer", "cls": "karaka_action", "role": "karwA"},
    "P50": {"label": "author", "cls": "karaka_action", "role": "karwA"},
    "P57": {"label": "director", "cls": "karaka_action", "role": "karwA"},
    "P61": {"label": "discoverer/inventor", "cls": "karaka_action", "role": "karwA"},
    "P127": {"label": "owned by", "cls": "karaka_action", "role": "karwA"},
    "P1056": {"label": "product or material produced", "cls": "karaka_action", "role": "karma"},
    "P828": {"label": "has cause", "cls": "karaka_action", "role": "apAxAnam"},
    "P1542": {"label": "has effect", "cls": "karaka_action", "role": "karma"},
    "P137": {"label": "operator", "cls": "karaka_action", "role": "karwA"},
    "P361": {"label": "part of", "cls": "copular", "role": "predicate"},
    "P527": {"label": "has part", "cls": "copular", "role": "predicate"},
    "P155": {"label": "follows", "cls": "karaka_action", "role": "apAxAnam"},
    "P156": {"label": "followed by", "cls": "karaka_action", "role": "karma"},
    "P186": {"label": "made from material", "cls": "karaka_action", "role": "karaNam"},
    "P1535": {"label": "used by", "cls": "karaka_action", "role": "karwA"},
    "P366": {"label": "has use", "cls": "karaka_action", "role": "sampraxAnam"},
    "P457": {"label": "foundational text", "cls": "copular", "role": "predicate"},
    "P17": {"label": "country", "cls": "karaka_action", "role": "aXikaraNam"},
    "P19": {"label": "place of birth", "cls": "karaka_action", "role": "aXikaraNam"},
    "P20": {"label": "place of death", "cls": "karaka_action", "role": "aXikaraNam"},
    "P800": {"label": "notable work", "cls": "karaka_action", "role": "karma"},
    "P106": {"label": "occupation", "cls": "copular", "role": "predicate"},
    "P39": {"label": "position held", "cls": "copular", "role": "predicate"},
    "P569": {"label": "date of birth", "cls": "unmappable", "role": ""},
    "P570": {"label": "date of death", "cls": "unmappable", "role": ""},
    "P580": {"label": "start time", "cls": "unmappable", "role": ""},
    "P582": {"label": "end time", "cls": "unmappable", "role": ""},
    "P585": {"label": "point in time", "cls": "unmappable", "role": ""},
    "P1082": {"label": "population", "cls": "unmappable", "role": ""},
    "P2044": {"label": "elevation", "cls": "unmappable", "role": ""},
    "P625": {"label": "coordinate location", "cls": "unmappable", "role": ""},
    "P18": {"label": "image", "cls": "unmappable", "role": ""},
    "P214": {"label": "VIAF ID", "cls": "unmappable", "role": ""},
}

# Dhātus verified for oblique kārakas in ``karaka_frames.VERIFIED_OBLIQUE_FRAMES``.
ROLE_DHATUS: dict[str, tuple[str, ...]] = {
    "aXikaraNam": ("vas1", "gam1", "BU1"),
    "karwA": ("kf8",),
    "karma": ("xfS1", "kf8", "KAx1"),
    "apAxAnam": ("gam1",),
    "karaNam": ("KAx1", "gam1"),
    "sampraxAnam": ("xA1",),
    "predicate": ("BU1",),
}

# In-domain properties for natural-kinds + physical-action (subset of full map).
IN_DOMAIN_PROPERTIES: frozenset[str] = frozenset(
    {
        "P31",
        "P279",
        "P276",
        "P131",
        "P706",
        "P17",
        "P361",
        "P527",
        "P186",
        "P828",
        "P1542",
        "P366",
        "P155",
        "P156",
    }
)


def property_info(pid: str) -> dict[str, str] | None:
    return PROPERTY_MAP.get(pid)


def frame_class(pid: str) -> FrameClass | None:
    info = property_info(pid)
    if info is None:
        return None
    return info["cls"]  # type: ignore[return-value]


def is_generable(pid: str) -> bool:
    cls = frame_class(pid)
    return cls in ("karaka_action", "copular")


def karaka_action_rate() -> float:
    total = len(PROPERTY_MAP)
    karaka = sum(1 for v in PROPERTY_MAP.values() if v["cls"] == "karaka_action")
    return karaka / total if total else 0.0
