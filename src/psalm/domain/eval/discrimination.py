"""Role-discrimination minimal-pair construction for the H1 readout (ADR-0015).

After full-LF exact-match was shown to floor at proxy scale (SCAN gen splits 0%;
COGS lexical EM ~0.02–0.05 even at 150M / 12k steps), the H1 readout moves from
*generating* the logical form to *discriminating* a correct LF from a minimally
role-corrupted one. The model "passes" an item when it assigns higher likelihood
to the correct LF than to the corrupted one — chance is 50%, so the readout is
**off-floor by construction**, and it reuses teacher-forced sequence scoring (no
generation, so it sidesteps both the EM floor and the token-F1 boilerplate trap).

The corruption is mechanism-aligned with kāraka (argument-role) theory: it swaps
the **agent** and **theme** fillers of the main predicate, producing a
syntactically valid but role-reversed LF. Sensitivity to that swap *is* sensitivity
to argument-role structure — exactly what a Pāṇinian prior should sharpen.

COGS LF grammar (the only shape we corrupt):
    ``verb . agent ( e , A ) AND verb . theme ( e , B ) AND ...``
where ``e`` is the shared event variable and ``A`` / ``B`` are arguments
(``x _ N`` or a proper name). Pure, deterministic, no I/O — unit-tested.
"""

from __future__ import annotations

import re
from collections.abc import Callable

#: Matches one role term: ``verb . role ( arg1 , arg2 )`` with single-spaced
#: COGS tokenization. arg1 is the event variable; arg2 is the filler we swap.
_ROLE = re.compile(
    r"(?P<verb>\w+) \. (?P<role>agent|theme) \( (?P<event>x _ \d+) , (?P<filler>x _ \d+|[A-Z]\w*) \)"
)


def corrupt_role_swap(lf: str) -> str | None:
    """Return ``lf`` with the main predicate's agent/theme fillers swapped.

    The "main predicate" is the first event variable (in left-to-right order)
    that carries **both** an ``agent`` and a ``theme`` term. Their second
    arguments (fillers) are exchanged. Returns ``None`` when no event has both
    an agent and a theme (intransitive / agent+ccomp / agent+recipient only) —
    such items are excluded from the discrimination set and counted as coverage
    misses rather than silently mangled.
    """
    agents: dict[str, tuple[int, int, str]] = {}  # event -> (start, end, filler)
    themes: dict[str, tuple[int, int, str]] = {}
    order: list[str] = []
    for m in _ROLE.finditer(lf):
        event = m.group("event")
        span = (m.start("filler"), m.end("filler"), m.group("filler"))
        if m.group("role") == "agent":
            agents.setdefault(event, span)
        else:
            themes.setdefault(event, span)
        if event not in order:
            order.append(event)

    for event in order:
        if event in agents and event in themes:
            if agents[event][2] == themes[event][2]:
                continue  # swapping identical fillers is a no-op; skip
            # Order the two spans by position; under a swap each position takes
            # the *other* span's filler. Rebuild left→right from fixed offsets.
            (s1, e1, _), (s2, e2, _) = sorted([agents[event], themes[event]])
            new1 = (themes[event] if agents[event][0] < themes[event][0] else agents[event])[2]
            new2 = (agents[event] if agents[event][0] < themes[event][0] else themes[event])[2]
            return lf[:s1] + new1 + lf[e1:s2] + new2 + lf[e2:]
    return None


#: Any role term filler (agent/theme/recipient/...) — used to harvest distractor
#: entities and to retarget a single binding.
_ANY_ROLE = re.compile(
    r"(?P<verb>\w+) \. (?P<role>\w+) \( (?P<event>x _ \d+) , (?P<filler>x _ \d+|[A-Z]\w*) \)"
)


def corrupt_distractor_theme(lf: str) -> str | None:
    """Rebind the main predicate's **theme** to a distractor entity from the LF.

    Harder than ``corrupt_role_swap``: rather than swapping two adjacent surface
    arguments (which COGS word-order trivially cues), this replaces the theme
    filler with a *different* entity that genuinely appears elsewhere in the LF.
    The corrupted LF is well-formed and locally plausible; only a model that
    tracks **which** entity fills the role (not just surface order) can reject it.
    Returns ``None`` if there is no main theme or no distinct distractor entity.
    """
    themes: dict[str, tuple[int, int, str]] = {}
    fillers: list[str] = []
    order: list[str] = []
    for m in _ANY_ROLE.finditer(lf):
        event, filler = m.group("event"), m.group("filler")
        fillers.append(filler)
        if m.group("role") == "theme":
            themes.setdefault(event, (m.start("filler"), m.end("filler"), filler))
        if event not in order:
            order.append(event)

    for event in order:
        if event in themes:
            start, end, cur = themes[event]
            # First distinct entity elsewhere in the LF (deterministic).
            distractor = next((f for f in fillers if f != cur), None)
            if distractor is None:
                return None
            return lf[:start] + distractor + lf[end:]
    return None


#: Named corruption operators for the graded ladder (ADR-0015 amendment / sweep).
CORRUPTIONS: dict[str, Callable[[str], str | None]] = {
    "swap": corrupt_role_swap,
    "distractor": corrupt_distractor_theme,
}
