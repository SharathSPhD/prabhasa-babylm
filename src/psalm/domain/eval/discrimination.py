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
