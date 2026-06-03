"""Information-parity ledger for the Paribhāṣā L2 stream (ADR-0034 D5 / ADR-0035 D6).

Answers the relabel objection quantitatively on a configurable number of frames:

* ``H(structure | kāraka)`` — conditional entropy of the canonical *structural
  template* given (a) the kāraka role multiset and (b) the full ``(stem, role)``
  parse. ``≫ 0`` against the role grammar means the typed graph is not an
  isomorphic relabel of the kāraka labelling; ``> 0`` against the full parse means
  the string carries strictly more than the gold parse (the saṃkhyā / number, which
  the parse omits).
* unique structural template count (``≫ 2``).
* ``VISAYATA`` present in 100% of transitive ``paribhasha_string``.
* padārtha categories realised (``≥ 3``; not all-DRAVYA).
* Vyutpattivāda coverage with the ākāṅkṣā / yogyatā skip ledger.

Frames are synthesised directly as ``(stem, vacana, role)`` words so the number is
sampled *independently* of the stem — exactly the distribution the gold parse cannot
predict. Output is a JSON ledger for the workstream closure record.

    uv run python scripts/measure_paribhasha_parity.py --n 10000 --seed 0 \\
        --out docs/data/paribhasha-parity-ledger.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
from collections import Counter, defaultdict

from psalm.application.data.ports import AnnotatedSentence
from psalm.infrastructure.generators.paribhasha.shabdabodha import (
    ShabdabodhaSuccess,
    compile_shabdabodha,
)
from psalm.infrastructure.generators.paribhasha.types import PadarthaCategory, ShabdabodhaGraph

# Lexicon partitioned by padārtha (matches shabdabodha.PADARTHA_LEXICON).
_DRAVYA = ("Pala", "aSva", "bAla", "gfha", "guru", "jala", "kanyA", "nara", "naxI", "puswaka", "rAma", "vana")
_GUNA = ("vixyA",)
_ALL_STEMS = _DRAVYA + _GUNA
_VACANA = ("eka", "xvi", "bahu")
_TRANSITIVE = ("KAx1", "gam1")
_AKARMAKA = ("vas1",)
_OBLIQUE_ROLES = ("karaNam", "aXikaraNam", "apAxAnam", "sampraxAnam")
_LAKARA = "viXiH"


def _frame(rng: random.Random) -> AnnotatedSentence:
    transitive = rng.random() < 0.7
    dhatu = rng.choice(_TRANSITIVE if transitive else _AKARMAKA)
    words: list[tuple[str, str, str]] = []  # (stem, vacana, role)
    words.append((rng.choice(_ALL_STEMS), rng.choice(_VACANA), "karwA"))
    if transitive and rng.random() < 0.85:
        words.append((rng.choice(_ALL_STEMS), rng.choice(_VACANA), "karma"))
    for role in _OBLIQUE_ROLES:
        if rng.random() < 0.4:
            words.append((rng.choice(_ALL_STEMS), rng.choice(_VACANA), role))
    surface = " ".join(f"{s}.{v}.{r}" for s, v, r in words) + f" {dhatu}.{_LAKARA}"
    karaka_parse = tuple((s, r) for s, _v, r in words)
    sig = f"{dhatu}|{_LAKARA}|" + "|".join(f"{s}:{v}:{r}" for s, v, r in words)
    return AnnotatedSentence(
        text=surface, language="sa", karaka_parse=karaka_parse, meta={"frame_signature": sig}
    )


def _template(graph: ShabdabodhaGraph) -> tuple[tuple[str, str, str, str], ...]:
    idx = {n.id: n for n in graph.nodes}
    items: list[tuple[str, str, str, str]] = []
    for e in graph.edges:
        src = idx[e.src]
        dst = idx[e.dst]
        # keep the saṃkhyā value in the template so the number (∉ parse) is visible
        src_label = (
            src.label
            if src.category is PadarthaCategory.GUNA and src.label.startswith("saMKyA")
            else src.category.value
        )
        items.append((e.sansa.value, src_label, dst.category.value, e.qualifier or ""))
    return tuple(sorted(items))


def _cond_entropy(pairs: list[tuple[object, object]]) -> float:
    by_k: dict[object, Counter[object]] = defaultdict(Counter)
    for k, t in pairs:
        by_k[k][t] += 1
    total = len(pairs)
    h = 0.0
    for tc in by_k.values():
        nk = sum(tc.values())
        pk = nk / total
        hk = 0.0
        for c in tc.values():
            p = c / nk
            hk -= p * math.log2(p)
        h += pk * hk
    return h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="docs/data/paribhasha-parity-ledger.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    templates: Counter[tuple[tuple[str, str, str, str], ...]] = Counter()
    pairs_full: list[tuple[object, object]] = []
    pairs_role: list[tuple[object, object]] = []
    cats: Counter[str] = Counter()
    skips: Counter[str] = Counter()
    n_ok = 0
    n_trans = 0
    n_trans_visaya = 0

    for _ in range(args.n):
        sent = _frame(rng)
        outcome = compile_shabdabodha(sent)
        if not isinstance(outcome, ShabdabodhaSuccess):
            skips[outcome.rule_id] += 1
            continue
        n_ok += 1
        tpl = _template(outcome.graph)
        templates[tpl] += 1
        for node in outcome.graph.nodes:
            cats[node.category.value] += 1
        pairs_full.append((tuple(sent.karaka_parse), tpl))
        pairs_role.append((tuple(sorted(r for _s, r in sent.karaka_parse)), tpl))
        if any(r == "karma" for _s, r in sent.karaka_parse):
            n_trans += 1
            if "VISAYATA(" in outcome.rendered.ascii:
                n_trans_visaya += 1

    ledger = {
        "schema_version": "paribhasha-parity-v1",
        "n_frames": args.n,
        "seed": args.seed,
        "n_valid_graphs": n_ok,
        "coverage_fraction": round(n_ok / args.n, 4) if args.n else 0.0,
        "skip_ledger": dict(skips),
        "unique_templates": len(templates),
        "H_structure_given_role_multiset_bits": round(_cond_entropy(pairs_role), 4),
        "H_structure_given_full_parse_bits": round(_cond_entropy(pairs_full), 4),
        "padartha_categories_used": sorted(cats),
        "padartha_node_counts": dict(cats),
        "transitive_frames": n_trans,
        "transitive_visayata_fraction": round(n_trans_visaya / n_trans, 4) if n_trans else None,
        "gates": {
            "P_visaya_lossless": n_trans_visaya == n_trans,
            "P_info_parity_role_grammar": _cond_entropy(pairs_role) > 0.0,
            "P_info_parity_full_parse": _cond_entropy(pairs_full) > 0.0,
            "P_template_count_gt_2": len(templates) > 2,
            "P_padartha_ge_3_categories": len(cats) >= 3,
        },
    }
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(ledger, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
