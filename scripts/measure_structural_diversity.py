"""Structural-phase diversity match between the Pāṇinian and Dyck priors (§1.2).

The decisive B-vs-C comparison requires the two structural streams to differ only
in *content*, not in *diversity*. They do not, as configured: ``pre_budget`` is
13M whitespace tokens, but the baked Saṃsādhanī cache holds only ~0.6M
no-repeat tokens, so arm B loops the cache ~20× while arm C (Dyck) draws fresh
tokens — confounding any B-vs-C difference with effective epochs.

This script measures, in the same whitespace-token unit ``take_until_tokens``
uses:
  * the Pāṇinian source's *no-repeat* token count (the diversity-safe cap), and
  * TTR + n-gram entropy for both sources at that matched budget.

It writes the recommended capped ``pre_budget`` so the battery matches B and C
on *fresh* structural tokens, not just on quantity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from psalm.domain.data.diversity import ngram_entropy, type_token_ratio
from psalm.infrastructure.generators.dyck_source import DyckSentenceSource
from psalm.infrastructure.generators.jsonl_source import JsonlSentenceSource


def wtokens(text: str) -> int:
    return max(len(text.split()), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache/samsadhani.jsonl")
    ap.add_argument(
        "--safety", type=float, default=0.9, help="fraction of no-repeat budget to cap at"
    )
    ap.add_argument("--out", default="docs/data/phase2-structural-diversity.json")
    args = ap.parse_args()

    # --- Pāṇinian: read the cache once, count unique sentences + no-repeat tokens.
    pan = JsonlSentenceSource(args.cache)
    pan_items = list(pan._load())  # noqa: SLF001 - measurement needs the raw, unshuffled set
    pan_texts = [s.text for s in pan_items]
    pan_unique = len(set(pan_texts))
    pan_norepeat_tokens = sum(wtokens(t) for t in pan_texts)

    # Diversity-safe cap: stay below the no-repeat count so B never loops.
    cap = int(pan_norepeat_tokens * args.safety)

    # --- Dyck: stream fresh until the matched (capped) token budget.
    dyck = DyckSentenceSource()
    dyck_texts: list[str] = []
    emitted = 0
    offset = 0
    while emitted < cap:
        produced = 0
        for item in dyck.stream(2048, seed=offset):
            produced += 1
            dyck_texts.append(item.text)
            emitted += wtokens(item.text)
            if emitted >= cap:
                break
        if produced == 0:
            break
        offset += 1

    # --- Pāṇinian at the matched budget: take cache sentences (no repeat) up to cap.
    pan_budget_texts: list[str] = []
    emitted = 0
    for t in pan_texts:
        pan_budget_texts.append(t)
        emitted += wtokens(t)
        if emitted >= cap:
            break

    report = {
        "unit": "whitespace tokens (matches take_until_tokens)",
        "paninian": {
            "cache_sentences": len(pan_texts),
            "unique_sentences": pan_unique,
            "no_repeat_tokens": pan_norepeat_tokens,
            "ttr": round(type_token_ratio(pan_budget_texts), 4),
            "bigram_entropy": round(ngram_entropy(pan_budget_texts, 2), 4),
            "trigram_entropy": round(ngram_entropy(pan_budget_texts, 3), 4),
        },
        "dyck_matched": {
            "sentences": len(dyck_texts),
            "tokens": emitted,
            "ttr": round(type_token_ratio(dyck_texts), 4),
            "bigram_entropy": round(ngram_entropy(dyck_texts, 2), 4),
            "trigram_entropy": round(ngram_entropy(dyck_texts, 3), 4),
        },
        "recommended_pre_budget_tokens": cap,
        "old_pre_budget_tokens_high": 130_000_000 // 10,
        "loop_factor_at_old_budget": round((130_000_000 // 10) / max(pan_norepeat_tokens, 1), 1),
        "verdict": (
            "CAP pre_budget to recommended value so B (Pāṇinian, no-repeat) and C "
            "(Dyck, fresh) are matched on fresh structural tokens. At the old "
            "13M budget B would loop the cache ~%.0fx, confounding B-vs-C with "
            "effective epochs." % ((130_000_000 // 10) / max(pan_norepeat_tokens, 1))
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
