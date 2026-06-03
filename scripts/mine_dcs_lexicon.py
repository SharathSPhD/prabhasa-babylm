"""Mine a frequency-ranked Sanskrit lexicon from the DCS CoNLL-U corpus.

The Pāṇinian frame generator's diversity is bounded by its lexicon. To lift the
~75k-frame ceiling *without* drowning the corpus in obscure stems, we ground the
generator's vocabulary in **real DCS usage**: count lemma frequencies by part of
speech (and, for nouns, dominant gender), transliterate IAST → SLP1 (Vidyut's
native channel), and emit a frequency table. A downstream step intersects this
with Vidyut's *verified* inventories (dhātupāṭha + kośa) so every entry is both
real (attested in DCS) and realizable (derivable by ``vidyut.prakriya``).

Output JSON schema::

    {"schema": "dcs-lexicon-v1",
     "noun_stems": [{"slp1": "brahman", "linga": "napum", "freq": 1234}, ...],
     "verb_lemmas": [{"slp1": "BU", "freq": 5678}, ...]}

Usage::

    python scripts/mine_dcs_lexicon.py \
        --dcs /home/sharaths/projects/slm-1/data/dcs-sanskrit/dcs/data/conllu \
        --out data/lexicon/dcs_lexicon.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import vidyut.lipi as lipi

_IAST = lipi.Scheme.Iast
_SLP1 = lipi.Scheme.Slp1

#: DCS Gender feature → realizer liṅga code.
_GENDER_TO_LINGA = {"Masc": "pum", "Neut": "napum", "Fem": "stri"}


def _to_slp1(iast: str) -> str | None:
    """Transliterate an IAST lemma to SLP1, or ``None`` if it round-trips empty."""
    try:
        out = lipi.transliterate(iast, _IAST, _SLP1)
    except Exception:  # pragma: no cover - defensive on malformed lemmas
        return None
    return out or None


def _gender(feats: str) -> str | None:
    for feat in feats.split("|"):
        if feat.startswith("Gender="):
            return _GENDER_TO_LINGA.get(feat.split("=", 1)[1])
    return None


def mine(dcs_dir: Path) -> dict[str, object]:
    noun_freq: Counter[str] = Counter()
    noun_gender: dict[str, Counter[str]] = defaultdict(Counter)
    verb_freq: Counter[str] = Counter()

    files = sorted(dcs_dir.rglob("*.conllu"))
    for i, path in enumerate(files):
        if i % 2000 == 0:
            print(f"  ... {i}/{len(files)} files", flush=True)
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line or line[0] in "#\n":
                    continue
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 6:
                    continue
                idx, _form, lemma, upos, _x, feats = cols[:6]
                if "-" in idx or lemma == "_" or not lemma:
                    continue  # multiword range / unanalysed
                slp1 = _to_slp1(lemma)
                if not slp1 or not slp1.isascii():
                    continue
                if upos == "NOUN":
                    noun_freq[slp1] += 1
                    lg = _gender(feats)
                    if lg:
                        noun_gender[slp1][lg] += 1
                elif upos == "VERB":
                    verb_freq[slp1] += 1

    noun_stems = [
        {
            "slp1": stem,
            "linga": noun_gender[stem].most_common(1)[0][0] if noun_gender[stem] else "pum",
            "freq": freq,
        }
        for stem, freq in noun_freq.most_common()
    ]
    verb_lemmas = [{"slp1": v, "freq": f} for v, f in verb_freq.most_common()]
    return {
        "schema": "dcs-lexicon-v1",
        "n_files": len(files),
        "noun_stems": noun_stems,
        "verb_lemmas": verb_lemmas,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dcs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    result = mine(args.dcs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=True), encoding="utf-8")
    nn = len(result["noun_stems"])  # type: ignore[arg-type]
    nv = len(result["verb_lemmas"])  # type: ignore[arg-type]
    print(f"wrote {args.out}: {nn} noun stems, {nv} verb lemmas from {result['n_files']} files")


if __name__ == "__main__":
    main()
