"""Build the generator's expanded, verified, frequency-grounded lexicon.

Combines three sources into one lexicon the frame generator + Vidyut realizer can
consume directly:

* **DCS frequency** (`data/lexicon/dcs_lexicon.json`) — what real Sanskrit uses.
* **Vidyut kośa** (`pratipadikas()`) — 164k *verified* Basic noun stems w/ liṅga;
  the intersection guarantees every stem actually declines.
* **Vidyut dhātupāṭha** (`load_dhatu_entries()`) — 2229 dhātus, every one derivable.

Selection:
* Noun stems: DCS noun stems whose SLP1 lemma is a kośa Basic stem with a matching
  liṅga, ranked by DCS frequency, top ``--max-stems`` kept (default 5000).
* Dhātus: all dhātupāṭha entries, annotated with the DCS frequency of their
  normalised root (anubandha/accent-stripped) so the generator can weight common
  verbs. The hand-verified transitive set is preserved and flagged.

Output (`data/lexicon/generator_lexicon.json`)::

    {"schema": "generator-lexicon-v1",
     "stems": [{"slp1","linga","nyap","freq"}],
     "dhatus": [{"aupadeshika","gana","freq","transitive"}]}
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import vidyut.kosha as kosha_mod
import vidyut.prakriya as P

# Hand-verified transitive dhātus (SLP1 normalised root) that may license a karma.
# Bootstrapped from the original verified realizer lexicon (ADR-0033); intransitive
# (akarmaka) roots are excluded so the transitivity guard stays sound.
_VERIFIED_TRANSITIVE = {"gam", "paW", "KAd", "dfS", "kf", "vad", "dA"}

#: Vidyut anubandha / accent marks to strip when normalising an upadeśa to a root.
_ANUBANDHA = re.compile(r"[\\~^0-9iIuUfFeEoO\u0300-\u036f]+$")


def _normalise_root(aupadeshika: str) -> str:
    """Strip trailing it-markers/accents to approximate the bare SLP1 root."""
    s = aupadeshika.replace("\\", "").replace("~", "").replace("^", "")
    # drop a trailing it-vowel marker (e.g. "paWa~" -> "paW", "vada~" -> "vad")
    s = re.sub(r"[aiufx]$", "", s)
    return s


def _kosha_basic_stems(data_dir: Path) -> dict[str, set[str]]:
    kosha = kosha_mod.Kosha(str(data_dir / "kosha"))
    stems: dict[str, set[str]] = {}
    for p in kosha.pratipadikas():
        if not type(p).__name__.endswith("_Basic"):
            continue
        lemma = p.lemma
        if not lemma or not lemma.isascii():
            continue
        lgs = {str(lg).split(".")[-1] for lg in (p.lingas or [])}
        if lgs:
            stems.setdefault(lemma, set()).update(lgs)
    return stems


_KOSHA_LINGA = {"puM": "pum", "napuMsaka": "napum", "strI": "stri"}


def build(dcs_path: Path, data_dir: Path, max_stems: int) -> dict[str, object]:
    dcs = json.loads(dcs_path.read_text(encoding="utf-8"))
    verified = _kosha_basic_stems(data_dir)

    stems: list[dict[str, object]] = []
    for entry in dcs["noun_stems"]:
        slp1 = entry["slp1"]
        linga = entry["linga"]  # pum|napum|stri (DCS-dominant)
        kosha_lgs = verified.get(slp1)
        if not kosha_lgs:
            continue
        # keep only if the DCS-dominant liṅga is attested as a verified kośa liṅga
        if not any(_KOSHA_LINGA.get(k) == linga for k in kosha_lgs):
            # fall back to a verified liṅga the kośa does attest
            verified_linga = next(
                (_KOSHA_LINGA[k] for k in kosha_lgs if k in _KOSHA_LINGA), None
            )
            if verified_linga is None:
                continue
            linga = verified_linga
        nyap = linga == "stri" and slp1[-1:] in {"A", "I"}
        stems.append({"slp1": slp1, "linga": linga, "nyap": nyap, "freq": entry["freq"]})
        if len(stems) >= max_stems:
            break

    verb_freq = {v["slp1"]: v["freq"] for v in dcs["verb_lemmas"]}
    entries = P.Data(str(data_dir / "prakriya")).load_dhatu_entries()
    dhatus: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for e in entries:
        aup = e.dhatu.aupadeshika
        gana = str(e.dhatu.gana).split(".")[-1]
        key = (aup, gana)
        if key in seen:
            continue
        seen.add(key)
        root = _normalise_root(aup)
        dhatus.append(
            {
                "aupadeshika": aup,
                "gana": gana,
                "freq": verb_freq.get(root, 0),
                "transitive": root in _VERIFIED_TRANSITIVE,
            }
        )

    return {
        "schema": "generator-lexicon-v1",
        "stems": stems,
        "dhatus": dhatus,
        "n_verified_kosha_stems": len(verified),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dcs", type=Path, default=Path("data/lexicon/dcs_lexicon.json"))
    ap.add_argument("--data-dir", type=Path, default=Path.home() / ".cache/vidyut/data")
    ap.add_argument("--max-stems", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=Path("data/lexicon/generator_lexicon.json"))
    args = ap.parse_args()

    result = build(args.dcs, args.data_dir, args.max_stems)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=True), encoding="utf-8")
    ns = len(result["stems"])  # type: ignore[arg-type]
    nd = len(result["dhatus"])  # type: ignore[arg-type]
    ntr = sum(1 for d in result["dhatus"] if d["transitive"])  # type: ignore[index]
    nf = sum(1 for s in result["stems"] if s["freq"] > 0)  # type: ignore[index]
    print(
        f"wrote {args.out}: {ns} verified stems ({nf} DCS-attested), "
        f"{nd} dhatus ({ntr} transitive-capable)"
    )


if __name__ == "__main__":
    main()
