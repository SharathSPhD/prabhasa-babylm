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
from pathlib import Path

import vidyut.kosha as kosha_mod
import vidyut.prakriya as P

# DCS transitivity gate: a dhātu may license a karma only if its attested
# single-verb sentences contain an accusative at least this often, with enough
# evidence to trust the estimate. Tuned to match linguistic reality (bhū/as ~0.07
# intransitive; gam/dṛś/vad/khād 0.58–0.88 transitive).
_TRANS_SCORE_MIN = 0.40
_TRANS_EVIDENCE_MIN = 20

#: it-prefix markers (ṭu/ḍu/ñi etc.) that begin some upadeśas.
_IT_PREFIXES = ("qu", "wu", "qa", "wa", "Yi")


def _match_dcs_root(aupadeshika: str, dcs_verbs: dict[str, dict]) -> dict | None:
    """Resolve an upadeśa to its attested DCS verb entry, or ``None``.

    Uses the DCS clean-root verb set as the validator: strip accents/it-prefixes,
    then return the longest trailing-trimmed candidate that is actually attested
    in DCS. Under-matching is the *safe* direction — an unmatched dhātu defaults to
    intransitive and simply never licenses a karma.
    """
    base = aupadeshika.replace("\\", "").replace("^", "").replace("/", "")
    for pre in _IT_PREFIXES:
        if base.startswith(pre):
            base = base[len(pre) :]
            break
    base = base.rstrip("~")
    best: dict | None = None
    for k in range(4):  # trim up to 3 trailing it/marker chars
        cand = base[: len(base) - k] if k else base
        cand = cand.rstrip("~")
        if cand and cand in dcs_verbs:
            entry = dcs_verbs[cand]
            if best is None or len(cand) > len(best["slp1"]):
                best = entry
    return best


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


def _gana_attr_map() -> dict[str, str]:
    """Map each Gaṇa's ``str()`` value (e.g. ``BvAdi``) → its Python attribute
    name (e.g. ``Bhvadi``), which is what ``getattr(prakriya.Gana, ...)`` needs."""
    out: dict[str, str] = {}
    for name in dir(P.Gana):
        if not name[:1].isupper():
            continue
        val = getattr(P.Gana, name)
        if isinstance(val, P.Gana):
            out[str(val).split(".")[-1]] = name
    return out


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
            verified_linga = next((_KOSHA_LINGA[k] for k in kosha_lgs if k in _KOSHA_LINGA), None)
            if verified_linga is None:
                continue
            linga = verified_linga
        nyap = linga == "stri" and slp1[-1:] in {"A", "I"}
        stems.append({"slp1": slp1, "linga": linga, "nyap": nyap, "freq": entry["freq"]})
        if len(stems) >= max_stems:
            break

    dcs_verbs = {v["slp1"]: v for v in dcs["verb_lemmas"]}
    gana_attr = _gana_attr_map()
    entries = P.Data(str(data_dir / "prakriya")).load_dhatu_entries()
    dhatus: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for e in entries:
        aup = e.dhatu.aupadeshika
        gana = gana_attr[str(e.dhatu.gana).split(".")[-1]]
        key = (aup, gana)
        if key in seen:
            continue
        seen.add(key)
        matched = _match_dcs_root(aup, dcs_verbs)
        freq = matched["freq"] if matched else 0
        transitive = bool(
            matched
            and matched["trans_score"] >= _TRANS_SCORE_MIN
            and matched["single_sent"] >= _TRANS_EVIDENCE_MIN
        )
        dhatus.append(
            {
                "aupadeshika": aup,
                "gana": gana,
                "freq": freq,
                "transitive": transitive,
                "dcs_root": matched["slp1"] if matched else None,
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
