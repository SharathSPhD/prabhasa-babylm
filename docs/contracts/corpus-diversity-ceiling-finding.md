# Finding — synthetic prior diversity ceiling (Strict-Small corpus build)

Measured on the consolidated `integration/data-engine-v2` tree (GB10), vidyut 0.4.0.

## Observation

The Pāṇinian frame generator (`karaka_frames.enumerate_frames`) exhausts at
**74,760 unique frames** — a hard ceiling set by a small lexicon:

| lexicon | size |
|---|---|
| DHATUS (verb roots) | 8 (3 akarmaka) |
| NOMINAL_STEMS | 14 |
| LAKARAS | 4 |
| NUMBERS | 3 |
| OBLIQUE_KARMA / VERIFIED_OBLIQUE_FRAMES | 3 / 4 |

Realized, that is ~**260k words** of unique Pāṇinian Sanskrit (3.5 words/sentence).
Paribhāṣā shares this frame space, so it inherits the same ~75k-frame ceiling.
Dyck, by contrast, is combinatorially **unbounded** (random bracket sequences).

## Why it matters

The Strict-Small arms use a **1M-word pre-pretrain dose** (ADR-0036). At that size:

- **Dyck (arm C)** → ~1M *unique* tokens.
- **Paninian (arm B) / Paribhāṣā (arm D)** → ~260k unique words × ~4 repetitions.

So B/C/D would differ not only in prior **type** (the intended variable) but also in
prior **diversity** (a confound). A B-vs-C or D-vs-C effect could then be attributed
to "Dyck saw 4× more unique material," not to structure. This violates the
budget-matched fairness ADR-0036 is built on.

## The fork

**Option A — cap-and-match (cheap, runnable now).** Equalize *unique* count across
arms: cap every prior at ~260k unique words and fill the 1M dose by repetition for
all of B/C/D (Dyck included). Removes the confound by leveling diversity downward.
Honest and fair, but low absolute diversity and heavy repetition.

**Option B — expand the lexicon from real Sanskrit (richer; the "augment & utilise"
path).** Mine a large dhātu + nominal-stem inventory from **DCS** (CoNLL-U lemmas) and
**IndicCorp** `sa.txt`, keep only stems/roots that **realize through Vidyut**, and feed
them into the frame generator. Real Sanskrit has thousands of dhātus and tens of
thousands of nominal stems, lifting the frame ceiling far past 1M → Paninian and
Paribhāṣā reach ~1M *unique* words, matching Dyck's natural diversity with no
confound. More work (lexicon mining + Vidyut-realization validation + re-running the
ADR-0035 information-parity check on the expanded space), and aligns with the
directive to use the AI4Bharat/DCS corpora.

Recommendation: **B** — it both removes the confound and is the corpus you asked me to
build from the real data, with A as a documented fallback if expansion underdelivers.
