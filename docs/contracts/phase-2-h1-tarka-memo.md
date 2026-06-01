# Phase 2 — H1 Tarka Memo (Integrity Gate)

*Adversarial self-review of the Phase 2 H1 finding, as required by Layer 3 of the
Ralph-loop closure contract. The standard is not "find objections and flag them" —
it is "find the strongest objection and either show it is not fatal, or fix it."
A **null** finding raises the bar: the strongest objections are that the test was
unfair, underpowered, or mis-specified, so that the prior never got a fair chance.*

## The finding under attack

> **H1 (proxy scale) = NEGATIVE / NULL.** At 100M parameters, a matched-epoch
> Pāṇinian structural prior (arm B) gives **no measurable advantage over a
> surface-statistics-matched Dyck control (arm C)** on COGS argument-role
> acquisition, on **either** axis:
> - *Final state* — role-discrimination saturates for all arms (A 0.91–0.99
>   across four corruption tiers; `phase2-h1-cogs-calib.json`).
> - *Sample-efficiency* — discrimination learning-curve AUC: A=0.888, **B=0.905,
>   C=0.902**, Δ_auc(B−C) = **+0.003 (95% CI −0.006..+0.014)**, early-checkpoint
>   gap −0.011, no T_θ separation (`phase2-h1-cogs-se-100m.json`, ADR-0016).
>
> Generic structural pretraining (B *and* C alike) lifts AUC ~+0.015 over the
> no-pretrain baseline (A); the **Pāṇinian content specifically adds nothing**
> over the bracket-language control.

## Objection 1 (the fatal-candidate): the test was underpowered / the readout never entered a measurable regime

A null is worthless if the instrument couldn't have detected an effect. Two
concrete failure modes: (a) the metric floors/ceilings so no arm separates, or
(b) 3 seeds give CIs too wide to conclude anything.

**Resolution — the regime was measurable, by construction and in fact.** The
whole three-instrument arc exists *because* we refused to report an
un-interpretable number: EM floored (0.02) and a single swap ceilinged (0.97), so
neither was reported as the H1 result. The learning-curve readout was
pre-registered (ADR-0016) precisely to sit off both bounds, and it did: baseline A
crossed θ=0.80 in the **0.05–0.1 budget region**, *after* the very-early
checkpoints — the `UNRESOLVED_AT_PROXY` and `NOT_LEARNED_AT_SCALE` branches both
**failed to trigger**, which is the pre-registered evidence that the curve is
resolvable. On power: the per-seed AUCs are tight (A 0.883–0.896, B 0.889–0.920,
C 0.888–0.912) and the B−C CI half-width is ~0.010 — narrow enough that a +0.02
effect (the pre-registered bar) would have been detected. The null is *measured*,
not *absent*.

## Objection 2: the prior was tested on a capability the baseline already has — an unfair venue

The calibration showed the **no-prior** baseline saturates role discrimination at
≥0.90. If the venue has no headroom, of course the prior shows nothing — so the
null is an artifact of venue choice, not evidence against the prior.

**Resolution — this is the finding, correctly scoped, not a confound.** The
result is explicitly *"no advantage on argument-role composition at proxy scale,"*
and the reason — the baseline already composes — is reported as the headline, not
hidden. The honest consequence is a **scope correction, applied below**: H1's null
is bound to *this capability at this scale*, and the live program claim is
**relocated** to the segment where the baseline does *not* get the answer for free
(facts + traced reasoning, Stage-2), per the sample-efficiency brief §5 and the
crystallization charter. We do not over-claim a universal negative.

## Objection 3: the Dyck control is unfairly strong (it secretly encodes the same structure)

If the matched-statistics Dyck control already supplies hierarchical structure,
then B≈C is rigged: both arms get "structure," so the comparison can't isolate
Pāṇinian content.

**Resolution — that is the control working exactly as designed, and it is the
scientifically conservative choice.** The pre-registered H1 contrast (ADR-0013) is
B-vs-C *specifically* to attribute any advantage to **Pāṇinian grammatical
content** over **generic bracket-matched hierarchical structure**. The observed
pattern — A < (B ≈ C) — is the most informative possible outcome: it localizes the
small lift to *generic structure*, not Sanskrit-specific annotation. Had we used a
weaker control (e.g. shuffled tokens, arm H), a B>H result would have been a false
positive attributable to structure-in-general. The strong control is the integrity
feature, not the bug.

## Objection 4: insufficient dose (3.9%) starved the prior

The structural dose was ~3.9% of the token budget; perhaps a larger dose would
reveal an effect.

**Resolution — dose is not the binding constraint here, and this is measured.**
The matched-epoch dose was pre-registered as the fair design (ADR-0014), repeating
the diversity-capped set equally for B and C so neither is advantaged. More
decisively: the calibration proved the baseline reaches role competence **with no
structural dose at all** — so additional dose cannot move an axis the downstream
data already saturates. (This is also why the crystallization track is explicitly
**decoupled** from H1: it lifts the dose *ceiling*, which H1 showed is not the H1
bottleneck.)

## Objection 5: metric / threshold gaming (researcher degrees of freedom)

Were AUC-as-primary, θ=0.80, the +0.02 bar, the distractor corruption, or the
checkpoint grid chosen after seeing the B-vs-C numbers?

**Resolution — no; every degree of freedom was fixed blind and committed before
the comparison.** ADR-0016 (and its two amendments — AUC-primary and the finer
early checkpoints) were committed to git **before** the A/B/C run produced any
B-vs-C value; the corruption operator was selected from the A-only calibration
sweep (which reveals no B-vs-C); the verdict logic, thresholds, and branch mapping
are all in the ADR. The amendments were motivated by an *independent* fact (final
saturation) and a *statistical* concern (T_θ censoring), not by peeking.

## Comparison-fairness check

B and C share identical model, tokenizer, downstream data, step budget, dose
(matched-epoch), and checkpoint grid; they differ only in the *content* of the
structural pre-pretraining stream (Pāṇinian kāraka frames vs surface-matched
Dyck). The single controlled variable is grammatical content. Fairness holds.

## Verdict

No fatal confound. One scope correction applied (Objection 2): the finding is a
**proxy-scale, argument-role-composition** null — *"a Pāṇinian structural prior
confers no measurable advantage over a matched Dyck control for compositional
argument-role acquisition at 100M, on final accuracy or sample-efficiency"* — and
is **not** a universal claim about the prior. The null is honestly earned,
adequately powered, and measured in a resolvable regime. **H1 closes NEGATIVE at
proxy scale.** The live program claim relocates to the Stage-2 data-format thesis
(crystallization charter), and Phase-3 scale is reserved for that claim, not spent
to rescue a proxy-scale null on a capability the baseline already has.
