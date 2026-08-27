# Journal paper — story draft

**Status:** working draft, 2026-08-27. Nothing here is final.
**Target:** open — see §2. Leading candidates: *Nature Communications* (Article/Analysis),
*Nature Methods* (Analysis or Perspective).
**Scope:** the whole project. Two applications — ants (social immunity) and mice (three
ASD-associated lines) — used as **motivating settings** to lay out the requirements, challenges and
guidelines for scaling experiments with AI annotation.

> **Standing rule for this file.** Every number below is copied from project notes and result
> payloads as of the date shown. Before any of it enters a manuscript it must be **re-read from the
> JSONs / run configs**, not carried forward from here. Three claims on this project have already
> died on recompute.

---

## 1. What the paper is

Not "here is a method". The paper answers the question an experimental biologist actually arrives
with:

> *I have a video experiment, a limited annotation budget, and a vision model. What do I have to
> decide, in what order, and what am I allowed to claim at the end?*

The two applications are the motivating settings, not two case studies of one algorithm. They were
chosen because they fail differently, and between them they cover most of the decision space:

| | **Ants** (social immunity) | **Mice** (three ASD lines) |
|---|---|---|
| Scale | 7 cohorts, ~600 annotated observations, 351 with **no labels at all** | 2 cohorts, 24 of 72 cages annotated, second cohort **entirely unlabelled** |
| Hard part | **deployment and transfer** — new batches, new cohorts, a new *species* | **confounding** — the treatment is physically visible in frame |
| Regime | mostly out-of-distribution; PPI++'s guarantee does not hold | mostly in-distribution; PPI++ applies |
| What it teaches | when you may deploy, and how much transfer costs | when you need DERM, and why accuracy cannot validate an annotator |

That contrast *is* the paper's structure. Ants shows the deployment problem, mice shows the
confounding problem, and the guidelines are what generalises out of both.

**The five decisions** (§4) are the spine. **The regime map** (§5) — when PPI++ applies and what
you fall back to when it does not — is the central methodological contribution.

---

## 2. Target journal — the open decision

You are undecided, and the honest position is that the framing decides the venue, not the other way
round. The criterion:

> **How much of the paper is new measurement, and how much is synthesis of existing knowledge?**

On this project it is mostly **new measurement** — the treatment-leakage probe, the DERM validation
on the estimand with seed replicates and a negative control, the annotation learning curve, the
measured cost of transfer to a held-out cohort, the estimator regime boundary. None of that is in
the literature. That makes an **Article or Analysis** defensible rather than a Perspective.

| Venue | Fit | Cost |
|---|---|---|
| **Nature Communications**, Article/Analysis | Good *if* framed as a systematic study with new measurement in two systems. Broad readership is the right audience for "guidelines". | Must not read as a review. Needs a crisp claim in the abstract, not a list. |
| **Nature Methods**, Analysis | Arguably the most natural home: methods-comparison studies with recommendations are exactly their Analysis format. | Narrower readership; less reach into experimental biology at large. |
| **Nature Methods / Nat Rev Methods Primers**, Perspective/Primer | Best fit for the guidelines *as guidelines*. | Spends the new measurements as background. Primers are usually commissioned. |
| **eLife / PLOS Biology**, Tools & Resources | Comfortable with "we built and validated a pipeline, here is what we learned". | Lower prestige than the above for a methods contribution. |

**Recommendation:** write it as an **Analysis-style Article** — the format is the same for both of
the top two venues, so the target can stay open until the draft exists. Decide after the
introduction is written, on the basis of how strong the single-sentence claim reads.

---

## 3. Title and abstract sketch

**Title candidates**
1. *Scaling behavioural experiments with AI annotation: requirements, failure modes and guidelines*
2. *When can machine annotations carry a causal claim? Lessons from two behavioural experiments*
3. *From annotation budget to causal effect: a decision framework for AI-assisted experiments*

(2) is the sharpest question and the most likely to survive an editor; (1) says what the paper is;
(3) is the most useful to a reader. Current preference: **(2)**.

**Abstract skeleton**
1. Annotation is the rate limiter in behavioural science, and vision models promise to lift it.
   Whether they can depends on decisions that are currently made by folklore.
2. Using two experiments that fail in different ways — ants, where the problem is deployment across
   cohorts and species, and mice, where the intervention is physically visible in frame — we
   measure what actually breaks.
3. **Annotation budget, not architecture, is the binding constraint** (log-linear learning curve
   with no plateau; one doubling of labels beats every modelling intervention tried, combined).
4. **Prediction accuracy cannot validate an annotator used causally.** A model can be systematically
   wrong in the direction of the treatment; here a probe on behaviour-free frames identifies the
   intervention at 0.946 balanced accuracy, the resulting bias halves the estimated effect out of
   distribution, its magnitude is a draw of the random seed, and no accuracy metric shows any of it.
5. **The repair is confound-specific and its cost is predictable.** An environment-aware objective
   removes the bias where it exists (indistinguishable from zero across all cells and seeds) and
   does nothing where it does not — the negative control — at a small, statable cost in accuracy.
6. **The estimator you may use depends on the regime.** PPI++ delivers valid intervals only when
   labelled and unlabelled units are exchangeable; outside that — a new cohort, a new species, no
   labels at all — the broader prediction-powered causal inference family applies but the claims
   available shrink from magnitudes to signs and patterns. We map the boundary and state what can
   be bounded on each side of it.
7. We distil the whole into a decision procedure that does not require adopting our estimator.

---

## 4. The spine — five decisions, each with challenges and guidelines

This is the answer to "how to structure challenges and guidelines". **Do not organise by topic**
(data / models / estimators / results) and do not organise by our pipeline. Organise by **the
decisions the reader has to make**, in the order they have to make them. Each decision gets:
*what breaks → how you would notice → what to do*.

Every challenge below is backed by a measurement on this project. That is what makes the guidelines
evidence rather than opinion.

---

### Decision 1 — What to annotate, and how much

The first decision, made before any model exists, and the one with the largest effect on the
outcome.

**What breaks.**

1. **The annotated subsample is a sampling design, and it is usually accidental.** In mice v1, 18 of
   24 annotated cages are heterozygous against a balanced 216/216 experimental design — a 3:1
   enrichment. Any estimator that pools arms inherits it.
2. **The reference standard is unvalidated.** Six annotators, **0 of 432 observations
   double-annotated**, and most annotators saw only one genotype within a line. The measured
   per-annotator effect reaches **3.7×** — several times the genotype effect it was meant to
   measure. There is now no way to bound the reliability of the ground truth, and no amount of
   modelling recovers it.
3. **Annotator identity can be confounded with the contrast.** In mice it is confounded with
   genotype, which is what compromised the genotype contrast entirely.
4. **Labels are the binding constraint, and it is quantifiable.** Nested learning curve over
   5/10/15/20 annotated cages: macro AP 0.2716 → 0.3461 → 0.3837 → 0.4289, log-linear at
   R² = 0.993, **no plateau anywhere in the affordable range**. Each doubling is worth ≈ **+0.076
   macro AP**, against **+0.033** for the best unsupervised intervention tried, and ≈ 0.009 for
   pure seed noise.

**How you would notice.** Cross-tabulate who annotated what against every experimental factor,
before training. Fit the learning curve before the architecture search.

**Guidelines.**
- **G1.1** Treat annotation as a **sampling design**: randomise or explicitly balance which units get
  annotated, against the same factors the experiment balances.
- **G1.2** Budget a **double-annotated subset up front**. A few percent of the labelling effort is
  the only thing that ever bounds your reference standard, and it is unrecoverable afterwards.
- **G1.3** Record annotator identity per unit and check it against the treatment. If annotator is
  confounded with the contrast, the contrast is not estimable — with or without AI.
- **G1.4** Measure the learning curve early and quote it **in doublings**. If it has not plateaued,
  more labels dominate more modelling. Say so in the paper rather than burying it.
- **G1.5** Convert every modelling gain into its **equivalent number of annotated units**. It is the
  only currency in which an experimentalist can compare "buy a GPU week" against "pay an annotator".

---

### Decision 2 — Which estimand, and can the design carry it

**What breaks.**

1. **The scientific question and the estimable quantity are not the same thing.** The mice programme
   is about genotype; in v1 genotype is between-cage, so it is confounded with cage, cohort, date
   and annotator. What survives is the **within-cage phase transition**: every cage contributes all
   three phases × two odours, same day, same animals, and 22 of 24 annotated cages have a single
   annotator — so cage, genotype, sex *and annotator* cancel by construction.
2. **The unit of analysis is not the frame.** Genotype is constant within a cage, so cages are the
   unit; a standard error computed on 144 observations when the design supplies 24 cages
   manufactures precision that does not exist.
3. **The outcome definition is a lever.** Bout counts, occupancy and timing are three different
   estimands with different resolvability. Choosing between them by how many contrasts come out
   significant is selection on significance; the defensible argument is mechanistic (duration is
   null in every cell, so the effect is on initiation; **49% of nose-to-nose bouts and 22% of
   nose-to-tail bouts are a single frame** at 5 fps, so duration is barely resolvable; occupancy is
   heavy-tailed, with the longest 10% of bouts carrying ~40% of the time).
4. **The time window is the largest single lever on the headline.** 11 of 12 phase × odour cells
   show significant within-phase decay, rates falling 4–6× across a recording, and habituation runs
   30 min against 15 for the other phases. Under three defensible windows one contrast reads
   **+0.47 (full), −0.50 (first 5 min), +0.98 (last 15 min)** — a sign flip. Only one of four
   contrasts survives all three.

**The ants side does this explicitly and it is worth showing.** The deployed ants model conditions
on `W_batch`, `W_annotator`, `W_weak_marking` and *excludes* nestbox, because batch determines
nestbox deterministically — a nuisance carrying no independent variation is not an extra
environment, it is the same one twice.

**Guidelines.**
- **G2.1** State the estimand before touching the model. Prefer a **within-unit contrast**: it
  cancels experimental *and* annotation nuisances in one move, and (see §5) it also kills the
  model's additive miscalibration.
- **G2.2** The unit of analysis is the unit of randomisation.
- **G2.3** Enumerate what varies *within* that unit. What does not vary has been cancelled; what
  does is still live.
- **G2.4** Pre-register the outcome definition and the time window on **biological** grounds, then
  report the sensitivity of the headline to both in full.
- **G2.5** Justify the outcome by measurability and mechanism, never by how many contrasts it
  resolves.

---

### Decision 3 — Which annotator to train, and how to validate it

**What breaks.**

1. **The AI can read the treatment instead of the behaviour.** In the mice cage a physical scent bag
   sits in a corner during the exposure phase. A leave-one-cage-out linear probe on a 32×32
   greyscale thumbnail of a **behaviour-free** frame — no scored behaviour, ≥5 s from any bout —
   separates exposure from non-exposure at **0.946 balanced accuracy** (chance 0.500). It localises
   where you would predict: bottom-left quadrant 0.903, border alone 0.951, centre 0.656, the same
   corner in all four probed cages. The **negative control is clean**: fear vs social exposure,
   which differ in odour but not in apparatus, reads **0.568** — the protocol is visible, the odour
   is not.
   ERM has no term discouraging this. The phase prior is genuinely informative for frame
   classification, so ERM is *correct* to use it and *wrong* for the causal purpose. Held out to an
   unseen exposure the resulting annotator **halves the true effect** (−0.314 bouts/min where the
   human labels say −0.658), and the size of the error is **a draw of the random seed**
   (+0.344 / +0.175 / +0.033 across three seeds on the same cell).
2. **Accuracy ranking and causal usefulness are anti-correlated.** The concrete instance: the arm
   that won macro AP was nearly useless downstream (r 0.214, interval crossing zero, 2–4% variance
   reduction), while an arm 0.003 AP *below* it was by far the best model for the estimate
   (r 0.582 [0.368, 0.765]). Selecting on AP would have chosen the worse model for the goal.
3. **The prediction metric can itself be the artefact.** Frame-exact AP spends much of its dynamic
   range on annotation boundary jitter when half of all bouts are one frame long.
4. **Thresholds tuned in probability space generate the artefact you then report.** Two real
   failures here: (a) an operating threshold chosen by max-F1 **on the very split it then scores**
   is an oracle, and it varied 0.80–0.95 by run, so those numbers were not mutually comparable;
   (b) DERM's probabilities are compressed against 1 by construction, so a threshold grid stopping
   below 1.0 mis-set its rate by up to **−22%**, landing straight in the quantity under test and
   in the direction of making DERM look wrong. Rate-matching the threshold is what resolved the
   entire ERM-vs-DERM question.
5. **Transfer does not degrade uniformly across metrics.** On the ants fine-tuning cohorts the
   deployed model reaches balanced accuracy 0.964 and 0.961; on the held-out cohort it falls to
   0.876 — but **precision falls furthest, 0.892 / 0.909 → 0.711**, while recall holds at 0.856.
   For a rate-based outcome a precision collapse *is* a rate inflation, which is exactly what the
   rectifier then has to absorb.
6. **Any upstream detector is part of the annotation chain.** The ants point-of-view crops come from
   an HSV colour tracker whose recall is 0.96–1.00 across cohorts but whose **precision is
   0.23–0.40** — it over-detects heavily, and every downstream embedding inherits that.
7. **What fine-tuning buys is domain recalibration, not new computation.** BitFit — biases,
   LayerNorm and LayerScale gains only — reaches **0.4902 macro AP with 24,576 trainable
   parameters**, matching full fine-tuning's **0.4889 with 14,180,352** (577×). And the
   hyperparameter nearly inverted the conclusion: BitFit reads 0.4509 at the inherited encoder LR
   of 1e-5 and 0.4902 at 1e-3.
8. **Where you look beats how big your model is.** Ants per-individual point-of-view crops beat
   whole-frame input on the held-out cohort (best test balanced accuracy **0.869 vs 0.833**), across
   every sweep. The mice mirror image is the resolution limit: an animal occupies ~2.2 patches
   whole-frame at the stored resolution, which is why per-animal attribution fails there.
9. **Unlabelled in-domain data is worth about one modelling trick, for free — and does not
   compose.** Domain-adaptive self-supervised pretraining on 374,400 unlabelled frames gives
   **0.4622 vs 0.4289** for a matched control (3.7× the 0.0089 seed spread) with zero labels. But
   SSL-init + BitFit (0.5127) is *below* BitFit alone (0.5409): both are doing domain
   recalibration, so they substitute rather than add.

**Guidelines.**
- **G3.1** **Probe for treatment leakage before training anything.** A linear probe on
  behaviour-free frames, leave-one-unit-out, plus a **negative control** contrast that shares the
  apparatus. Report both. Minutes of compute, and it determines everything downstream.
- **G3.2** Prefer removing leakage **physically**: mask the region, or randomise the apparatus.
  Blanking the leaking corner takes the probe from 0.946 to 0.657. Generic augmentation is **not** a
  substitute — D4 augmentation randomises *where* the cue is, not whether it is present (border
  alone still reads 0.951).
- **G3.3** **Select the annotator on the estimate, not on prediction accuracy.** Define an explicit
  estimand-bias diagnostic — predicted effect minus human-label effect on held-out units — and make
  it the promotion criterion.
- **G3.4** Report the quantity that governs the downstream gain: the per-unit correlation *r*
  between predicted and true outcome.
- **G3.5** Never tune an operating threshold on the split you score. Use leave-one-fold-out
  selection, and **rate-match** rather than optimise F1 when comparing models whose probability
  scales differ.
- **G3.6** Report recall and precision **separately**; they do different downstream damage.
- **G3.7** Give any upstream tracker or segmenter its own measured error budget. A tracked crop is
  not raw data.
- **G3.8** **Crop to the subject before scaling the model.** Effective resolution on the animal is a
  bigger lever than encoder choice.
- **G3.9** Adapt cheaply — try bias-only tuning before full fine-tuning — and **re-tune the learning
  rate** whenever the trainable parameter count changes by orders of magnitude.
- **G3.10** State what your model-selection gate compared against. "Better than its control" is not
  "better than the alternatives".

---

### Decision 4 — Whether you need DERM, and what it costs

The question this section answers is **when**, not whether. DERM is a targeted correction, not a
general-purpose regulariser, and the evidence says so in both directions.

**When it applies.** The environment (here, the treatment phase) must **vary within the unit of the
contrast**. In mice, phase varies within all 24 cages, annotator within 2, and line/sex/genotype
within none — so phase is the only environment whose shift can reach a within-cage contrast, and
also the only one that could overshoot. `--env-key annotator` cancels and is no substitute.

**What the evidence shows.**
- **Out of distribution** (trained on one exposure, tested on the other), DERM's bias is
  indistinguishable from zero in **all 12 cells** (3 seeds × 4 cells, max |mean| 0.18, every
  interval covering zero) against ERM's +0.344 worst case. Seed-averaged paired test on the cell
  where ERM has bias to remove: ERM +0.184 vs DERM −0.015, **p = 0.0023**.
- **It makes the estimate reproducible.** Across-seed SD on that cell falls from 0.156 (ERM) to
  0.038 (DERM). With ERM the acquired bias is luck; DERM removes the channel.
- **In distribution**, over 48 paired cross-fitted units, bias as a share of the true pooled effect
  falls on **both** behaviours: nt 1.03× → 0.88× (p 0.0026), nn 1.27× → 0.85× (p 0.0136). On a
  second backbone: nt 1.94× → 0.19× (p 0.0124), nn 0.27× → 0.31× (p 0.82). **DERM never makes bias
  worse in any of the four**, and is null exactly where ERM was already unbiased.
- **The negative control is the strongest single piece of evidence.** Trained in the exposure
  direction where the confound is nearly absent (prevalence ratio 0.8× rather than 3.1×), every
  paired cell is null and DERM ≡ ERM. The correction is **confound-specific**.
- **Weights must be population prevalences, not sample ones.** Weights estimated on the balanced
  training subsample collapsed the environment-variance ratio the objective is built on and trained
  the correction at half strength — and, on one behaviour, introduced harm that the corrected
  weights removed entirely.
- **The price is predictable and must be stated as a price.** Macro AP 0.4107 → 0.3878 (stock
  encoder), 0.3819 → 0.3640 (SSL encoder). A model that has stopped using an informative prior
  *must* be worse at frame classification.

**Two honest cuts to keep** (do not let these get polished away):
- **The correction is only as large as the confound in its own training distribution.** Per-phase
  prevalence ratios are 3.1× / 2.5× in one exposure and 0.8× in the other; pooling both dilutes to
  1.4×, which is why the deployment effect is real but modest.
- **It is a population-average correction, never a per-recording guarantee.** On one cell DERM is
  nearer zero in only **20 of 48 units** while the mean improves sharply — structurally expected,
  since one shift per environment can only move a mean, but it must be said that way.
- **Unexplained:** why ERM on one backbone carries ~4.7× less bias than on another (0.27× vs 1.27×).
  "Nothing left to correct" fits but is post-hoc. Do not quote it as mechanism.

**Guidelines.**
- **G4.1** Use DERM when (i) the leakage probe is positive **and** (ii) the environment varies within
  the unit of the contrast. If either fails, it will do nothing — and the negative control shows it
  does no harm either.
- **G4.2** Use **population** prevalence weights, not weights estimated on the annotated subsample.
- **G4.3** Expect and report the accuracy price. Judging the correction on accuracy is judging it on
  the very quantity it deliberately gives up.
- **G4.4** Validate on the estimand, with seed replicates. A single seed's p-value is not a result
  here — ERM's own bias varied from +0.344 to +0.033 across seeds.
- **G4.5** Claim a population-average correction, not a per-unit one.

---

### Decision 5 — Which estimator, and what you are allowed to claim

Promoted to its own section, §5, because it is the paper's central methodological contribution.

---

## 5. The regime map — PPI++, PPCI, and the label-shift boundary

**The distinction the paper should make explicit, and which the literature blurs:**

- **PPCI** is the broad family: solving a *causal* inference task with AI annotations.
- **PPI++** is one specific estimator inside it, and its confidence-interval validity holds only
  **in distribution** — labelled and unlabelled units must be exchangeable.

Almost every real deployment eventually leaves that regime: a new batch, a new cohort, a new
species, a cohort with no labels at all. So the useful thing for a reader is a **map**, not a
recommendation.

### Regime A — labelled and unlabelled units exchangeable

*Mice v1 (24 of 72 cages annotated, rest unlabelled from the same cohort). Ants within-cohort.*

Here PPI++ applies and gives valid intervals whatever the model gets wrong. **Recommended
combination: DERM-trained annotator + PPI++.**

What has to be right:
- **Never plug in raw AI labels.** On one headline cell: classical (human) +0.650, naive plug-in
  +0.428 — the correction is +0.222, i.e. the plug-in is **~34% too small**.
- **Vanilla PPI (λ = 1) is catastrophic under miscalibration** — simulated SD **20–43× classical**
  at this design. It stays unbiased; it is simply useless. This is the crisp answer to "is
  calibrating the model cheating?": power-tuned λ drives itself to ≈ 1/13, performing the
  calibration in the one place where getting it wrong costs *variance* rather than *validity*.
- **One shared, leave-one-out λ.** Per-stratum λ undercovers (0.912–0.923) because it overfits the
  covariance on a six-unit arm; shared-LOO gives 0.946 / 0.940 / 0.945. The slope describes the
  **predictor**, not a stratum — transport it, do not refit it.
- **Cross-fit.** Predictions on labelled units must be out of fold, or the rectifier is fitted on
  the model's own training data.
- **Quote the honest gain.** It is **1 − r²/(1 + n/N)**, not 1 − r²; at n/N = 1 that halves the
  advertised gain. Then convert it to *equivalent additional annotated units*.
- **Guard small strata.** A λ denominator computed on labelled units only hit a two-unit stratum
  whose predicted deltas differed by one float64 ULP: variance 2.5e-32 passed a bare `> 0` check,
  λ reached 5e14, the estimate 2.6e14 bouts/min. Computing the variance over all n + N units makes
  λ correctly fall to zero and the estimator return exactly classical. **Falling back to classical
  is the correct behaviour; producing a number is not.**

**The identity that marks the regime boundary.** For a within-unit contrast with n labelled and N
unlabelled units,

```
PPI++            = classical + λ  · (mean_N(Df) − mean_n(Df)),   λ = β/(1 + n/N)
PPCI + intercept = classical + β  · (mean_{n+N}(Df) − mean_n(Df))
                 = classical + β·N/(n+N) · (mean_N(Df) − mean_n(Df))
```

and `β·N/(n+N) = λ`. **Inside Regime A the two coincide exactly** (verified numerically to nine
decimals), and PPCI-as-usually-written differs from PPI++ by precisely the fitted intercept — *the
intercept is the rectifier*. This is worth stating not to claim the methods are interchangeable, but
because it **defines the edge**: the thing PPI++ adds over plug-in PPCI is exactly the quantity that
requires labelled data from the target distribution. When that data does not exist, that is the
term you lose.

Second consequence, and the reason a badly calibrated model is survivable: because the estimand is a
**within-unit difference**, an affine calibration's additive offset cancels (`D_Y = b·D_f`). The
calibration has exactly **one** free parameter, the slope, so the model's several-fold
over-prediction of absolute rate never reaches the estimate.

### Regime B — deployment under label shift

*Mice v2 (36 cages, zero labels). Ants v6 (175 observations, no labels). Ants vA (176 observations,
no labels, **and a different species**, Lasius niger).*

Here there are no target-distribution labels, so no rectifier can be fitted on the target. The slope
must be **transported** from the source cohort, and that assumption is **untestable on the target**.
PPI++'s guarantee is gone. What remains:

- **Claims shrink from magnitudes to signs and patterns.** An uncalibrated estimate is on the
  *model's* scale, not the behaviour scale, and must never share an axis with a calibrated one.
- **Sign agreement against the classical estimator is the available validation** where a labelled
  cohort exists alongside — currently 7 of 8 headline cells in mice v1, with the single miss on a
  cell whose classical value is ≈ 0.033 bouts/min, i.e. indistinguishable from no effect.
- **A held-out *annotated* cohort is the only way to measure what transfer costs.** Ants v5 is the
  model example on this project and should be presented as the template: fine-tune on v3+v4, test
  on v5 with ground truth, and read the damage directly (balanced accuracy 0.964/0.961 → 0.876,
  precision 0.892/0.909 → 0.711). Every deployment claim on v6/vA rests on that measurement.
- **Label-free sanity checks bound the failure without annotations.** Predicted-positive rate per
  behaviour against the known range from annotated cohorts is the cheapest one: ants v5 sits at
  ~0.21–0.25 true and the model reproduces it, so a rate far outside that range on vA signals
  transfer failure with no ground truth needed.

**This is the frontier, and the paper should say so.** Bounding uncertainty under label shift, with
no target labels, is not solved here. What the paper can honestly offer is: the regime map, the
measurement template (a held-out annotated cohort), the label-free sanity checks, and a clear
statement of which claims survive. Proposing that as an open problem *with* a worked example of how
far you can get is a stronger contribution than pretending it is closed.

**Guidelines.**
- **G5.1** **State the regime.** Are your labelled units exchangeable with your target? The answer
  determines which estimator is legitimate and which claims you may make.
- **G5.2** In Regime A: DERM + PPI++, cross-fitted, shared-LOO λ, honest gain formula, explicit
  small-stratum fallback to classical.
- **G5.3** In Regime B: report signs and patterns, keep scales on separate axes, and never present
  an uncalibrated magnitude as a behavioural quantity.
- **G5.4** **Reserve one annotated cohort as a transfer test set**, chosen to differ from training in
  the way your real deployment will. It is the only measurement that licenses deployment onto
  unlabelled cohorts.
- **G5.5** Run label-free sanity checks (predicted rate against known ranges) on every unlabelled
  cohort, and pre-declare what value would count as failure.

---

## 6. Evidence ledger — what each system can actually carry

| Claim | Ants | Mice |
|---|---|---|
| Annotated units | v1 44, v2 44, v3 212, v4 113, v5 190 (partial); v6 175 and vA 176 **unannotated** | v1 24 of 72 cages; v2 **0 of 36** |
| Deployed annotator | **DERM**, DINOv2 class token, pretrain v2 → ft v3+v4 → test v5 | **DERM** cross-fit, 3 folds over 24 annotated cages |
| Held-out performance | bacc 0.876 on v5 (precision 0.711) | macro AP ≈ 0.39 (DERM), ≈ 0.51 (BitFit-6) |
| Treatment-leakage probe | **not run** | 0.946, localised, clean negative control |
| DERM vs ERM on the estimand | **not measured** | measured in and out of distribution, seed-replicated |
| Rectified estimate | ATE numbers **not in this repo** — see §7 | full grid, 3–4 predictors, both cohorts |
| Annotation learning curve | not measured | measured, log-linear, no plateau |
| Inter-annotator reliability | unknown — `W_annotator` is recorded, so checkable | **0 of 432 double-annotated** |
| Regime B evidence | v6 + vA = 351 observations, one across a **species boundary** | v2 = 36 cages, zero labels |

**Read this as the work plan.** Mice carries Decisions 1, 3 and 4 and Regime A. Ants carries
Decision 5 / Regime B and the transfer measurement. The blank cells are §7.

Two structural facts to know before writing:
- **The ants ATE numbers are not in this repository.** `notebooks/ppci/ants.ipynb` has **zero saved
  cell outputs** and no script computes effects — `compute_ate` is only ever called from the
  notebook. The real published figures appear to live in adjacent repos under
  `/fs3/group/locatgrp/rcadei/` (`causal-lifting/`, which holds `ppci.ipynb` and
  `results/ants/{main,cfl,trex,generalization,tracking}/`, and `ISTAnt/`).
- **`CLAUDE.md` is stale on code layout.** Live PPCI code is `src/ppci/`; the modules CLAUDE.md
  describes (`src/data.py`, `model.py`, `train.py`, `causal.py`) moved to `src/old/`, and
  `scripts/run_preprocess_eci.py` does not exist. Fix before anyone writes Methods from it.

---

## 7. Experiments to plan (none run yet)

Ordered by value per unit of effort. **Nothing here has been executed** — this is the proposal list.

| # | Experiment | Why it matters | Rough cost |
|---|---|---|---|
| 1 | **Ants leakage probe.** Quiet-frame linear probe, treated vs control, leave-one-unit-out, plus a negative control sharing the apparatus. | Decides whether treatment leakage is a *general phenomenon* or a mouse anecdote. **Either result is publishable**: a positive makes it general; a null tells readers when they are safe and gives a cross-system negative control. | minutes |
| 2 | **Consolidate the ants effect estimates** into this repo from `causal-lifting/` / `ISTAnt/`, as a script rather than a notebook. | Without it the paper's ants numbers have no reproducible home and the data-availability statement is hard to write. Cheap now, expensive at revision. | hours |
| 3 | **Ants inter-annotator check.** `W_annotator` is recorded — find out whether any observation was scored twice. | If ants has double annotation and mice does not, that contrast alone demonstrates G1.2 instead of asserting it. | hours |
| 4 | **Ants DERM-vs-ERM on the estimand**, if #1 is positive. | Otherwise the paper reports a general problem with a single-system solution. | days |
| 5 | **Finish the 2×2** — the BitFit-6 × DERM cross-fit already launched. | It is the only arm that can settle whether DERM or the stronger backbone is the better annotator, rather than trading wins across outcomes. Currently the honest statement is that the promotion gate never compared them, and head-to-head the backbone wins 2 of 3 outcomes. | already running |
| 6 | **Label-shift bounding, on ants v5.** Use the one cohort with both a distribution shift and ground truth to test candidate bounds. | Turns §5 Regime B from "here is the open problem" into "here is the open problem and one honest attempt". Highest-risk, highest-reward item. | weeks |
| 7 | **Mice double annotation**, ~10 observations. | The only way to bound the reference standard. Small, and it never gets cheaper. | annotator time |
| 8 | **End-to-end coverage under known truth** for the whole pipeline, not just the estimator in isolation. | A causal-methods reviewer will ask. | days |

---

## 8. Figure plan

| Fig | Content | Status |
|---|---|---|
| **1** | **The decision tree.** Five decisions, with the regime split at the end: exchangeable labels → PPI++ + DERM, quantitative claims; label shift → PPCI, signs and patterns. This figure *is* the paper. | to draw |
| **2** | **What breaks, measured.** One panel per decision, each showing the single measurement that motivates it — annotation curve, leakage probe, AP-vs-estimand anti-correlation, DERM bias with negative control, transfer damage. | data exists except ants probe |
| **3** | **Treatment leakage and its repair.** Probe accuracy + localisation + negative control; then ERM vs DERM estimand bias per seed against human truth; then the AP price. | mice yes, ants pending |
| **4** | **Deployment across regimes.** Both organisms: classical / PPI++ / PPCI with interval widths, and the unlabelled cohorts shown as sign-and-pattern only. | mice done, ants pending §7.2 |
| **5** | **Annotation economics.** Learning curve in doublings, with every modelling intervention plotted on the same axis in the same units. The most useful figure in the paper. | mice data exists |
| **Box 1** | The guidelines, one line each, grouped by decision. The thing readers screenshot. | from §4–5 |

---

## 9. Section outline

1. **Introduction** — annotation is the rate limiter; AI promises to lift it; the promise is
   conditional and the conditions are measurable, not folklore.
2. **Two settings that fail differently** — ants and mice, and why the pair covers the space.
3. **Decision 1 — What and how much to annotate.** Learning curve; the annotation sampling design.
4. **Decision 2 — Which estimand the design can carry.** Within-unit contrasts; outcome and window.
5. **Decision 3 — Training and validating the annotator.** Leakage probe; why accuracy fails as a
   criterion; thresholds; transfer damage; cheap adaptation.
6. **Decision 4 — When DERM is needed.** Evidence, negative control, price, boundaries.
7. **Decision 5 — The regime map.** PPI++ in distribution; PPCI under label shift; the identity that
   marks the boundary; what can be bounded on each side.
8. **Deployment at scale.** Both organisms, including 351 ant observations and 36 mouse cages with
   no human labels, and one cohort across a species boundary.
9. **Discussion — guidelines** (Box 1), scope, and the open problem of bounding uncertainty under
   label shift.
10. **Methods.**

---

## 10. Deliberately out of scope

- **ECI / NES (unsupervised discovery).** One sentence in the discussion as outlook. It is a
  different claim with a different method, already on arXiv, and carrying two methodological stories
  in one Article weakens both.
- **Per-animal identity attribution.** It does not work here; the honest reasons are that the derived
  data drops the per-animal indices and an animal occupies ~2.2 patches at the stored resolution.
  One paragraph in limitations — it is genuinely useful to readers as *the* barrier to
  individual-level outcomes — but it is not a result.
- **Genotype as intervention.** Confounded with annotator in this dataset. Effect modification only,
  and stating that boundary is part of the contribution.

---

## 11. Open questions

1. **Target** — stays open until the introduction exists (§2). Decide on how strongly the
   single-sentence claim reads, not in advance.
2. **How hard to push the conditional (line × genotype × sex) effects?** A headline result, or a
   demonstration that the machinery reaches strata the human labels cannot?
3. **How much of the ants biology to tell.** The paper currently uses ants for deployment and mice
   for confounding; if the ants effects are a story in their own right that changes the balance.
4. **Co-authorship and data availability** for both experimental labs — decide early, it constrains
   what can be shown, and ants v3/v4/v5 are marked "to be released".
5. **Is "DERM + PPI++" claimed as a recommendation or as a finding?** Currently the evidence supports
   it as a recommendation in Regime A; the head-to-head against the stronger backbone is unfinished
   (§7.5).
