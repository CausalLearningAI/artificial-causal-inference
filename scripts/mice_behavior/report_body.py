# Consumed by build_report.py via exec(). Expects `img` (data URIs), `CHART` (the interactive
# figure, already JSON-injected) and `E` (estimates.json, for inline numbers).
n_lab = max((c['n_lab'] for c in E['cells'] if c['exp'] == 'v1' and c['method'] == 'classical'),
            default=24)

BODY = f'''
<div class="wrap">

<header class="top"><div class="measure">
  <p class="eyebrow">Mice v1 / v2 &middot; status &middot; 23 August 2026</p>
  <h1>Genotype under hormonal exposure</h1>
  <p class="lede">Three ASD-associated mouse lines, wild-type against heterozygous
  littermates &mdash; the mutation is not expressed in the wild type and is expressed in the
  heterozygote, filmed before, during and after two hormonal exposures. The
  programme asks how the genotype changes social behaviour. This report covers the step
  in front of that:
  the <b>average</b> effect of the exposure itself, pooled over genotypes &mdash; and the vision
  model that has to carry it to the 84 pools nobody has annotated.</p>
</div></header>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">01 &middot; Experiments</p>
  <h2>Two cohorts, one recording protocol</h2></div>
  <p>Both cohorts run the same six recordings per pool of four littermates: three phases in fixed
  order &mdash; <b>H</b>abituation (30 min) &rarr; <b>O</b> exposure (15 min) &rarr; <b>P</b>ost
  (15 min) &mdash; crossed with two hormonal exposures, <b>fear</b> and <b>social</b>. All six
  share a cage, a day and an annotator. Video is 2064&sup2; at 30 fps, stored 512&sup2; at 5 fps.</p>
</div>
  <div class="figwrap"><div class="scroll"><table>
    <thead><tr><th></th><th>v1</th><th>v2</th></tr></thead>
    <tbody>
      <tr><td>pools &times; observations</td><td>72 &times; 6 = 432</td><td>36 &times; 6 = 216</td></tr>
      <tr><td>design</td><td><b>12 per line &times; genotype</b><br>3 &times; 2 &times; 12 = 72</td><td><b>12 per line</b><br>3 &times; 12 = 36</td></tr>
      <tr><td>genotype</td><td>pure cage: 36 wt, 36 het</td><td>mixed cage: <b>3 wt + 1 het</b> per cage</td></tr>
      <tr><td>strata</td><td class="hi">6 (line &times; genotype)</td><td class="hi">3 (line)</td></tr>
      <tr><td>lines</td><td colspan="2"><i>Ash1l</i> / <i>Kdm6b</i> / <i>Kmt5b</i>, 1:1:1, sexes balanced within every line</td></tr>
      <tr><td>annotated pools</td><td class="hi">24 of 72</td><td class="lo">0 of 36</td></tr>
      <tr><td>annotated observations</td><td class="hi">144 of 432</td><td class="lo">0 of 216</td></tr>
      <tr><td>annotated frames</td><td>864k of 2.59 M</td><td>0 of 1.30 M</td></tr>
      <tr><td>annotators</td><td>6 &middot; 22 of 24 pools single-scored</td><td>&mdash;</td></tr>
      <tr><td>pools with no labels</td><td>48</td><td>36</td></tr>
      <tr><td>where genotype lives</td><td>between pools</td><td>within a pool</td></tr>
      <tr><td>estimator available</td><td>classical, PPI++, PPCI</td><td class="lo">PPCI only</td></tr>
    </tbody></table></div></div>
<div class="measure">
  <p><b>The two cohorts fail in opposite ways for the genotype question, and that is why this
  report is about the exposure.</b> In v1 the genotype contrast is between cages, and annotation is
  not exchangeable across it: 18 of the 24 annotated pools are het, and annotator is confounded
  with genotype (MF scored 18 het observations and no wt; CP scored 3 wt and no het). In v2 the
  cage is mixed, so genotype is a within-pool contrast &mdash; but attributing a behaviour to one
  animal needs per-mouse identity, and the shave marks that carry it are destroyed by the
  512&sup2; downsample. Neither blocks the exposure contrast, which is taken <em>within</em> a
  pool and therefore cancels cage, genotype, sex and annotator by construction.</p>
  <div class="note"><b>Estimand.</b> The unit of analysis is the <b>pool</b>, clustered. The mean <em>within-pool</em> change in behaviour across one
  phase transition, per exposure. Consecutive transitions only &mdash; H&rarr;O and O&rarr;P.
  P&minus;H is their sum, not an independent contrast. The two exposures are separate treatments
  with opposite signs on nose-to-tail and are never pooled. Genotype-specific effects are the next
  layer, not this one.</div>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">02 &middot; Outcome</p><h2>Count events, do not time them</h2></div>
  <p class="defn"><b>Bouts per minute.</b> A <em>bout</em> is one uninterrupted run of annotated
  frames; the outcome counts how often a behaviour begins, per minute of recording. The obvious
  alternative, <b>occupancy</b> (percentage of frames in the behaviour), answers a different
  question &mdash; how much time it takes up.</p>
  <figure><img src="{img['outcome']}" alt="Occupancy resolves 3 of 8 contrasts, bouts per minute resolves 7 of 8.">
    <figcaption>&ldquo;Resolved&rdquo; = 95% CI excludes zero, over 2 behaviours &times; 2
    exposures &times; 2 transitions. Bout duration resolves 1 of the same 8.</figcaption></figure>
  <p>Six independent reasons point the same way, and the first is the substantive one:</p>
  <div class="scroll"><table>
    <thead><tr><th>#</th><th>argument</th><th>evidence</th></tr></thead>
    <tbody>
      <tr><td>1</td><td>the effect <em>is</em> on initiation</td><td>counts resolve 7 of 8 contrasts, occupancy 3, mean bout duration 1</td></tr>
      <tr><td>2</td><td>durations are barely resolvable</td><td>49% of nn bouts and 22% of nt bouts last a single 5&nbsp;fps frame</td></tr>
      <tr><td>3</td><td>occupancy is heavy-tailed</td><td>the longest 10% of bouts carry 40&ndash;43% of all behaviour time</td></tr>
      <tr><td>4</td><td>counts are the steadier measurement</td><td>within-cell CV 0.68 vs 0.88 (nn), 0.96 vs 1.17 (nt)</td></tr>
      <tr><td>5</td><td>counts halve the treatment-linked model bias</td><td>predicted/true ratio varies 0.30&ndash;0.59 across phases on counts, 0.69&ndash;1.38 on occupancy</td></tr>
      <tr><td>6</td><td>the model predicts counts better</td><td>cross-fitted r&Delta; 0.72 vs 0.56 (class&nbsp;2), 0.48 vs 0.36 (nt)</td></tr>
    </tbody></table></div>
  <p>Arguments 5 and 6 are about the model rather than the biology, and they matter because the
  same outcome has to be estimated on pools with no labels. A bias that moves with the treatment
  is exactly what corrupts an average effect when no rectifier is available.</p>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">03 &middot; Effects</p>
  <h2>Every estimate, in one figure</h2></div>
  <p>Pick a cohort, an outcome unit, a behaviour and a breakdown. The figure draws each estimator
  against a shared axis, one panel per exposure, with both phase transitions. <b>Start on
  <i>classical</i>: human labels only, no model anywhere.</b> The two model-based estimators are
  what sections 04 and 05 exist to justify &mdash; <i>PPI++</i> adds the unlabelled pools with a
  rectifier that keeps it unbiased for any predictor, and <i>PPCI</i> is the plug-in that has no
  rectifier and is the only thing available on v2.</p>
</div>
  <div class="figwrap">{CHART}</div>
<div class="measure">
  <p><b>Two patterns are already legible in the classical numbers.</b> Nose-to-nose contact rises
  on exposure and falls when it is withdrawn &mdash; the sign flips between the two transitions,
  and both halves resolve under both exposures. And the exposures are genuinely different
  treatments: nose-to-nose rises on exposure under both (+0.65 fear, +0.47 social) while
  nose-to-tail runs in OPPOSITE directions (+0.35 fear, &minus;0.36 social). Averaging over
  exposures would cancel the nt effect outright, which is why they are never pooled.</p>
  <div class="note"><b>There are two behaviours here, not three.</b> The lab's
  <code>behavior_type</code> column carries three codes and an earlier version of this report
  read the third, <code>np</code>, as &ldquo;nose-to-anogenital&rdquo; &mdash; then built a
  three-behaviour dissociation on it. Cross-tabulating the code against the annotation files' own
  human-readable <code>Behavior</code> column over all 144 files shows what it really is:
  <code>nn</code> is <b>nose-to-nose, mutual</b> (<i>nose-nose_reciprocal</i>) and <code>np</code>
  is <b>nose-to-nose, directional</b> (<i>nose-nose_passive</i> &mdash; one animal sniffs, the
  other does not reciprocate). There is no anogenital behaviour anywhere in this dataset. The
  label is a DIRECTED pair, so a mutual bout is a 1 for both animals and a one-sided bout only for
  the active one &mdash; which means the model's <code>nn</code> head predicts the union, and nn
  and np can never be reported as separate effects.</div>
  <p><b>Switch the breakdown to the strata</b> &mdash; 6 line&nbsp;&times;&nbsp;genotype cells on
  v1, 3 lines on v2 &mdash; and the reason for everything that follows becomes visible.
  Annotation gave the wild-type strata <b>2 pools each</b> against the heterozygous strata's 6, so
  a classical stratified interval there has one degree of freedom and runs off the axis. Those are
  exactly the cells with 10 unlabelled pools apiece to borrow from.</p>
</div>
  <div class="figwrap">{DECAY}
    <p class="deccap">Pick a unit; both behaviours redraw. Hover any minute for its value and
    interval, or any bar for the phase mean it summarises.</p>
  </div>
<div class="measure">
  <h3 style="margin-top:26px">Nothing is stationary inside a phase</h3>
  <p>Rates fall several-fold across a recording. Fitting a Poisson decay per cell,
  <code>log E[N] = a + b&middot;t</code>, the half-life is <b>4&ndash;14 minutes</b> and
  <b>P decays fastest in every single cell</b> (&tau; &asymp; 6 min against 9&ndash;20 for H).
  One cell does not decay at all: <b>nose-to-tail under social exposure during O has a positive
  slope</b> &mdash; the exposure sustains investigation while everything else habituates.</p>
  <div class="note"><b>The window problem is confined to one contrast, and that is the useful
  part.</b> O and P are both 15 minutes, so any window rule applied to both leaves
  O&rarr;P <em>bit-for-bit unchanged</em> (verified: identical in all four cells). Only H&rarr;O
  is affected, because habituation runs 30 minutes and a mean over a decaying curve depends on how
  long you watch. Re-estimating H&rarr;O on a matched first-15-minute window:
  <div class="scroll" style="margin-top:11px"><table>
    <thead><tr><th>H &rarr; O</th><th>full window</th><th>matched 15 min</th><th>shift</th></tr></thead>
    <tbody>
      <tr><td>nt &middot; fear</td><td>+0.36</td><td>+0.22</td><td>&minus;0.14</td></tr>
      <tr><td>nt &middot; social</td><td>&minus;0.37</td><td class="hi">&minus;0.67</td><td>&minus;0.30</td></tr>
      <tr><td>nn &middot; fear</td><td>+0.66</td><td class="hi">+0.45</td><td>&minus;0.21</td></tr>
      <tr><td>nn &middot; social</td><td>+0.47</td><td class="lo">&minus;0.03</td><td class="lo">&minus;0.50 &mdash; sign flip</td></tr>
    </tbody></table></div></div>
  <p><b>So one headline number is an artefact.</b> nn&nbsp;&middot;&nbsp;social H&rarr;O is +0.47
  on the full window and &minus;0.03 on a matched one: the apparent rise is the H mean being
  dragged down by fifteen extra minutes of decay that O never gets. The other three keep their
  sign, and nt&nbsp;&middot;&nbsp;social gets <em>stronger</em> when matched.</p>

  <div class="sub">
    <p class="q">outcome design</p>
    <h3>Which window, and how to measure decay <span class="verdict v-part">two effects, not one</span></h3>
    <div class="note warnbox"><b>First, a confound that decides the window question: the
    phase-onset spike is not the treatment.</b> Every phase is a separate recording, and the
    experimenter opens the cage to start it. <b>P is the phase where the odour is <em>removed</em>
    &mdash; and P has the LARGEST onset spike of the three in 3 of 4 cells</b> (first-2-minutes
    over last-2-minutes rate: nn&nbsp;&middot;&nbsp;fear H 7.6, O 6.7, <b>P 12.3</b>).
    A response that is strongest when the odour is taken away cannot be a response to the odour.
    It is handling.</div>
    <p>That settles what the two candidate windows for the 30-minute habituation phase actually
    measure. Comparing the <b>last</b> 15 minutes of H against O uses the state the animals were
    genuinely in when the odour arrived &mdash; the better <em>baseline</em> &mdash; but it puts a
    decayed tail on one side and a fresh onset on the other, so it charges the handling spike to
    the odour. Comparing the <b>first</b> 15 minutes matches onset position, so the handling
    response appears on both sides and cancels, at the cost of contrasting cage-novelty with
    odour-novelty. Neither is clean, and they disagree by a lot:</p>
    <div class="scroll"><table>
      <thead><tr><th>H &rarr; O</th><th>full H</th><th>first 15</th><th>last 15</th><th>spread</th></tr></thead>
      <tbody>
        <tr><td>nt &middot; fear</td><td>+0.36</td><td>+0.22</td><td>+0.49</td><td>0.28</td></tr>
        <tr><td>nt &middot; social</td><td>&minus;0.37</td><td>&minus;0.67</td><td>&minus;0.07</td><td>0.60</td></tr>
        <tr><td>nn &middot; fear</td><td class="hi">+0.66</td><td class="hi">+0.45</td><td class="hi">+0.86</td><td>0.42</td></tr>
        <tr><td>nn &middot; social</td><td>+0.47</td><td>&minus;0.03</td><td>+0.97</td><td class="lo">1.01 &mdash; spans zero</td></tr>
      </tbody></table></div>
    <p><b>nose-to-nose under social exposure must not be a headline under any window.</b> It runs
    from &minus;0.03 to +0.97 across three defensible choices &mdash; a full bout per minute, and
    it changes sign. <b>nose-to-nose under fear is the one H&rarr;O effect that survives
    everything</b>, positive and resolving in all three. And all four O&rarr;P contrasts are
    window-invariant by construction, so those are the ones to trust.</p>
    <h3 style="margin-top:22px">Measuring the decay: not a slope, and not a time constant</h3>
    <p>Two candidates fail before they start. A <b>Poisson slope</b> assumes log-linear decay,
    which the curvature test rejects in 7 of 12 cells. A <b>time constant &tau;</b> inherits that
    misspecification and blows up wherever the slope approaches zero &mdash; nose-to-tail under
    social exposure during O actually <em>rises</em>, giving &tau;&nbsp;=&nbsp;&minus;27 min. A
    <b>time-to-asymptote</b> needs the floor, which needs a three-parameter fit that 0&ndash;3
    counts per minute will not support per pool.</p>
    <div class="note"><b>What works is a front-loading fraction.</b>
    F = bouts in the first 5 minutes / bouts in the first 15. Flat process &rarr; 0.33; strong
    decay &rarr; higher. Bounded, model-free, length-invariant, defined per observation so it
    drops straight into the same pool-level contrast, and it needs no exponential. It resolves
    <b>5 of 8</b> contrasts &mdash; and it says something the level cannot:
    <div class="scroll" style="margin-top:11px"><table>
      <thead><tr><th>&Delta;F</th><th>nt &middot; fear</th><th>nt &middot; social</th><th>nn &middot; fear</th><th>nn &middot; social</th></tr></thead>
      <tbody>
        <tr><td>H &rarr; O &nbsp;(odour ON)</td><td class="hi">&minus;0.40*</td><td>&minus;0.12</td><td class="hi">&minus;0.19*</td><td class="hi">&minus;0.11*</td></tr>
        <tr><td>O &rarr; P &nbsp;(odour OFF)</td><td>+0.19</td><td class="hi">+0.31*</td><td>+0.04</td><td class="hi">+0.27*</td></tr>
      </tbody></table></div>
    <b>Every sign is negative turning the odour on and positive turning it off.</b> The exposure
    <em>flattens</em> the habituation curve, and withdrawing it restores fast habituation. That is
    a second, separable effect: not &ldquo;how much behaviour the odour triggers&rdquo; but
    &ldquo;how long it holds attention&rdquo;. Its cost is that F is undefined for an observation
    with no bouts in the window, so n falls to 17&ndash;24 depending on the cell.</div>
    <p><b>Recommendation.</b> Report <b>both</b>, per transition, as the two effects they are:
    the <b>level</b> on a matched window &mdash; first-15 if the question is the odour net of
    handling, last-15 if it is the odour against a settled baseline, stated explicitly either way
    &mdash; and <b>&Delta;F</b> as the decay effect. Do not replace the level with an extrapolated
    amplitude: measured above, that trades a small bias for a large variance and resolves 3 of 8.</p>
  </div>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">04 &middot; Model</p>
  <h2>A frame classifier for the 84 unlabelled pools</h2></div>
  <p>Only 24 of 108 pools are annotated, and none of v2 is. Everything in this section exists to
  put a number on the other 84 &mdash; a per-frame detector whose per-observation aggregate can
  stand in for a human score.</p>
  <div class="flow">
    <div class="step"><b>Video</b><span>2064&sup2; @ 30 fps<br>&rarr; stored 512&sup2; @ 5 fps</span></div>
    <div class="step"><b>Encoder</b><span>DINOv2-base ViT-B/14<br>448 px &rarr; 1024 tokens/frame</span></div>
    <div class="step"><b>Spatial pool</b><span>1 learned query over<br>1024 patch tokens</span></div>
    <div class="step"><b>Temporal</b><span>attention over 5 frames<br>(&plusmn;0.4 s)</span></div>
    <div class="step"><b>Output</b><span>2 sigmoids: nt, nn<br>multi-label</span></div>
  </div>
  <div class="scroll"><table>
    <thead><tr><th>setting</th><th>value</th><th>setting</th><th>value</th></tr></thead>
    <tbody>
      <tr><td>optimiser</td><td>AdamW</td><td>head params</td><td>5.03 M</td></tr>
      <tr><td>lr / weight decay</td><td>3e-4 / 0.05</td><td>batch</td><td>64</td></tr>
      <tr><td>schedule</td><td>3 warmup, cosine</td><td>negatives</td><td>1 per positive</td></tr>
      <tr><td>dropout</td><td>0.4</td><td>train / val</td><td>20 / 4 pools</td></tr>
      <tr><td>augmentation</td><td colspan="3">D4 dihedral (exact &mdash; the cage is filmed top-down) + brightness / contrast / gamma</td></tr>
    </tbody></table></div>

  <h3 style="margin-top:28px">What actually moves the number</h3>
  <p>Every lever tried, largest first, against a seed-noise floor of <b>0.0089</b> macro AP
  measured from two identical runs. Read this table first; the subsections below are the evidence
  for each row, in the same order.</p>
  <div class="scroll"><table>
    <thead><tr><th>#</th><th>lever</th><th>what varies</th><th>&Delta; macro AP</th><th>verdict</th></tr></thead>
    <tbody>
      <tr><td>1</td><td>annotate more pools</td><td>data</td><td class="hi">+0.076 per doubling</td><td class="hi">the binding constraint, no plateau</td></tr>
      <tr><td>2</td><td>adapt the encoder (BitFit)</td><td>encoder</td><td class="hi">+0.112</td><td class="hi">recalibration, at 1/577th the params</td></tr>
      <tr><td>3</td><td>224 &rarr; 448 px input</td><td>input</td><td class="hi">+0.132</td><td>real, but now saturated</td></tr>
      <tr><td>4</td><td>SSL on unlabelled frames</td><td>encoder, label-free</td><td class="hi">+0.033</td><td class="hi">what we deploy; also reaches v2</td></tr>
      <tr><td>5</td><td>vREx</td><td>objective</td><td>&plusmn;0.008</td><td class="lo">inside seed noise</td></tr>
      <tr><td>6</td><td>DERM</td><td>objective</td><td>&mdash;</td><td class="v-part">never run until now</td></tr>
      <tr><td>&mdash;</td><td>head capacity (self-attn, multi-query)</td><td>head</td><td>&minus;0.003 to +0.014</td><td class="lo">flat</td></tr>
    </tbody></table></div>
  <div class="note warnbox"><b>One structural limit, before any of it.</b> The regime overfits
  &mdash; training loss falls monotonically while validation AP plateaus near epoch 24, so longer
  schedules and extra head capacity do nothing. That is row 3 and the last row of the table in one
  sentence, and it is why rows 1 and 2 are where the leverage is.
  <br><br>And <b>AP is not the quantity that decides any of this</b>. What PPI's variance reduction
  depends on is r&Delta;, the correlation between true and predicted <em>within-pool phase
  differences</em>. The AP leader and the r&Delta; leader are different arms &mdash; see section 05.
  Rows are ordered by AP here because that is what every arm reports; it is not the ranking key.</div>
</div>

<div class="measure">
  <div class="sub">
    <p class="q">ablation &middot; data</p>
    <h3>The scaling law of annotation <span class="verdict v-yes">the binding constraint</span></h3>
    <p>Nested subsets of the labelled pools, so each point differs from the last for exactly one
    reason.</p>
  </div>
  <figure><img src="{img['lcurve']}" alt="Macro AP against number of annotated pools, still rising at 20.">
    <figcaption>Log-linear fit, R&sup2; = 0.993, no plateau.</figcaption></figure>
  <div class="scroll"><table>
    <thead><tr><th>annotated pools</th><th>observations</th><th>macro AP</th><th>gain</th></tr></thead>
    <tbody>
      <tr><td>5</td><td>30</td><td>0.2716</td><td>&mdash;</td></tr>
      <tr><td>10</td><td>60</td><td>0.3461</td><td>+0.0745</td></tr>
      <tr><td>15</td><td>90</td><td>0.3837</td><td>+0.0376</td></tr>
      <tr><td>20 (all we have for training)</td><td>120</td><td>0.4289</td><td>+0.0453</td></tr>
      <tr><td>40 &mdash; extrapolated</td><td>240</td><td>~0.499</td><td>+0.070</td></tr>
      <tr><td>72 &mdash; extrapolated</td><td>432</td><td>~0.564</td><td>+0.135</td></tr>
    </tbody></table></div>
  <div class="note"><b>Each doubling of annotated pools is worth ~0.076 macro AP</b> and the curve
  has not bent. For scale, every modelling intervention below is worth between &minus;0.09 and
  +0.11, and the best label-free one is +0.033. <b>Twenty more annotated pools would beat all of
  them combined.</b></div>

  <div class="sub">
    <p class="q">ablation &middot; encoder</p>
    <h3>Supervised fine-tuning <span class="verdict v-yes">yes &mdash; +0.11 AP</span></h3>
    <p>Unfreezing the last DINOv2 blocks is the single largest modelling gain measured: macro AP
    0.4289 frozen &rarr; 0.4889 at two blocks &rarr; 0.5243 at six. The seed-noise yardstick is
    0.0089 (two identical runs, seeds 42 and 1), so these are real.</p>
    <p><b>What it is doing is recalibration, not new computation.</b> BitFit &mdash; training only
    biases, LayerNorm gains and LayerScale gains, and nothing that can form a new function of two
    patch features &mdash; matches and then beats full fine-tuning: <b>0.5409 with 70,656 trainable
    encoder parameters against 0.5243 with 42.5 M</b>, a 602&times; cut. Two head-capacity arms
    (patch self-attention, multi-query pooling) were flat. One trap is worth recording: BitFit
    reads 0.4509 at the inherited encoder LR of 1e-5 and 0.4902 at 1e-3, so running the single
    inherited LR would have inverted the conclusion.</p>
  </div>

  <div class="sub">
    <p class="q">ablation &middot; unlabelled frames</p>
    <h3>Self-supervised adaptation <span class="verdict v-yes">yes &mdash; and it is what we deploy</span></h3>
    <p>data2vec-style masked patch-feature regression against an EMA teacher, on 374,400 unlabelled
    frames spanning v1 and v2 &mdash; including 216 v2 observations no supervised arm can reach.
    Stage&nbsp;B then <b>freezes that encoder and trains the head only</b>, so the comparison is
    against the stock-DINOv2 frozen control.</p>
    <div class="scroll"><table>
      <thead><tr><th>arm</th><th>encoder</th><th>macro AP</th><th>read</th></tr></thead>
      <tbody>
        <tr><td>frozen control</td><td>stock DINOv2</td><td>0.4289</td><td>reference</td></tr>
        <tr><td>SSL, 2 blocks, stride 10</td><td>adapted</td><td class="hi">0.4622</td><td class="hi">+0.033 with zero labels</td></tr>
        <tr><td>&mdash; 2&times; the frames, matched compute</td><td>adapted</td><td>0.4524</td><td>neutral</td></tr>
        <tr><td>&mdash; 6 blocks instead of 2</td><td>adapted</td><td class="lo">0.3831</td><td class="lo">harmful</td></tr>
        <tr><td>BitFit 6 blocks</td><td>stock DINOv2</td><td class="hi">0.5409</td><td>best model overall</td></tr>
        <tr><td>BitFit 6 blocks <b>on the SSL encoder</b></td><td>adapted</td><td class="lo">0.5127</td><td class="lo">&minus;0.028 against BitFit alone</td></tr>
      </tbody></table></div>
    <p><b>SSL works, and an earlier draft of this report undersold it.</b> +0.033 macro AP is 3.7&times;
    the 0.0089 seed spread, it is bought with <em>zero labels</em>, and it is the only intervention
    here that also reaches v2 &mdash; 216 observations no supervised arm can touch. It is also
    <b>the encoder every number in section 05 rests on</b>: the three cross-fitting folds were run
    on it. Calling it &ldquo;closed&rdquo; was wrong.</p>
    <p>On the quantity that actually decides whether a model helps the estimate, it is the best
    thing measured. r&Delta; is the correlation between true and predicted <em>within-pool phase
    differences</em>, and PPI's variance reduction is a function of it and nothing else:</p>
    <div class="scroll"><table>
      <thead><tr><th>arm</th><th>macro AP</th><th>r&Delta; nt</th><th>r&Delta; nn</th></tr></thead>
      <tbody>
        <tr><td>frozen control (seed 1)</td><td>0.4200</td><td>0.417</td><td>0.768</td></tr>
        <tr><td>SSL-adapted, frozen</td><td>0.4622</td><td>0.455</td><td class="hi">0.902 &mdash; highest of any run</td></tr>
        <tr><td>BitFit 6 blocks</td><td class="hi">0.5409</td><td class="hi">0.669</td><td>0.872</td></tr>
        <tr><td>BitFit 6 blocks on the SSL encoder</td><td>0.5127</td><td>0.626</td><td>0.760</td></tr>
      </tbody></table></div>
    <p>The AP leader and the r&Delta; leader are <b>different arms</b>, which is the whole reason
    section 05 shortlists on AP and chooses on the causal quantity.</p>
    <div class="note warnbox"><b>&ldquo;It does not stack&rdquo; is one seed, and should not be
    quoted as settled.</b> The claim rests on a single comparison &mdash; BitFit-6 on the SSL
    encoder at 0.5127 against BitFit-6 alone at 0.5409. That gap is 0.028, about 3&times; a seed
    spread estimated from a <em>different</em> configuration, from one draw each. It is suggestive
    that both interventions buy the same domain recalibration and therefore substitute rather than
    compose, and the r&Delta; column above is consistent with that. It is not established. Two
    seeds per arm would settle it for ~3 GPU-hours.<br><br>
    What <em>is</em> established, because each was a clean single-variable change: scaling the
    corpus 2&times; at matched compute is neutral (0.4524), and adapting six blocks instead of two
    is clearly harmful (0.3831). The pretrained features are easy to damage. Do not spend more
    runs on <em>scaling</em> SSL; that is a narrower claim than closing SSL.</div>
  </div>

  <div class="sub">
    <p class="q">ablation &middot; input</p>
    <h3>Resolution against tokens <span class="verdict v-part">token-bound, now saturated</span></h3>
    <p>Raising the input from 224 px to 448 px changes two things at once &mdash; the number of
    patch tokens and the pixel detail inside each. Capping the pixels while keeping the tokens
    separates them.</p>
  </div>
</div>
  <div class="figwrap"><figure>
    <img src="{img['tokpix']}" alt="Waterfall decomposing the 224 to 448 improvement into a token step and a pixel step.">
    <figcaption>Quadrupling tokens at fixed pixel detail is worth +0.103 macro AP; restoring full
    pixels at fixed tokens adds +0.029; going further to 1,296 tokens adds +0.003.</figcaption>
  </figure></div>
<div class="measure">
  <div class="scroll"><table>
    <thead><tr><th>option</th><th>tokens/frame</th><th>patches per mouse</th><th>verdict</th></tr></thead>
    <tbody>
      <tr><td>today: 448 px whole frame</td><td>1,024</td><td>2.2</td><td>at the ceiling</td></tr>
      <tr><td>504 px whole frame</td><td>1,296</td><td>2.5</td><td class="lo">+0.003, saturated</td></tr>
      <tr><td>full 2064 px whole frame</td><td>21,609</td><td>10</td><td class="lo">quadratic cost; 466 GB</td></tr>
      <tr><td>224 px crop around a pair</td><td>256</td><td>10</td><td class="hi">4.5&times; the detail at &frac14; the cost</td></tr>
    </tbody></table></div>
  <p>Whole-frame resolution is spent. Cropping is the lever that is left, and it is the same fix
  the identity problem needs. Separately, the pipeline resamples twice (2064&rarr;512&rarr;448)
  and so retains only <b>76% of the fine detail</b> of a direct 2064&rarr;448 &mdash; recoverable
  for free by extracting frames at the working size.</p>

  <div class="sub">
    <p class="q">ablation &middot; invariance</p>
    <h3>vREx <span class="verdict v-no">no measurable help</span></h3>
    <p><b>Where it should have been needed.</b> The model's bias is treatment-dependent: the ratio
    of predicted to true rate varies across phases, and on the best-AP arm it varied enough to
    predict O&nbsp;&lt;&nbsp;H &mdash; the wrong sign on the primary effect. Within v1 that costs
    nothing, because PPI's rectifier corrects any predictor; on v2, where no labels exist and no
    rectifier can be built, it is exactly the failure mode invariant training targets.</p>
    <div class="scroll"><table>
      <thead><tr><th>environment definition</th><th>&beta;</th><th>macro AP</th><th>r&Delta; nt</th><th>r&Delta; nn</th></tr></thead>
      <tbody>
        <tr><td>none (control, seed 1)</td><td>&mdash;</td><td>0.4200</td><td>0.417</td><td>0.768</td></tr>
        <tr><td>the 6 phase &times; exposure cells</td><td>1</td><td>0.4366</td><td class="lo">0.300</td><td class="hi">0.861</td></tr>
        <tr><td>the 6 phase &times; exposure cells</td><td>10</td><td>0.4265</td><td class="lo">0.205</td><td>0.770</td></tr>
        <tr><td>annotator</td><td>10</td><td>0.4272</td><td>0.542</td><td class="hi">0.876</td></tr>
        <tr><td>annotator</td><td>100</td><td class="lo">0.3352</td><td>0.736</td><td>0.702</td></tr>
      </tbody></table></div>
    <p>Every AP is inside seed noise and none of them flattened the per-phase bias. The r&Delta;
    columns are new here and they do not rescue it either: the moves go in both directions, and
    they are measured on the standing 4-pool split, which supplies <b>16 points</b> (4 pools
    &times; 2 exposures &times; 2 transitions) at an operating threshold picked by max-F1 on that
    same split. That is a screen, not a result. Stop spending runs on vREx.</p>
  </div>

  <div class="sub">
    <p class="q">ablation &middot; deconfounding</p>
    <h3>DERM <span class="verdict v-part">never actually run &mdash; now running</span></h3>
    <div class="note warnbox"><b>A correction to the previous version of this report.</b> This
    section was headed &ldquo;DERM / vREx &mdash; no measurable help&rdquo; and the summary listed
    DERM as closed. That was a misattribution: <code>train_online_aug.py</code> only ever
    implemented <b>vREx</b>, the four runs behind the claim are named <code>vrexCond_b1/b10</code>
    and <code>vrexAnn_b10/b100</code>, and <b>DERM had never been run on this dataset at all</b>.
    The two methods attack the same failure by opposite means and there is no reason a null for one
    transfers to the other.</div>
    <div class="scroll"><table>
      <thead><tr><th></th><th>what it changes</th><th>mechanism</th></tr></thead>
      <tbody>
        <tr><td><b>vREx</b></td><td>the objective</td><td>keeps the training distribution, ADDS a penalty on the variance of risk across environments</td></tr>
        <tr><td><b>DERM</b></td><td>the distribution</td><td>keeps the loss, REWEIGHTS every sample by Var(<i>Y</i>|<i>E</i>) / P(<i>Y</i>,<i>E</i>)</td></tr>
      </tbody></table></div>
    <p>For a binary label DERM's weight collapses to something you can read off. With
    <i>p<sub>e</sub></i> = P(<i>Y</i>=1|<i>E</i>=<i>e</i>) and P(<i>e</i>) the environment's share
    of the training pool:</p>
    <div class="note"><code>w(y=1, e) = (1 &minus; p<sub>e</sub>) / P(e)</code> &nbsp;&nbsp;and&nbsp;&nbsp;
    <code>w(y=0, e) = p<sub>e</sub> / P(e)</code><br><br>
    so positives and negatives end up carrying <b>equal total mass inside every environment</b>,
    and each environment's contribution becomes proportional to its own outcome variance. Verified
    numerically against the general formula: raw prevalence spread 3.5&times; across environments
    &rarr; DERM-weighted prevalence exactly 0.5 in every one, mean weight normalised to 1 so the
    effective learning rate is unchanged and the comparison against ERM is not confounded by a
    quietly different step size.</div>
    <p><b>Why this is the mechanism the problem calls for.</b> Phase predicts <em>prevalence</em>
    here &mdash; the odour port visibly changes the scene, and the predicted/true rate ratio moves
    1.80&ndash;3.51 across phases. A classifier can therefore score a frame by which phase it looks
    like rather than by what the mice are doing, and a bias that moves with the treatment is
    exactly what corrupts an ATE estimated without a rectifier. DERM removes that route by
    construction, and unlike a penalty on predictions it never asks the model to be invariant to
    phase &mdash; it only breaks the label&ndash;environment association, leaving the biology
    intact.</p>
    <p><b>Four arms are running</b>, all at the config the controls and the vREx arms used, so the
    method is the only thing that varies: environments = the 3 <b>phases</b> (the estimand's own
    treatment variable), environments = the 6 <b>phase &times; exposure</b> cells, a <b>seed
    replicate</b> of the phase arm because every vREx arm was a single draw, and environments =
    <b>annotator</b> as a contrast rather than a candidate. That last one matters for
    interpretation: annotator is exactly balanced across phase &mdash; every scorer took all six
    observations of each pool they touched, so H/O/P = 48/48/48 &mdash; which means annotator
    <em>already cancels</em> in a within-pool phase contrast. If deconfounding against annotator
    moves r&Delta; as much as deconfounding against phase does, the mechanism is not what the
    argument above says it is, and the phase result would need a different explanation.</p>
    <div class="note warnbox"><b>These will be screened on r&Delta;, not AP, and a screen is not a
    CI.</b> The vREx round was judged on macro AP alone &mdash; and AP is not what PPI's variance
    reduction depends on. Whichever arm clears the controls will then be <b>cross-fitted over three
    folds</b>, because only out-of-fold predictions on all 24 annotated pools license a real
    shrinkage number. Until that runs, no CI shrinkage is quoted for DERM.</div>
  </div>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">05 &middot; Results</p>
  <h2>The model we run, and what it buys</h2></div>
  <p><b>Shortlist on frame AP; choose on the causal quantity.</b> The two agree only loosely, and
  the disagreement is not academic &mdash; the best-AP arm of an earlier ablation was among the
  worst for the estimate.</p>
</div>
  <div class="figwrap"><figure>
    <img src="{img['apcausal']}" alt="Scatter of macro AP against causal correlation across runs.">
    <figcaption>&rho; &asymp; +0.5 across runs. r&Delta; is the correlation between true and
    predicted <em>within-pool phase differences</em> &mdash; the quantity PPI's variance reduction
    actually depends on.</figcaption>
  </figure></div>
<div class="measure">
  <div class="scroll"><table>
    <thead><tr><th>candidate</th><th>macro AP</th><th>event F1 nt / nn</th><th>r&Delta; nt</th><th>r&Delta; nn</th></tr></thead>
    <tbody>
      <tr><td>BitFit, 6 blocks</td><td class="hi">0.5409</td><td>0.473 / 0.553</td><td>0.669</td><td>0.872</td></tr>
      <tr><td>full FT, 2 blocks (seed 1)</td><td>0.4927</td><td>0.428 / 0.537</td><td class="hi">0.824</td><td class="hi">0.875</td></tr>
      <tr><td>SSL-adapted, frozen</td><td>0.4622</td><td>0.432 / 0.489</td><td>0.455</td><td class="hi">0.902</td></tr>
      <tr><td>region-preserving head, 0.44 M</td><td>0.4328</td><td>0.394 / 0.483</td><td>0.675</td><td>0.781</td></tr>
      <tr><td><b>cross-fitted deployment</b> (3 folds)</td><td>0.382</td><td>0.384 / 0.482</td><td>0.573</td><td>0.798</td></tr>
    </tbody></table></div>
  <p class="defn"><b>Read the last row differently from the others.</b> The first four are scored
  on the standing 4-pool validation split &mdash; 24 observations, far too few to separate close
  models, which is why the r&Delta; column cannot be used to rank them. The last row is the average
  over three folds that between them hold out <em>all</em> 24 annotated pools, 8 at a time, so
  every labelled observation is scored by a model that never saw its pool. That is the only
  condition under which PPI's rectifier is valid, and it is a harder split than the standing one
  &mdash; hence the lower AP.</p>
  <div class="note warnbox"><b>The deployed configuration is not the best one we found.</b>
  Cross-fitting was run on the SSL-adapted frozen encoder with a plain 5.03 M head and a 20-epoch
  schedule, not on BitFit-6 with the 0.52 M cross-attention head that leads on every accuracy axis.
  The choice bought label-free adaptation that also covers v2, and it is what all the numbers below
  rest on &mdash; but re-running the three folds on the selected configuration is the cheapest
  outstanding improvement to the estimate, at roughly 18 GPU-hours.</div>

  <h3 style="margin-top:26px">Where it is right and wrong</h3>
  <p>Most confident case per bucket, one frame per observation, on held-out pools.</p>
</div>
  <div class="figwrap">
  <figure><img src="{img['conf_nt']}" alt="True positive, false positive, false negative and true negative examples for nose-to-tail.">
    <figcaption><b>nt.</b> Confident false positives are mostly frames adjacent to a scored bout
    &mdash; the model holds p&nbsp;&asymp;&nbsp;1 through a gap the annotator left.</figcaption></figure>
  <figure><img src="{img['conf_nn']}" alt="Examples for the nose-to-nose class.">
    <figcaption><b>nn</b> (mutual and directional together), same run and operating point.</figcaption></figure>
  </div>
<div class="measure">
  <h3 style="margin-top:26px">Detections where no annotator ever looked</h3>
  <p>No ground truth exists for any of the 84 unlabelled pools, so this is a sanity check and a
  qualitative exhibit, never a performance claim. Two things are checkable: whether predicted rates
  stay physically plausible, and whether the confident detections actually show the behaviour.</p>
</div>
  <div class="figwrap">
  <figure><img src="{img['un_nn_v1']}" alt="Confident nose-to-nose detections on unannotated v1 pools.">
    <figcaption><b>Unannotated v1 &middot; nn.</b> The 48 pools PPI rectifies.</figcaption></figure>
  <figure><img src="{img['un_nt_v1']}" alt="Confident nose-to-tail detections on unannotated v1 pools.">
    <figcaption><b>Unannotated v1 &middot; nt.</b></figcaption></figure>
  <figure><img src="{img['un_nn_v2']}" alt="Confident nose-to-nose detections on v2.">
    <figcaption><b>v2 &middot; nn.</b> A different cohort recorded months later, with zero
    annotations anywhere.</figcaption></figure>
  <figure><img src="{img['un_nt_v2']}" alt="Confident nose-to-tail detections on v2.">
    <figcaption><b>v2 &middot; nt.</b></figcaption></figure>
  </div>
<div class="measure">
  <div class="scroll"><table>
    <thead><tr><th>set</th><th>pools</th><th>observations</th><th>predicted nt</th><th>predicted nn</th><th>read</th></tr></thead>
    <tbody>
      <tr><td>v1 labelled (out-of-fold)</td><td>24</td><td>144</td><td>2.9&ndash;7.2%</td><td>8.3&ndash;13.1%</td><td>true 0.5&ndash;1.4% / 1.2&ndash;2.9%</td></tr>
      <tr><td>v1 unlabelled</td><td>48</td><td>288</td><td>3.9&ndash;10.1%</td><td>5.5&ndash;8.5%</td><td>plausible</td></tr>
      <tr><td>v2 (target cohort)</td><td>36</td><td>216</td><td>5.8&ndash;11.4%</td><td>8.1&ndash;12.1%</td><td>plausible, shifted up</td></tr>
    </tbody></table></div>
  <p>Ranges are over the six phase &times; exposure cells. Nothing collapses or saturates, and the
  v2 detections show genuine contact. Predicted occupancy runs about <b>5&times; above truth</b>
  throughout &mdash; part calibration offset, part the deliberate prior shift in training &mdash; so these must never be
  read as behaviour rates directly. It costs nothing downstream: PPI's &lambda; absorbs the scale.</p>
</div>

<div class="measure">
  <h3 style="margin-top:26px">The three estimators, and what separates them</h3>
  <p class="defn"><b>They are not the same estimator.</b> Write
  <span style="white-space:nowrap">Y&#772;</span> for the classical mean over the {n_lab} labelled
  pools and <span style="white-space:nowrap">f&#772;<sub>n</sub></span>,
  <span style="white-space:nowrap">f&#772;<sub>N</sub></span>,
  <span style="white-space:nowrap">f&#772;<sub>all</sub></span> for the mean predicted difference
  on the labelled, unlabelled and all pools.</p>
  <div class="scroll"><table>
    <thead><tr><th>estimator</th><th>formula</th><th>needs labels?</th><th>what it assumes</th></tr></thead>
    <tbody>
      <tr><td>classical</td><td>Y&#772;</td><td>yes &mdash; it <em>is</em> the labels</td><td>nothing beyond the design</td></tr>
      <tr><td><b>PPI++</b></td><td>Y&#772; + &lambda;&thinsp;(f&#772;<sub>N</sub> &minus; f&#772;<sub>n</sub>)</td><td class="hi">yes, for the rectifier</td><td class="hi">nothing &mdash; unbiased for ANY predictor</td></tr>
      <tr><td><b>PPCI</b></td><td>k&thinsp;&middot;&thinsp;f&#772;<sub>all</sub>,&nbsp;&nbsp;k = Y&#772; / f&#772;<sub>n</sub></td><td>yes, for k only</td><td class="lo">the model is off by a constant FACTOR</td></tr>
      <tr><td>PPCI, uncalibrated</td><td>f&#772;<sub>all</sub></td><td class="hi">no &mdash; none at all</td><td>nothing, but reports the MODEL's scale</td></tr>
    </tbody></table></div>
  <div class="note warnbox"><b>A correction: PPCI used to rescale by the regression slope, and
  that was wrong.</b> The slope
  &beta;&nbsp;=&nbsp;Cov(D<sub>Y</sub>,D<sub>f</sub>)/Var(D<sub>f</sub>) is the right multiplier
  for predicting <em>one</em> pool's outcome from its own prediction, but it is the wrong
  multiplier for rescaling a <em>mean</em>: &beta;&nbsp;=&nbsp;&rho;&thinsp;&sigma;<sub>Y</sub>/&sigma;<sub>f</sub>
  is attenuated by noise in the predictor. Measured here, &beta; ran 0.10&ndash;0.46 while the
  ratio of means ran 0.44&ndash;2.18 &mdash; a factor of <b>seven</b> apart on
  nose-to-nose&nbsp;&middot;&nbsp;social, where &beta;&nbsp;=&nbsp;0.21 against a true ratio of
  1.56. Every &beta;-rescaled estimate was pulled toward zero by regression attenuation: +0.04
  where the classical mean is +0.47. Replaced by
  k&nbsp;=&nbsp;Y&#772;/f&#772;<sub>n</sub>, the ratio of means, which is the calibration of the
  quantity actually being reported. PPCI now agrees with classical where it is identified
  (+0.66 against +0.65 on nn&nbsp;&middot;&nbsp;fear) instead of undershooting it.</div>
  <p><b>k is a ratio, so it is refused near a zero denominator.</b> In three of the eight cells
  the mean predicted difference is not separated from zero, k is not identified, and PPCI is
  reported as undefined rather than as a large number. That is the honest failure mode of any
  scale calibration and it is worth seeing rather than smoothing over.</p>
  <div class="note"><b>&ldquo;Is calibrating cheating?&rdquo; &mdash; the answer differs by cohort.</b>
  <br><br><b>On v1: no, but it is not label-free either.</b> k is fitted on the annotated pools'
  out-of-fold predictions and the bootstrap refits it, so there is no leakage. But PPCI is not
  &ldquo;the model alone&rdquo; &mdash; it borrows one number from the labels. If you have those
  labels, PPI++ uses them better: it is unbiased for <em>any</em> predictor, where PPCI needs the
  model to be wrong only by a constant factor.
  <br><br><b>On v2: you cannot fit k, and that is the whole problem.</b> Nothing in v2 is
  annotated. The v2 column transports v1's k, and that transport &mdash; not the model, not the
  inference &mdash; is the load-bearing assumption. The <em>uncalibrated</em> row in the table
  view is the only estimate on this page that needs no labels anywhere; it is on the model's
  scale, so it supports claims about <b>sign and relative pattern only</b>. Read the v2 agreement
  that way: 7 of 8 signs reproduce v1, which is evidence the model transfers, and it is not
  evidence that any v2 magnitude is right.</div>
  <div class="note warnbox"><b>Classical and PPI++ do not target quite the same population.</b>
  Annotation is <b>3:1 het-enriched</b>: the 24 labelled pools are 18 het / 6 wt against a 36/36
  design. Classical therefore estimates the effect <em>in the annotated pools</em>, while PPI++
  pulls in 48 unlabelled pools that are wt-enriched (30 wt / 18 het) and so targets the full
  72-pool population. Those coincide only if the phase effect does not vary with genotype, and
  the stratified view suggests it varies a little &mdash; nose-to-nose under fear reads about
  +0.79 across the wt strata against +0.60 across the het strata, which moves the population
  target roughly +0.05 (about 7%) away from the labelled one. Small next to the intervals here,
  but it is a difference in <em>estimand</em>, not in precision, so it does not shrink with more
  data. The clean fix is to estimate within stratum and recombine with design weights &mdash;
  which is what the stratified view is for.</div>
  <div class="note warnbox"><b>Read the shrinkage honestly.</b> A narrower interval is only worth
  having if it still covers the truth. PPI++'s does, by construction, for any predictor.
  <b>PPCI's is not guaranteed</b>: dropping the intercept buys its narrowing partly with bias, and
  on v2 it additionally assumes a slope fitted on v1 transfers to a cohort recorded months later.
  Every interval in the figure uses the same <i>t</i><sub>n&minus;1</sub> quantile on the same
  pool-clustered scale, so the percentages are comparable across methods; an earlier version mixed
  <i>t</i> for classical with 1.96 for PPI and attributed the 5% difference to PPI.</div>
</div></section>


<section><div class="measure">
  <div class="sechead"><p class="eyebrow">06 &middot; Next</p><h2>What to do next</h2></div>
  <div class="scroll"><table>
    <thead><tr><th></th><th>action</th><th>why now</th></tr></thead>
    <tbody>
      <tr><td>1</td><td>annotate ~20 more v1 pools</td><td>+0.076 AP per doubling and no plateau &mdash; worth more than every modelling change combined</td></tr>
      <tr><td>2</td><td>annotate 4&ndash;6 v2 pools</td><td>converts the v2 extrapolation into a real PPI estimate with a rectifier</td></tr>
      <tr><td>3</td><td>fix the observation window on biological grounds</td><td>largest single lever on the headline number; currently inherited, not chosen</td></tr>
      <tr><td>5</td><td>re-run the three cross-fitting folds on BitFit-6</td><td>~18 GPU-h to put the estimate on the configuration that actually won</td></tr>
      <tr><td>6</td><td>per-animal crops from the 2064 px source</td><td>only resolution lever left, and the same fix the genotype question needs in v2</td></tr>
      <tr><td>7</td><td>amplitude + decay constant per phase</td><td>replaces a mean whose value depends on how long the recording ran</td></tr>
      <tr><td>8</td><td>double-annotate 15&ndash;20 observations</td><td>the only clean way to bound irreducible label noise, which caps everything above</td></tr>
    </tbody></table></div>
  <div class="note warnbox"><b>Closed, deliberately &mdash; and narrower than it used to read.</b>
  Two things are closed: <em>scaling</em> the SSL corpus (2&times; frames at matched compute is
  neutral; six adapted blocks is harmful), and <em>vREx</em> as a headline (three variants across
  two environment definitions, all inside seed noise). Neither &ldquo;SSL&rdquo; nor
  &ldquo;DERM&rdquo; is closed: SSL is the encoder this report deploys, and DERM had never been run
  at all &mdash; see section 04.</div>
</div></section>

<div class="measure"><footer>
  All intervals 95%, clustered on pool (n = 24 labelled, 48 unlabelled v1, 36 v2). Every figure and
  every bracketed number is regenerated from source at build time by story_figures.py,
  event_eval.py, ppi_phase.py and build_report.py &mdash; none is transcribed by hand.
</footer></div>

</div>
'''
