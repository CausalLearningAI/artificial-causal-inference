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
  <div class="scroll"><table>
    <thead><tr><th></th><th>v1</th><th>v2</th></tr></thead>
    <tbody>
      <tr><td>pools &times; observations</td><td>72 &times; 6 = 432</td><td>36 &times; 6 = 216</td></tr>
      <tr><td>design</td><td><b>12 pools per line &times; genotype</b><br>3 &times; 2 &times; 12 = 72</td><td><b>12 pools per line</b><br>3 &times; 12 = 36</td></tr>
      <tr><td>genotype</td><td>pure cage: 36 wt, 36 het</td><td>mixed cage: <b>3 wt + 1 het</b> per cage</td></tr>
      <tr><td>strata</td><td class="hi">6 (line &times; genotype)</td><td class="hi">3 (line)</td></tr>
      <tr><td>lines</td><td colspan="2"><i>Ash1l</i> / <i>Kdm6b</i> / <i>Kmt5b</i>, 1:1:1, sexes balanced within every line</td></tr>
      <tr><td>annotated</td><td class="hi">24 pools / 144 obs</td><td class="lo">none</td></tr>
      <tr><td>where genotype lives</td><td>between pools</td><td>within a pool</td></tr>
    </tbody></table></div>
  <p><b>The two cohorts fail in opposite ways for the genotype question, and that is why this
  report is about the exposure.</b> In v1 the genotype contrast is between cages, and annotation is
  not exchangeable across it: 18 of the 24 annotated pools are het, and annotator is confounded
  with genotype (MF scored 18 het observations and no wt; CP scored 3 wt and no het). In v2 the
  cage is mixed, so genotype is a within-pool contrast &mdash; but attributing a behaviour to one
  animal needs per-mouse identity, and the shave marks that carry it are destroyed by the
  512&sup2; downsample. Neither blocks the exposure contrast, which is taken <em>within</em> a
  pool and therefore cancels cage, genotype, sex and annotator by construction.</p>
  <div class="tiles">
    <div class="tile"><span class="k">unit of analysis</span><span class="v">pool</span><span class="s">n = 24 labelled, clustered</span></div>
    <div class="tile"><span class="k">unlabelled pools</span><span class="v">84</span><span class="s">48 in v1 &middot; 36 in v2</span></div>
    <div class="tile"><span class="k">annotators</span><span class="v">6</span><span class="s">22 of 24 pools single-scored</span></div>
    <div class="tile"><span class="k">annotated frames</span><span class="v">864k</span><span class="s">of 2.59 M in v1</span></div>
  </div>
  <div class="note"><b>Estimand.</b> The mean <em>within-pool</em> change in behaviour across one
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
  <div class="sechead"><p class="eyebrow">03 &middot; Preliminary patterns</p>
  <h2>What the 24 annotated pools already show</h2></div>
  <p>Human labels only, no model anywhere in this section.</p>
</div>
  <div class="figwrap"><figure>
    <img src="{img['behav']}" alt="Effects on bouts per minute for nose-to-tail and nose-to-nose, split by fear and social exposure.">
    <figcaption>Filled = 95% CI excludes zero. Unit of analysis is the pool (n = 24); each estimate
    is a mean of within-pool differences.</figcaption>
  </figure></div>
<div class="measure">
  <div class="scroll"><table>
    <thead><tr><th>behaviour</th><th>exposure</th><th>H &rarr; O</th><th>O &rarr; P</th></tr></thead>
    <tbody>
      <tr><td>nt &middot; nose-to-tail</td><td>fear</td><td class="hi">+0.35</td><td>&minus;0.11</td></tr>
      <tr><td>nt &middot; nose-to-tail</td><td>social</td><td class="hi">&minus;0.36</td><td class="hi">+0.27</td></tr>
      <tr><td>nn &middot; nose-to-nose</td><td>fear</td><td class="hi">+0.65</td><td class="hi">&minus;0.28</td></tr>
      <tr><td>nn &middot; nose-to-nose</td><td>social</td><td class="hi">+0.47</td><td class="hi">&minus;0.70</td></tr>
    </tbody></table></div>
  <p><b>Two patterns are already legible.</b> Nose-to-nose contact rises on exposure and falls
  when it is withdrawn &mdash; the sign flips between the two transitions, and both halves
  resolve under both exposures. And the exposures are genuinely different treatments: nose-to-nose
  rises on exposure under both (+0.65 fear, +0.47 social) while nose-to-tail runs in OPPOSITE
  directions (+0.35 fear, &minus;0.36 social). Averaging over exposures would cancel the nt effect
  outright, which is why they are never pooled.</p>
  <div class="note"><b>There are two behaviours here, not three.</b> The lab's
  <code>behavior_type</code> column carries three codes and an earlier version of this report
  read the third, <code>np</code>, as &ldquo;nose-to-anogenital&rdquo; &mdash; then built a
  three-behaviour dissociation on it. Cross-tabulating the code against the annotation files' own
  human-readable <code>Behavior</code> column over all 144 files shows what it really is:
  <code>nn</code> is <b>nose-to-nose, mutual</b> (<i>nose-nose_reciprocal</i>) and <code>np</code>
  is <b>nose-to-nose, directional</b> (<i>nose-nose_passive</i> &mdash; one animal sniffs, the
  other does not reciprocate). There is no anogenital behaviour anywhere in this dataset. The
  dissociation was an artefact of the misreading and has been removed.</div>
</div>
  <div class="figwrap"><figure>
    <img src="{img['within']}" alt="Bouts per minute against elapsed minute within each phase, with 95% bands, split by behaviour and exposure.">
    <figcaption>Bands are 95% intervals bootstrapped over the 24 pools. 11 of 12 phase &times;
    exposure &times; behaviour cells decay significantly.</figcaption>
  </figure></div>
<div class="measure">
  <p><b>Nothing is stationary inside a phase.</b> Rates fall several-fold across a recording, so a
  phase mean is an average over whatever stretch of that decay the schedule happened to sample.
  One cell is the exception and it is the interesting one: nose-to-tail under social exposure
  during O does not decay at all (slope +0.017/min, CI includes zero) &mdash; the exposure sustains
  investigation while everything else habituates.</p>
  <div class="note warnbox"><b>The window is not yet fixed, and it moves the headline.</b>
  Habituation runs 30 minutes against 15 for the other two phases, so H&rarr;O compares a mean
  taken over a longer tail of decay with one taken over a shorter one. Re-estimating on a matched
  last-15-minute window strengthens both fear effects and unsettles the social ones. That choice
  is a biological question &mdash; a novelty response and a settled baseline are different
  comparators &mdash; and it should be pre-specified rather than inherited. The durable fix is to
  stop summarising a phase by a mean at all: fit the decay and report an initial amplitude plus a
  habituation time constant, which no longer depends on recording length.</div>
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
  <div class="note warnbox"><b>One structural limit, before any ablation.</b> The regime overfits
  &mdash; training loss falls monotonically while validation AP plateaus near epoch 24, so longer
  schedules and extra head capacity do nothing.
  <br><br>The head's two outputs are <b>the right two</b>, which was not clear until the
  annotation vocabulary was checked. The label for a directed pair
  (<i>i</i>&nbsp;&rarr;&nbsp;<i>j</i>) is &ldquo;<i>i</i> directs nose-to-nose contact at
  <i>j</i>&rdquo;, so a mutual bout is a 1 for both animals and a one-sided bout only for the
  active one. That is exactly what <code>build_pair_labels.py</code> does. The consequence bounds
  what may be claimed downstream: the <code>nn</code> head predicts mutual and directional
  contact together, and no predictor here separates them, so <b>nn and np can never be reported
  as separate effects</b>. Splitting them would need its own label definition and its own head,
  not a regrouping of these outputs.</div>

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
  <h3 style="margin-top:26px">The estimates</h3>
  <p class="defn"><b>Three estimators of the same quantity.</b>
  <b>Classical</b> averages within-pool differences on human labels &mdash; unbiased, and confined
  to the {n_lab} annotated pools. <b>PPI++</b> adds the unlabelled pools and rectifies them
  against the labelled ones; it is <em>unbiased for any predictor</em>, so a miscalibrated model
  costs variance and never validity. <b>PPCI</b> is the plug-in: rescale the model's predictions
  and average them over <em>every</em> pool, annotated or not, with no rectifier. It is the only
  one of the three that exists on v2.</p>
  <div class="note"><b>The intercept is the rectifier &mdash; exactly.</b> Verified algebraically
  and numerically: PPCI over all <i>n</i>+<i>N</i> pools <em>with the fitted intercept restored</em>
  is identical to PPI++ with its power-tuned &lambda;, because
  &beta;&middot;<i>N</i>/(<i>n</i>+<i>N</i>) = &beta;/(1+<i>n</i>/<i>N</i>) = &lambda;. So the gap
  between the PPCI and PPI++ marks below is not a modelling detail &mdash; it <em>is</em> the
  correction PPI applies, drawn to scale. Because the estimand is a within-pool <em>difference</em>,
  any additive calibration offset cancels and the calibration has exactly one free parameter, the
  slope. None of the model's several-fold over-prediction of absolute rate reaches the estimate.</div>
</div>
  <div class="figwrap">{CHART}</div>
<div class="measure">
  <p><b>What the figure is for.</b> Pick a cohort, an outcome unit, a behaviour and a model, and it
  draws every estimator against the same axis, per exposure and per transition, so PPI++ and PPCI
  are read against the classical interval they are trying to beat. Switch <em>breakdown</em> to the
  strata &mdash; 6 line&nbsp;&times;&nbsp;genotype cells on v1, 3 lines on v2 &mdash; and the
  reason to do any of this becomes visible: <b>the wild-type strata hold 2 annotated pools each</b>
  against the heterozygous strata's 6, so a classical stratified interval there spans several
  bouts per minute and says nothing. Those are exactly the cells with 10 unlabelled pools apiece
  to borrow from.</p>
  <div class="note warnbox"><b>Read the shrinkage honestly.</b> A narrower interval is only worth
  having if it is still covering the truth. PPI++'s is, by construction, for any predictor.
  <b>PPCI's is not guaranteed</b>: it drops the rectifier, so its narrowing is bought partly with
  bias, and on v2 it additionally assumes a slope fitted on v1 transfers to a cohort recorded
  months later. Every interval in the figure uses the same
  <i>t</i><sub>n&minus;1</sub> quantile on the same pool-clustered scale, so the percentages are
  comparable across methods; an earlier version mixed <i>t</i> for classical with 1.96 for PPI and
  attributed the 5% difference to PPI.</div>
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
