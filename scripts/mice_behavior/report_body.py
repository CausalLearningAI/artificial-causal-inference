# Consumed by build_report.py via exec(); expects `img` (dict of data URIs) and `P` (ppi_results).
def f(t):
    return f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"

BODY = f'''
<div class="wrap">

<header class="top"><div class="measure">
  <p class="eyebrow">Mice v1 / v2 &middot; status &middot; 23 August 2026</p>
  <h1>Genotype under hormonal exposure</h1>
  <p class="lede">Three ASD-associated mouse lines, wild-type against heterozygous knockout
  littermates, filmed before, during and after two hormonal exposures. The programme asks how the
  genotype intervention changes social behaviour. This report covers the step in front of that:
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
      <tr><td>genotype</td><td>pure cage: 36 wt, 36 het</td><td>mixed cage: wt + het littermates</td></tr>
      <tr><td>lines</td><td colspan="2">ash1l / kdm6b / kmt5b, 1:1:1, sexes balanced within every line</td></tr>
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
    <img src="{img['behav']}" alt="Effects on bouts per minute for nose-to-tail, nose-to-nose and nose-to-anogenital, split by fear and social exposure.">
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
      <tr><td>np &middot; nose-to-anogenital</td><td>fear</td><td class="hi">+0.64</td><td class="hi">&minus;0.38</td></tr>
      <tr><td>np &middot; nose-to-anogenital</td><td>social</td><td>+0.03</td><td>&minus;0.03</td></tr>
    </tbody></table></div>
  <p><b>Two patterns are already legible.</b> Head-directed contact rises on exposure and falls
  when it is withdrawn &mdash; the sign flips between the two transitions in every cell where both
  resolve. And the exposures are genuinely different treatments: under fear the two head-directed
  behaviours move together (nn +0.65, np +0.64), while under social they come apart (nn +0.47,
  np +0.03) and nose-to-tail runs the other way (&minus;0.36). Averaging over exposures would
  cancel the nt effect outright.</p>
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
    <div class="step"><b>Output</b><span>2 sigmoids: nt, nn+np<br>multi-label</span></div>
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
  <div class="note warnbox"><b>Two structural limits, before any ablation.</b> The regime overfits
  &mdash; training loss falls monotonically while validation AP plateaus near epoch 24, so longer
  schedules and extra head capacity do nothing. And the head has <b>two outputs for three
  annotated behaviours</b>: nn and np are merged into one class, and np contributes nearly twice
  the positive frames (11,772 against 6,516), so every &ldquo;nn&rdquo; number the model produces
  is really about the union. Defensible under fear, where both move together; it dilutes the
  social result, where only nn moves.</div>

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
    <h3>Self-supervised adaptation <span class="verdict v-part">yes, but it does not stack</span></h3>
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
    <p>So: yes, as a label-free replacement for supervised recalibration; no, as an addition to it.
    The last row is the informative one &mdash; the SSL encoder <em>was</em> subsequently given
    supervised recalibration and a head, and the combination lands below BitFit on the stock
    encoder. Both interventions appear to be buying the same thing, and doing it twice costs
    accuracy. Scaling SSL does not help either: doubling the corpus at matched compute is neutral,
    and adapting six blocks instead of two is clearly harmful, so the pretrained features are easy
    to damage.</p>
  </div>

  <div class="sub">
    <p class="q">ablation &middot; invariance</p>
    <h3>DERM / vREx <span class="verdict v-no">no measurable help</span></h3>
    <p><b>Where it should have been needed.</b> The model's bias is treatment-dependent: the ratio
    of predicted to true rate varies across phases, and on the best-AP arm it varied enough to
    predict O&nbsp;&lt;&nbsp;H &mdash; the wrong sign on the primary effect. Within v1 that costs
    nothing, because PPI's rectifier corrects any predictor; on v2, where no labels exist and no
    rectifier can be built, it is exactly the failure mode that invariant training targets.</p>
    <div class="scroll"><table>
      <thead><tr><th>environment definition</th><th>&beta;</th><th>macro AP</th><th>read</th></tr></thead>
      <tbody>
        <tr><td>none (control)</td><td>&mdash;</td><td>0.4289</td><td>reference, seed noise &plusmn;0.009</td></tr>
        <tr><td>the 6 phase &times; exposure cells</td><td>1</td><td>0.4366</td><td>within noise</td></tr>
        <tr><td>the 6 phase &times; exposure cells</td><td>10</td><td>0.4265</td><td>within noise</td></tr>
        <tr><td>annotator</td><td>10</td><td>0.4272</td><td>within noise</td></tr>
        <tr><td>annotator</td><td>100</td><td class="lo">0.3352</td><td class="lo">penalty swamps the task</td></tr>
      </tbody></table></div>
    <p>Three variants across two environment definitions, all inside seed noise, and none of them
    flattened the per-phase bias. That is unsurprising in hindsight: switching the outcome from
    occupancy to bout counts already halved the treatment-linked bias (argument 5 in section 02),
    which is most of what vREx was there to remove. Keep the code; stop spending runs on it.</p>
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
    <thead><tr><th>candidate</th><th>macro AP</th><th>event F1 nt / nn+np</th><th>r&Delta; nt</th><th>r&Delta; nn+np</th></tr></thead>
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
  <figure><img src="{img['conf_nn']}" alt="Examples for the merged nose-to-nose plus nose-to-anogenital class.">
    <figcaption><b>nn+np</b>, same run and operating point.</figcaption></figure>
  </div>
<div class="measure">
  <h3 style="margin-top:26px">Detections where no annotator ever looked</h3>
  <p>No ground truth exists for any of the 84 unlabelled pools, so this is a sanity check and a
  qualitative exhibit, never a performance claim. Two things are checkable: whether predicted rates
  stay physically plausible, and whether the confident detections actually show the behaviour.</p>
</div>
  <div class="figwrap">
  <figure><img src="{img['un_nn_v1']}" alt="Confident merged-class detections on unannotated v1 pools.">
    <figcaption><b>Unannotated v1 &middot; nn+np.</b> The 48 pools PPI rectifies.</figcaption></figure>
  <figure><img src="{img['un_nt_v1']}" alt="Confident nose-to-tail detections on unannotated v1 pools.">
    <figcaption><b>Unannotated v1 &middot; nt.</b></figcaption></figure>
  <figure><img src="{img['un_nn_v2']}" alt="Confident merged-class detections on v2.">
    <figcaption><b>v2 &middot; nn+np.</b> A different cohort recorded months later, with zero
    annotations anywhere.</figcaption></figure>
  <figure><img src="{img['un_nt_v2']}" alt="Confident nose-to-tail detections on v2.">
    <figcaption><b>v2 &middot; nt.</b></figcaption></figure>
  </div>
<div class="measure">
  <div class="scroll"><table>
    <thead><tr><th>set</th><th>pools</th><th>observations</th><th>predicted nt</th><th>predicted nn+np</th><th>read</th></tr></thead>
    <tbody>
      <tr><td>v1 labelled (out-of-fold)</td><td>24</td><td>144</td><td>2.9&ndash;7.2%</td><td>8.3&ndash;13.1%</td><td>true 0.5&ndash;1.4% / 1.2&ndash;2.9%</td></tr>
      <tr><td>v1 unlabelled</td><td>48</td><td>288</td><td>3.9&ndash;10.1%</td><td>5.5&ndash;8.5%</td><td>plausible</td></tr>
      <tr><td>v2 (target cohort)</td><td>36</td><td>216</td><td>5.8&ndash;11.4%</td><td>8.1&ndash;12.1%</td><td>plausible, shifted up</td></tr>
    </tbody></table></div>
  <p>Ranges are over the six phase &times; exposure cells. Nothing collapses or saturates, and the
  v2 detections show genuine contact. Predicted occupancy runs about <b>5&times; above truth</b>
  throughout &mdash; part calibration offset, part the nn+np merge &mdash; so these must never be
  read as behaviour rates directly. It costs nothing downstream: PPI's &lambda; absorbs the scale.</p>
</div>

<div class="measure">
  <h3 style="margin-top:26px">Prediction-powered inference</h3>
  <p class="defn"><b>PPI++ on v1</b> combines {P['nn_fear']['n']} labelled pools carrying
  out-of-fold predictions with {P['nn_fear']['N']} unlabelled ones. It is <b>unbiased for any
  predictor</b>: the rectifier subtracts exactly what the unlabelled term added, so a miscalibrated
  model costs variance and never validity. <b>Transport to v2</b> has no labels anywhere and
  therefore no rectifier &mdash; it applies the calibration slope fitted on v1 to
  {P['nn_fear']['Nv2']} v2 pools.</p>
</div>
  <div class="figwrap"><figure>
    <img src="{img['ppi']}" alt="Classical, PPI++ and transported-to-v2 intervals for four behaviour by exposure cells.">
    <figcaption>H&rarr;O contrast on the occupancy scale, because the unlabelled inference stores a
    per-observation mean probability rather than a frame series.</figcaption>
  </figure></div>
<div class="measure">
  <div class="scroll"><table>
    <thead><tr><th>cell</th><th>within-cell r</th><th>classical</th><th>PPI++</th><th>CI</th><th>v2 transported</th></tr></thead>
    <tbody>
      <tr><td>nn+np &middot; fear</td><td class="hi">{P['nn_fear']['r']:.2f}</td><td>{f(P['nn_fear']['classical'])}</td><td>{f(P['nn_fear']['ppi'])}</td><td class="hi">&minus;{100*P['nn_fear']['shrink']:.0f}%</td><td class="hi">{f(P['nn_fear']['v2'])}</td></tr>
      <tr><td>nn+np &middot; social</td><td>{P['nn_social']['r']:.2f}</td><td>{f(P['nn_social']['classical'])}</td><td>{f(P['nn_social']['ppi'])}</td><td>&minus;{100*P['nn_social']['shrink']:.0f}%</td><td>{f(P['nn_social']['v2'])}</td></tr>
      <tr><td>nt &middot; fear</td><td>{P['nt_fear']['r']:.2f}</td><td>{f(P['nt_fear']['classical'])}</td><td>{f(P['nt_fear']['ppi'])}</td><td>&minus;{100*P['nt_fear']['shrink']:.0f}%</td><td>{f(P['nt_fear']['v2'])}</td></tr>
      <tr><td>nt &middot; social</td><td>{P['nt_social']['r']:.2f}</td><td>{f(P['nt_social']['classical'])}</td><td>{f(P['nt_social']['ppi'])}</td><td>&minus;{100*P['nt_social']['shrink']:.0f}%</td><td>{f(P['nt_social']['v2'])}</td></tr>
    </tbody></table></div>
  <p><b>The model helps in exactly one cell, and that is the honest result.</b> nn+np under fear
  exposure has a within-cell correlation of {P['nn_fear']['r']:.2f} and gains a
  {100*P['nn_fear']['shrink']:.0f}% narrower interval &mdash; worth roughly
  {24/(1-P['nn_fear']['shrink'])**2 - 24:.0f} extra annotated pools. The other three sit at
  r &asymp; 0.15&ndash;0.19 and revert to the classical interval, which is precisely what PPI is
  supposed to do when the predictor carries no information: lose nothing.</p>
  <div class="note warnbox"><b>Two caveats.</b> The within-cell r is far below the 0.72 obtained by
  pooling all cells, because pooling adds between-cell signal a single-contrast estimate cannot
  use &mdash; so the pooled figure overstates PPI's value for any one number here. And
  <b>the v2 column is an extrapolation, not an estimate with guarantees</b>: it stands or falls on
  the v1 calibration transferring to a cohort recorded months later. The encouraging evidence is
  that all four v1 signs reproduce on v2 and the fear cell lands at {P['nn_fear']['v2'][0]:+.2f}
  against v1's {P['nn_fear']['classical'][0]:+.2f}. Treat it as a hypothesis until a handful of v2
  pools are annotated.</div>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">06 &middot; Next</p><h2>What to do next</h2></div>
  <div class="scroll"><table>
    <thead><tr><th></th><th>action</th><th>why now</th></tr></thead>
    <tbody>
      <tr><td>1</td><td>annotate ~20 more v1 pools</td><td>+0.076 AP per doubling and no plateau &mdash; worth more than every modelling change combined</td></tr>
      <tr><td>2</td><td>annotate 4&ndash;6 v2 pools</td><td>converts the v2 extrapolation into a real PPI estimate with a rectifier</td></tr>
      <tr><td>3</td><td>fix the observation window on biological grounds</td><td>largest single lever on the headline number; currently inherited, not chosen</td></tr>
      <tr><td>4</td><td>split the merged class into nn and np heads</td><td>recovers the social-exposure dissociation the model cannot currently express</td></tr>
      <tr><td>5</td><td>re-run the three cross-fitting folds on BitFit-6</td><td>~18 GPU-h to put the estimate on the configuration that actually won</td></tr>
      <tr><td>6</td><td>per-animal crops from the 2064 px source</td><td>only resolution lever left, and the same fix the genotype question needs in v2</td></tr>
      <tr><td>7</td><td>amplitude + decay constant per phase</td><td>replaces a mean whose value depends on how long the recording ran</td></tr>
      <tr><td>8</td><td>double-annotate 15&ndash;20 observations</td><td>the only clean way to bound irreducible label noise, which caps everything above</td></tr>
    </tbody></table></div>
  <div class="note warnbox"><b>Closed, deliberately.</b> DERM as a headline and further SSL scaling.
  Both are characterised, neither pays, and the bias DERM targeted is largely gone once the outcome
  is bout counts. Keep the code; stop spending runs.</div>
</div></section>

<div class="measure"><footer>
  All intervals 95%, clustered on pool (n = 24 labelled, 48 unlabelled v1, 36 v2). Every figure and
  every bracketed number is regenerated from source at build time by story_figures.py,
  event_eval.py, ppi_phase.py and build_report.py &mdash; none is transcribed by hand.
</footer></div>

</div>
'''
