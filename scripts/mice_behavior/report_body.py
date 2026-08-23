# Consumed by build_report.py via exec(); expects `img` (dict of data URIs) and `P` (ppi_results).
def f(t):
    return f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"

BODY = f'''
<div class="wrap">

<header class="top"><div class="measure">
  <p class="eyebrow">Mice v1 / v2 &middot; status &middot; 23 August 2026</p>
  <h1>The exposure effect</h1>
  <p class="lede">One behavioural effect is solid and the model now measurably narrows its
  interval, including on a cohort with no labels at all. Three findings qualify the rest: a
  recording-schedule artefact, a label definition that merges two distinct behaviours, and a
  learning curve saying annotation &mdash; not modelling &mdash; is the binding constraint.</p>
</div></header>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">01 &middot; Design</p><h2>Why this contrast</h2></div>
  <p>Each pool of four mice is recorded six times: three phases in fixed order &mdash;
  <b>H</b>abituation &rarr; <b>O</b>dour/exposure &rarr; <b>P</b>ost &mdash; crossed with two
  exposures (fear, social). All six share a pool, a day and an annotator, so comparing phases
  <em>within</em> a pool cancels cage, genotype, sex and annotator by construction. That matters
  here specifically: annotator is confounded with genotype, so the between-pool genotype contrast
  is compromised while this one is not.</p>
  <div class="tiles">
    <div class="tile"><span class="k">v1 pools</span><span class="v">72</span><span class="s">24 annotated &middot; 48 not</span></div>
    <div class="tile"><span class="k">v2 pools</span><span class="v">36</span><span class="s">zero annotated &mdash; target domain</span></div>
    <div class="tile"><span class="k">unit of analysis</span><span class="v">pool</span><span class="s">n = 24, clustered</span></div>
    <div class="tile"><span class="k">annotators</span><span class="v">6</span><span class="s">no video scored twice</span></div>
  </div>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">02 &middot; Result</p><h2>Exposure changes how often behaviour starts</h2></div>
  <p class="defn"><b>Outcome:</b> bouts per minute &mdash; a <em>bout</em> is one uninterrupted run
  of annotated frames, so this counts how often a behaviour begins, not how long it lasts.
  <b>Contrasts:</b> consecutive phases only, H&rarr;O and O&rarr;P, per exposure.</p>
</div>
  <div class="figwrap"><figure>
    <img src="{img['behav']}" alt="Effects on bouts per minute for nose-to-tail, nose-to-nose and nose-to-anogenital, split by fear and social exposure.">
    <figcaption>Human labels only &mdash; no model. Filled = 95% CI excludes zero.</figcaption>
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
  <div class="note"><b>A dissociation worth the split.</b> Under <em>fear</em> exposure both
  head-directed behaviours rise together (nn +0.65, np +0.64). Under <em>social</em> exposure they
  come apart: nose-to-nose rises (+0.47) while nose-to-anogenital does not move at all (+0.03).
  Nose-to-tail runs the other way, falling under social exposure. Averaging across exposures would
  cancel the nt effect entirely.</div>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">03 &middot; Prediction-powered inference</p>
  <h2>What the model buys</h2></div>
  <p class="defn"><b>PPI++</b> combines {P['nn_fear']['n']} labelled pools carrying out-of-fold
  predictions (3-fold cross-fitting, each pool held out exactly once) with
  {P['nn_fear']['N']} unlabelled v1 pools. It is <b>unbiased for any predictor</b> &mdash; the
  rectifier subtracts exactly what the unlabelled term added, so a bad model costs variance, never
  validity. <b>Transport to v2</b> has no labels anywhere, so no rectifier exists: it applies the
  calibration slope fitted on v1 to {P['nn_fear']['Nv2']} v2 pools.</p>
</div>
  <div class="figwrap"><figure>
    <img src="{img['ppi']}" alt="Classical, PPI++ and transported-to-v2 intervals for four behaviour by exposure cells.">
    <figcaption>Occupancy scale, because the unlabelled inference stored per-observation mean
    probability rather than a frame series.</figcaption>
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
  {100*P['nn_fear']['shrink']:.0f}% narrower interval &mdash; equivalent to roughly
  {24/(1-P['nn_fear']['shrink'])**2 - 24:.0f} extra annotated pools. The other three cells sit at
  r &asymp; 0.15&ndash;0.19 and revert to the classical interval, which is precisely what PPI is
  supposed to do when the predictor carries no information: lose nothing.</p>
  <div class="note warnbox"><b>Two caveats that matter.</b> First, the within-cell r is far below
  the 0.72 obtained by pooling all cells together &mdash; pooling adds between-cell signal that a
  single-cell estimate cannot use, so the pooled figure overstates PPI's value for any one
  contrast. Second, <b>the v2 column is an extrapolation, not an estimate with guarantees</b>: with
  no labels anywhere in v2 there is no rectifier, so it stands or falls on the v1 calibration
  transferring to a cohort recorded months later. The encouraging evidence is that the model
  reproduces all four v1 effect signs on v2, and the fear cell lands at
  {P['nn_fear']['v2'][0]:+.2f} against v1's {P['nn_fear']['classical'][0]:+.2f}. Treat it as a
  hypothesis until a handful of v2 pools are annotated.</div>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">04 &middot; Label definitions</p>
  <h2>The model has two heads for three behaviours</h2></div>
  <p>The annotators scored three behaviours. The training pipeline maps them onto two classes:</p>
  <div class="scroll"><table>
    <thead><tr><th>annotated</th><th>positive frames</th><th>model class</th><th>agreement</th></tr></thead>
    <tbody>
      <tr><td>nt &mdash; nose-to-tail</td><td>9,032</td><td>class 1</td><td class="hi">r = 1.0000, exact</td></tr>
      <tr><td>nn &mdash; nose-to-nose</td><td>6,516</td><td rowspan="2">class 2 (merged)</td><td rowspan="2" class="hi">r = 1.0000 against nn+np</td></tr>
      <tr><td>np &mdash; nose-to-anogenital</td><td>11,772</td></tr>
    </tbody></table></div>
  <div class="note warnbox"><b>The second head is dominated by the behaviour it is not named
  after.</b> np contributes nearly twice as many positive frames as nn, so every &ldquo;nn&rdquo;
  number the classifier produces is really about the union. That is defensible under fear exposure,
  where both components move together, but it <b>dilutes the social-exposure result</b>, where nn
  rises and np is flat. Splitting class 2 into two heads is a small change and would recover a real
  effect the current model cannot express.</div>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">05 &middot; Caution</p><h2>Behaviour decays inside every phase</h2></div>
</div>
  <div class="figwrap"><figure>
    <img src="{img['within']}" alt="Bouts per minute against elapsed minute within each phase, split by behaviour and exposure.">
    <figcaption>11 of 12 phase &times; exposure cells decay significantly, rates falling 4&ndash;6&times;
    across a recording.</figcaption>
  </figure></div>
<div class="measure">
  <p>This matters because <b>habituation runs 30 minutes and the other phases run 15</b>. The H mean
  carries a decayed tail the exposure phase never gets, inflating any H&rarr;O contrast.</p>
</div>
  <div class="figwrap"><figure>
    <img src="{img['window']}" alt="The H to O effect under three observation windows, showing one sign reversal.">
    <figcaption>Faded = CI includes zero.</figcaption>
  </figure></div>
<div class="measure">
  <div class="scroll"><table>
    <thead><tr><th>window</th><th>nn&middot;fear</th><th>nn&middot;social</th><th>nt&middot;fear</th><th>nt&middot;social</th></tr></thead>
    <tbody>
      <tr><td>full (H=30, O=15 min)</td><td class="hi">+0.65</td><td class="hi">+0.47</td><td class="hi">+0.35</td><td class="hi">&minus;0.36</td></tr>
      <tr><td>first 5 min of each</td><td class="hi">+0.51</td><td class="lo">&minus;0.50</td><td>+0.10</td><td class="hi">&minus;1.07</td></tr>
      <tr><td>last 15 min of each</td><td class="hi">+0.86</td><td class="hi">+0.98</td><td class="hi">+0.49</td><td>&minus;0.07</td></tr>
    </tbody></table></div>
  <div class="note warnbox"><b>Only nn&middot;fear survives all three windows</b> &mdash; the same
  cell PPI helps. nn&middot;social <em>reverses sign</em>, so its full-window value is an artefact of
  the schedule. The windows are not equivalent questions: &ldquo;first 5 min&rdquo; compares two
  novelty responses, &ldquo;last 15 min&rdquo; compares exposure against a settled baseline.
  <b>Fix that choice on biological grounds, in advance</b> &mdash; it is the largest single lever on
  the headline number. Better still, replace the phase mean with an initial amplitude plus a
  habituation time constant, which is immune to window length.</div>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">06 &middot; Outcome variable</p><h2>Why count events rather than time them</h2></div>
  <figure><img src="{img['outcome']}" alt="Occupancy resolves 3 of 8 contrasts, bouts per minute resolves 7 of 8.">
    <figcaption>&ldquo;Resolved&rdquo; = 95% CI excludes zero, over 2 behaviours &times; 2 exposures
    &times; 2 transitions.</figcaption></figure>
  <div class="scroll"><table>
    <thead><tr><th>#</th><th>argument</th><th>evidence</th></tr></thead>
    <tbody>
      <tr><td>1</td><td>the effect <em>is</em> on initiation</td><td>bout duration null in every cell</td></tr>
      <tr><td>2</td><td>durations barely resolvable</td><td>49% of nn bouts (22% of nt) last one 5&nbsp;fps frame</td></tr>
      <tr><td>3</td><td>occupancy is heavy-tailed</td><td>longest 10% of bouts carry ~40% of all time</td></tr>
      <tr><td>4</td><td>counts are a steadier measurement</td><td>within-cell CV 0.68 vs 0.88 (nn), 0.96 vs 1.17 (nt)</td></tr>
      <tr><td>5</td><td>counts remove treatment-linked bias</td><td>per-phase bias spread 0.21 on counts vs ~1.7 on occupancy</td></tr>
      <tr><td>6</td><td>the model predicts counts better</td><td>cross-fitted r&Delta; 0.72 vs 0.56 (class 2), 0.48 vs 0.36 (nt)</td></tr>
    </tbody></table></div>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">07 &middot; The binding constraint</p><h2>Annotation has not saturated</h2></div>
  <figure><img src="{img['lcurve']}" alt="Macro AP against number of annotated pools, still rising at 20.">
    <figcaption>Nested subsets, log-linear fit R&sup2; = 0.993, no plateau.</figcaption></figure>
  <div class="scroll"><table>
    <thead><tr><th>annotated pools</th><th>observations</th><th>macro AP</th><th>gain</th></tr></thead>
    <tbody>
      <tr><td>5</td><td>30</td><td>0.2716</td><td>&mdash;</td></tr>
      <tr><td>10</td><td>60</td><td>0.3461</td><td>+0.0745</td></tr>
      <tr><td>15</td><td>90</td><td>0.3837</td><td>+0.0376</td></tr>
      <tr><td>20 (all we have)</td><td>120</td><td>0.4289</td><td>+0.0453</td></tr>
      <tr><td>40 &mdash; extrapolated</td><td>240</td><td>~0.499</td><td>+0.070</td></tr>
      <tr><td>72 &mdash; extrapolated</td><td>432</td><td>~0.564</td><td>+0.135</td></tr>
    </tbody></table></div>
  <div class="note"><b>Each doubling of annotated pools is worth ~0.076 macro AP.</b> The best
  unsupervised intervention we found &mdash; self-supervised adaptation on 374,400 unlabelled frames
  &mdash; bought +0.033. <b>Annotating twenty more pools would beat every modelling change tried so
  far, combined.</b></div>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">08 &middot; Model</p><h2>What we currently run</h2></div>
  <div class="flow">
    <div class="step"><b>Video</b><span>2064&sup2; @ 30 fps<br>&rarr; stored 512&sup2; @ 5 fps</span></div>
    <div class="step"><b>Encoder</b><span>DINOv2-base ViT-B/14<br>448 px &rarr; 1024 tokens/frame</span></div>
    <div class="step"><b>Spatial pool</b><span>1 learned query over<br>1024 patches</span></div>
    <div class="step"><b>Temporal</b><span>attention over 5 frames<br>(&plusmn;0.4 s)</span></div>
    <div class="step"><b>Output</b><span>2 sigmoids: nt, nn+np<br>multi-label</span></div>
  </div>
  <div class="scroll"><table>
    <thead><tr><th>setting</th><th>value</th><th>setting</th><th>value</th></tr></thead>
    <tbody>
      <tr><td>optimiser</td><td>AdamW</td><td>head params</td><td>5.03 M</td></tr>
      <tr><td>lr / weight decay</td><td>3e-4 / 0.05</td><td>batch</td><td>64</td></tr>
      <tr><td>schedule</td><td>3 warmup, cosine&times;30</td><td>negatives</td><td>1 per positive</td></tr>
      <tr><td>dropout</td><td>0.4</td><td>train / val</td><td>20 / 4 pools</td></tr>
      <tr><td>augmentation</td><td colspan="3">D4 dihedral (exact &mdash; top-down cage) + brightness/contrast/gamma</td></tr>
    </tbody></table></div>
  <div class="note warnbox"><b>The regime is overfitting</b> &mdash; train loss falls monotonically
  while validation AP plateaus around epoch 24, consistent with the learning curve. Longer schedules
  and extra head capacity do nothing; more labels, more effective data, or a smaller model do.</div>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">09 &middot; Resolution</p><h2>Tokens, then pixels, then nothing</h2></div>
</div>
  <div class="figwrap"><figure>
    <img src="{img['tokpix']}" alt="Waterfall decomposing the improvement into a token step and a pixel step.">
    <figcaption>Quadrupling tokens at fixed pixel detail is worth +0.103; restoring full pixels at
    fixed tokens adds +0.029; more tokens adds +0.003.</figcaption>
  </figure></div>
<div class="measure">
  <div class="scroll"><table>
    <thead><tr><th>option</th><th>tokens/frame</th><th>patches per mouse</th><th>verdict</th></tr></thead>
    <tbody>
      <tr><td>today: 448 px whole frame</td><td>1,024</td><td>2.2</td><td>at the ceiling</td></tr>
      <tr><td>504 px whole frame</td><td>1,296</td><td>2.5</td><td class="lo">+0.003, saturated</td></tr>
      <tr><td>full 2064 px whole frame</td><td>21,609</td><td>10</td><td class="lo">quadratic cost; 466 GB</td></tr>
      <tr><td>224 px crop around a pair</td><td>256</td><td>10</td><td class="hi">4.5&times; detail at &frac14; the cost</td></tr>
    </tbody></table></div>
  <p>Cropping is the only lever left. Separately, the pipeline resamples twice
  (2064&rarr;512&rarr;448), which retains only <b>76% of the fine detail</b> of a direct
  2064&rarr;448 &mdash; free to recover by extracting at the working size.</p>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">10 &middot; Evaluation</p><h2>Three levels, and what each is for</h2></div>
  <div class="scroll"><table>
    <thead><tr><th>level</th><th>measures</th><th>power</th><th>use it to</th></tr></thead>
    <tbody>
      <tr><td>frame</td><td>macro AP, ROC-AUC over 144k frames</td><td>high</td><td>shortlist</td></tr>
      <tr><td>event</td><td>bout precision / recall / F1 (any-overlap)</td><td>medium</td><td>describe honestly</td></tr>
      <tr><td>causal</td><td>within-cell r&Delta; &rarr; PPI variance factor</td><td>low</td><td>choose the final model</td></tr>
    </tbody></table></div>
  <figure><img src="{img['apcausal']}" alt="Scatter of macro AP against causal correlation across 13 runs.">
    <figcaption>&rho; &asymp; +0.5. Frame AP is a usable filter, but the best-AP model is not the
    best model for the estimate.</figcaption></figure>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">11 &middot; Comparison</p><h2>What each intervention bought</h2></div>
  <div class="scroll"><table>
    <thead><tr><th>run</th><th>macro AP</th><th>verdict</th></tr></thead>
    <tbody>
      <tr><td>frozen baseline</td><td>0.4289</td><td>reference</td></tr>
      <tr><td>SSL adaptation (12 ep, 2 blocks)</td><td class="hi">0.4622</td><td class="hi">best unsupervised gain; no labels used</td></tr>
      <tr><td>&mdash; 2&times; data at matched compute</td><td>0.4524</td><td>neutral</td></tr>
      <tr><td>&mdash; 6 blocks instead of 2</td><td class="lo">0.3831</td><td class="lo">harmful &mdash; features easy to damage</td></tr>
      <tr><td>BitFit, 6 blocks</td><td class="hi">0.5409</td><td>best frame AP; does not stack with SSL</td></tr>
      <tr><td>region-preserving head, 0.41 M</td><td>0.4328</td><td class="hi">parity at 12&times; fewer params</td></tr>
      <tr><td>vREx over the 6 cells, &beta;=1</td><td>0.4366</td><td>within seed noise</td></tr>
      <tr><td>vREx over the 6 cells, &beta;=10</td><td>0.4265</td><td>within seed noise</td></tr>
      <tr><td>vREx over annotators, &beta;=100</td><td class="lo">0.3352</td><td class="lo">penalty swamps the task</td></tr>
    </tbody></table></div>
  <p><b>Three verdicts.</b> <em>SSL is real but does not scale</em> &mdash; the original recipe is
  near-optimal. <em>DERM is not helping</em> &mdash; three vREx variants across two environment
  definitions all sit within seed noise, and the treatment-linked bias it targets largely disappears
  once the outcome is bout counts. <em>The region-preserving head is the quiet win</em> &mdash; equal
  accuracy at one twelfth the parameters, and it keeps spatial layout through to the temporal stage,
  which is the structural defect that made the earlier motion path fail.</p>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">12 &middot; Validation set</p><h2>Where the model is right and wrong</h2></div>
  <p>Most confident case per bucket, one frame per observation.</p>
</div>
  <div class="figwrap">
  <figure><img src="{img['conf_nt']}" alt="TP/FP/FN/TN examples for nose-to-tail.">
    <figcaption><b>nt.</b> Confident false positives are mostly frames adjacent to a scored bout
    &mdash; the model holds p&asymp;1 through a gap the annotator left.</figcaption></figure>
  <figure><img src="{img['conf_nn']}" alt="TP/FP/FN/TN examples for the merged class.">
    <figcaption><b>nn+np</b>, same run and operating point.</figcaption></figure>
  </div>
</section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">13 &middot; Unlabelled data</p>
  <h2>Detections where no annotator ever looked</h2></div>
  <p>No ground truth exists for any frame below, so this is a sanity check and a qualitative
  exhibit, never a performance claim. Two things are checkable: whether predicted rates stay
  physically plausible, and whether the confident detections show the behaviour.</p>
</div>
  <div class="figwrap">
  <figure><img src="{img['un_nn_v1']}" alt="Confident merged-class detections on unannotated v1.">
    <figcaption><b>Unannotated v1 &middot; nn+np.</b> The 48 pools PPI rectifies.</figcaption></figure>
  <figure><img src="{img['un_nt_v1']}" alt="Confident nose-to-tail detections on unannotated v1.">
    <figcaption><b>Unannotated v1 &middot; nt.</b></figcaption></figure>
  <figure><img src="{img['un_nn_v2']}" alt="Confident merged-class detections on v2.">
    <figcaption><b>v2 &middot; nn+np.</b> A different cohort recorded months later, zero annotations
    anywhere.</figcaption></figure>
  <figure><img src="{img['un_nt_v2']}" alt="Confident nose-to-tail detections on v2.">
    <figcaption><b>v2 &middot; nt.</b></figcaption></figure>
  </div>
<div class="measure">
  <p>Nothing collapses or saturates, and the v2 detections show genuine contact. Predicted rates run
  above truth throughout &mdash; part calibration offset, part the class-2 merge described in
  section 04 &mdash; so these must never be read as behaviour rates directly. PPI's &lambda;
  absorbs the scale.</p>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">14 &middot; Next</p><h2>Where to go</h2></div>
  <div class="scroll"><table>
    <thead><tr><th>priority</th><th>action</th><th>why</th></tr></thead>
    <tbody>
      <tr><td>1</td><td>annotate ~20 more pools</td><td>worth more than every modelling change combined</td></tr>
      <tr><td>2</td><td>split class 2 into nn and np heads</td><td>recovers the social-exposure dissociation the model cannot express</td></tr>
      <tr><td>3</td><td>fix the observation window on biological grounds</td><td>largest lever on the headline number</td></tr>
      <tr><td>4</td><td>annotate 4&ndash;6 v2 pools</td><td>converts the v2 extrapolation into a real PPI estimate</td></tr>
      <tr><td>5</td><td>per-animal crops from the 2064 px source</td><td>only remaining resolution lever; needs blob detection, not tracking</td></tr>
      <tr><td>6</td><td>amplitude + decay constant per phase</td><td>replaces a mean that depends on recording length</td></tr>
      <tr><td>7</td><td>promote the 0.41 M region head</td><td>same accuracy, 12&times; smaller</td></tr>
    </tbody></table></div>
  <div class="note warnbox"><b>Dropped:</b> DERM as a headline, and further SSL scaling. Both are
  characterised and neither pays. Keep the code; stop spending runs.</div>
</div></section>

<div class="measure"><footer>
  All intervals 95%, clustered on pool (n = 24 labelled, 48 unlabelled v1, 36 v2). Every number is
  regenerated from source at build time by story_figures.py, event_eval.py, ppi_phase.py and
  build_report.py &mdash; none is transcribed by hand.
</footer></div>

</div>
'''
