# Consumed by build_report.py via exec(). Expects `img` (data URIs), the three interactive
# figures already JSON-injected (`CHART`, `DECAY`, `MODELS`) and the three JSON payloads they are
# views over, for inline numbers: `E` (estimates), `M` (models), `O` (outcome units).
n_lab = max((c['n_lab'] for c in E['cells'] if c['exp'] == 'v1' and c['method'] == 'ci'),
            default=24)

# Inline formatters. Every number below reads out of a JSON built by a script in this directory,
# so a rerun cannot leave the prose disagreeing with the tables.
def cv(u):
    d = O['units'][u]['cv']
    return f"{d['nn']:.2f} / {d['nt']:.2f}"


def rd(u):
    d = O['units'][u]['r_delta']
    return f"{d['nn']:.2f} / {d['nt']:.2f}"


def bs(u):
    d = O['units'][u]['bias_spread']
    return f"{d['nn']:.2f}× / {d['nt']:.2f}×"


nnf, ntf = (round(100 * O['bouts'][l]['one_frame']) for l in ('nn', 'nt'))
nnt, ntt = (round(100 * O['bouts'][l]['tail10']) for l in ('nn', 'nt'))


def run(tag):
    """One scored run, by directory name, from models.json."""
    return next(r for r in M['runs'] if r['tag'] == tag)


def dist(behav, bucket, field):
    """Error-distance statistics for the deployed model, from examples.json."""
    m = next(x for x in X['models'] if x['key'] == 'xfit')
    return m['annotated'][behav]['buckets'][bucket]['dist'][field]


def rr(tag):
    """The pair of r-delta values for a run, as the tables print them."""
    r = run(tag)
    return f"{r['rd_nt']:.3f} / {r['rd_nn']:.3f}"


# Section 05's shortlist. Named here, scored from models.json, so the table cannot drift from the
# runs -- and the `hi` classes are computed, not asserted, so the highlight always marks the
# actual column leader.
CANDIDATES = [
    ('BitFit, 6 blocks', 'res448_k2_bit6_d4'),
    ('full fine-tune, 2 blocks (seed 1)', 'res448_k2_ft2_d4photo_seed1'),
    ('SSL-adapted, frozen', 'res448_k2_frozen_d4photo_sslinit'),
    ('region-preserving head, 0.44 M', 'res448_k2_frozen_d4photo_rgrid4'),
]
_c = [(nice, run(tag)) for nice, tag in CANDIDATES]
_best = {k: max(r[k] for _, r in _c) for k in ('ap', 'rd_nt', 'rd_nn')}


def _td(r, k, fmt):
    cls = ' class="hi"' if r[k] == _best[k] else ''
    return f'<td{cls}>{format(r[k], fmt)}</td>'


cand_rows = '\n      '.join(
    f'<tr><td>{nice}</td>{_td(r, "ap", ".4f")}'
    f'<td>{r["f1_nt"]:.3f} / {r["f1_nn"]:.3f}</td>'
    f'{_td(r, "rd_nt", ".3f")}{_td(r, "rd_nn", ".3f")}</tr>' for nice, r in _c)

# The deployment row is the mean over the three cross-fitting folds, which between them hold out
# all 24 annotated pools. Averaged here rather than transcribed.
_folds = [r for r in M['runs'] if r['role'] == 'deployment fold']
xf = {k: sum(r[k] for r in _folds) / len(_folds)
      for k in ('ap', 'f1_nt', 'f1_nn', 'rd_nt', 'rd_nn')}

BODY = f'''
<div class="wrap">

<header class="top"><div class="measure">
  <p class="eyebrow">Mice v1 / v2 &middot; status &middot; 23 August 2026</p>
  <h1>Genotype under hormonal exposure</h1>
  <p class="lede">Three ASD-associated mouse lines, wild-type against heterozygous carriers of the
  same knockout, filmed in cages of four before, during and after two hormonal exposures. The
  programme asks how the genotype changes social behaviour. This report covers the step in front
  of that: the <b>average</b> effect of the exposure itself, pooled over genotypes &mdash; and the
  vision model that has to carry it to the 84 pools nobody has annotated.</p>
</div></header>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">01 &middot; Experiments</p>
  <h2>Two cohorts, one recording protocol</h2></div>
  <p>Both cohorts run the same six recordings per cage of four animals: three phases in fixed
  order &mdash; <b>H</b>abituation (30 min) &rarr; <b>O</b> exposure (15 min) &rarr; <b>P</b>ost
  (15 min) &mdash; crossed with two hormonal exposures, <b>fear</b> and <b>social</b>. All six
  share a cage, a day and an annotator. Video is 2060&sup2; at 30 fps, stored 512&sup2; at 5 fps.
  Throughout, <b>pool</b> means the cage of four; they share a line and a sex, and in v1 a
  genotype.</p>
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
      <tr><td>estimators available</td><td>CI, PPI++, PPCI</td><td class="lo">PPCI only</td></tr>
    </tbody></table></div></div>
<div class="measure">
  <p><b>Why this report is about the exposure and not the genotype.</b> Each cohort blocks the
  genotype contrast for a different reason, and neither reason touches the exposure.</p>
  <ul>
    <li><b>v1 puts genotype between cages, and annotation is not balanced across it.</b> 18 of the
    24 annotated pools are heterozygous, and annotator is confounded with genotype &mdash; MF scored
    18 het observations and no wild-type, CP scored 3 wild-type and no het. A genotype contrast here
    compares animals <em>and</em> the people who scored them.</li>
    <li><b>v2 puts genotype inside the cage, and nothing records which animal carries it.</b>
    <code>genotype</code> is the string <code>mixed</code> on all 216 observations, the per-frame
    labels drop the annotator's own animal indices, and the model emits one label per frame rather
    than per animal. This is a missing-record problem before it is a computer-vision one.</li>
  </ul>
  <p>The exposure contrast is untouched by both: taken <em>within</em> a pool, it cancels cage,
  genotype, sex and annotator by construction.</p>
  <div class="note"><b>Estimand.</b> The unit of analysis is the <b>pool</b>, clustered. The mean
  <em>within-pool</em> change in behaviour across one phase transition, per exposure. Consecutive
  transitions only &mdash; H&rarr;O and O&rarr;P. P&minus;H is their sum, not an independent
  contrast. The two exposures are separate treatments with opposite signs on nose-to-tail and are
  never pooled. Genotype-specific effects are the next layer, not this one.</div>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">02 &middot; Outcome</p>
  <h2>What is one unit of behaviour?</h2></div>
  <p>Behaviour is a continuous stream. Turning it into a number needs three decisions, and the
  biology supplies none of them.</p>
  <div class="scroll"><table>
    <thead><tr><th>decision</th><th>what this report chose</th><th>what it costs</th></tr></thead>
    <tbody>
      <tr><td>what counts as ONE event</td><td>a <b>bout</b> &mdash; one uninterrupted run of
        annotated frames</td><td>the run is defined at 5&nbsp;fps, so a real bout split by a
        two-frame gap becomes two</td></tr>
      <tr><td>what the DENOMINATOR is</td><td>per <b>minute of recording</b></td>
        <td>a rate that decays inside a phase depends on how long you watch &mdash; section 03</td></tr>
      <tr><td>what you MEASURE</td><td><b>how often</b> it starts</td>
        <td>says nothing about how long it lasts, which is a separate effect</td></tr>
    </tbody></table></div>
  <p>The third decision is the one with real alternatives. <b>Counts</b> (bouts per minute) measure
  how often the behaviour is initiated; <b>occupancy</b> (percentage of frames in it) how much of
  the recording it fills; <b>duration</b> (mean bout length) how long one bout lasts. Occupancy is
  close to counts &times; duration, so it is not a third independent choice so much as the product
  of the other two &mdash; and it inherits both of their noise sources.</p>
  <div class="scroll"><table>
    <thead><tr><th></th><th>counts</th><th>occupancy</th><th>duration</th></tr></thead>
    <tbody>
      <tr><td>within-cell CV &mdash; nn / nt</td>
        <td>{cv('counts')}</td><td>{cv('occupancy')}</td><td class="hi">{cv('duration')}</td></tr>
      <tr><td>r&Delta; against the model &mdash; nn / nt</td>
        <td class="hi">{rd('counts')}</td><td>{rd('occupancy')}</td><td>&mdash; no model head</td></tr>
      <tr><td>treatment-linked bias (max/min ratio across phases)</td>
        <td>{bs('counts')}</td><td>{bs('occupancy')}</td><td>&mdash;</td></tr>
      <tr><td>contrasts resolved of 8 &nbsp;<i>(not an argument &mdash; see below)</i></td>
        <td>{O['units']['counts']['resolves']}</td><td>{O['units']['occupancy']['resolves']}</td>
        <td>{O['units']['duration']['resolves']}</td></tr>
    </tbody></table></div>
  <p><b>Counts win on measurability.</b> At 5&nbsp;fps
  {nnf}% of nose-to-nose bouts and {ntf}% of nose-to-tail bouts last a <em>single frame</em>, so
  their length is set by sub-frame timing the pipeline introduced rather than by the animals
  &mdash; duration has almost no dynamic range to carry an effect. Occupancy is dominated by its
  tail: the longest 10% of bouts carry {nnt}% of all nose-to-nose behaviour time and {ntt}% of
  nose-to-tail, so a single long huddle moves the number more than ten short contacts. And on the
  quantity the vision model has to reproduce, counts are the clearly better target &mdash; the
  correlation between true and predicted <em>within-pool phase differences</em> is about twice as
  high on counts as on occupancy.</p>
  <div class="note warnbox"><b>Two things the resolved-contrast count does not license.</b>
  Choosing the outcome that yields the most rejections of the null is selection on significance, so
  it is reported above but is not the reason for the choice. And it is not simply a noise story
  either: duration has the <b>lowest</b> CV of the three units and still resolves
  {O['units']['duration']['resolves']} of 8. Nor does the choice of unit fix the model's
  treatment-linked bias &mdash; the predicted/true ratio moves by the same <em>factor</em> on both
  ({bs('counts')} against {bs('occupancy')}). That bias is real, and it is what DERM targets in
  section 04.</div>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">03 &middot; Effects</p>
  <h2>Every estimate, in one figure</h2></div>
  <p>Pick a cohort, an outcome unit, a behaviour and a breakdown. One panel per exposure, both
  phase transitions in each, and <b>three estimators</b>:</p>
  <div class="scroll"><table>
    <thead><tr><th>estimator</th><th>annotations it uses</th><th>cohorts</th><th>reads as</th></tr></thead>
    <tbody>
      <tr><td><b>CI</b></td><td>human only</td><td>v1 (24 pools)</td>
        <td>the answer, on a third of the data</td></tr>
      <tr><td><b>PPI++</b></td><td>human + AI</td><td>v1 (72 pools)</td>
        <td class="hi">the same answer, narrower &mdash; unbiased for ANY predictor</td></tr>
      <tr><td><b>PPCI</b></td><td>AI only, uncalibrated</td>
        <td class="hi">v1 (72) and v2 (36)</td>
        <td class="lo">sign and pattern only &mdash; it is on the model's scale</td></tr>
    </tbody></table></div>
  <p><b>Start on CI</b> and read the other two against it. PPCI is the only one that needs no
  annotation anywhere, which is why it is the only estimator that exists on v2. Sections 04 and 05
  say how far that model can be trusted.</p>
</div>
  <div class="figwrap">{CHART}</div>
<div class="measure">
  <p><b>The two exposures are different treatments and are never pooled.</b> Nose-to-nose rises on
  exposure under both (+0.65 fear, +0.47 social) and falls when it is withdrawn, but nose-to-tail
  runs in <em>opposite</em> directions (+0.35 fear, &minus;0.36 social) &mdash; averaging over
  exposures would cancel that effect outright.</p>
  <div class="note"><b>Two behaviours, not three.</b> The <code>behavior_type</code> column carries
  three codes, but <code>np</code> is not a third behaviour: cross-tabulated against the annotation
  files' own <code>Behavior</code> column over all 144 files, <code>nn</code> is nose-to-nose
  <b>mutual</b> and <code>np</code> is nose-to-nose <b>directional</b> (one animal sniffs, the other
  does not reciprocate). Labels are DIRECTED pairs, so the <code>nn</code> head predicts their union
  and the two cannot be reported separately.</div>
  <p><b>Switch the breakdown to the strata</b> and the reason for everything that follows is
  visible: annotation gave the wild-type strata <b>2 pools each</b> against the heterozygous
  strata's 6, so a stratified CI there has one degree of freedom and runs off the
  axis &mdash; and those are exactly the cells with 10 unlabelled pools apiece to borrow from.</p>
</div>
  <div class="figwrap">{DECAY}
    <p class="deccap">Pick a unit; both behaviours redraw. Hover any minute for its value and
    interval, or any bar for the phase mean it summarises.</p>
  </div>
<div class="measure">
  <h3 style="margin-top:26px">Nothing is stationary inside a phase</h3>
  <p>Rates fall several-fold across every recording: half-life <b>4&ndash;14 minutes</b>, and P
  decays fastest in every cell. One cell rises instead &mdash; nose-to-tail under social exposure
  during O &mdash; so the exposure sustains investigation while everything else habituates. That
  makes a phase <em>mean</em> an average over whichever stretch of a decaying curve the schedule
  happened to sample, and the two phases being compared do not sample the same stretch.</p>
  <p><b>The damage is confined to one of the two transitions.</b> O and P are both 15 minutes, so
  any window rule applied to both leaves O&rarr;P bit-for-bit unchanged &mdash; verified identical
  in all four cells. Only H&rarr;O moves, because habituation runs 30 minutes. Across the three
  defensible windows for it:</p>
  <div class="scroll"><table>
    <thead><tr><th>H &rarr; O</th><th>full H (30 min)</th><th>first 15</th><th>last 15</th><th>spread</th></tr></thead>
    <tbody>
      <tr><td>nt &middot; fear</td><td>+0.36</td><td>+0.22</td><td>+0.49</td><td>0.28</td></tr>
      <tr><td>nt &middot; social</td><td>&minus;0.37</td><td>&minus;0.67</td><td>&minus;0.07</td><td>0.60</td></tr>
      <tr><td>nn &middot; fear</td><td class="hi">+0.66</td><td class="hi">+0.45</td><td class="hi">+0.86</td><td>0.42</td></tr>
      <tr><td>nn &middot; social</td><td>+0.47</td><td>&minus;0.03</td><td>+0.97</td><td class="lo">1.01 &mdash; changes sign</td></tr>
    </tbody></table></div>
  <p><b>So nose-to-nose under social exposure cannot be a headline number.</b> It runs from
  &minus;0.03 to +0.97 across three defensible choices and changes sign; its full-window +0.47 is
  the H mean being dragged down by fifteen extra minutes of decay that O never gets.
  <b>nose-to-nose under fear is the one H&rarr;O effect that survives every window</b>, and all
  four O&rarr;P contrasts are window-invariant by construction.</p>
  <div class="note warnbox"><b>Which window to prefer turns on a confound: the phase-onset spike is
  not the treatment.</b> Every phase is a separate recording the experimenter starts by opening the
  cage, and <b>P &mdash; where the odour is <em>removed</em> &mdash; has the largest onset spike of
  the three in 3 of 4 cells</b> (first-2-min over last-2-min rate, nn&nbsp;&middot;&nbsp;fear: H 7.6,
  O 6.7, <b>P 12.3</b>). A response that peaks when the odour is taken away is handling. So the
  <b>first</b> 15 min of H matches onset position and cancels it, at the cost of contrasting
  cage-novelty with odour-novelty; the <b>last</b> 15 gives the truer baseline but charges the
  handling spike to the odour. Neither is clean &mdash; the choice has to be stated, not
  inherited.</div>

  <div class="sub">
    <p class="q">outcome design</p>
    <h3>The decay is a second effect, not a nuisance
      <span class="verdict v-part">report both</span></h3>
    <p>A slope or a time constant will not summarise it &mdash; log-linearity is rejected in 7 of
    12 cells and &tau; blows up wherever the slope nears zero (&minus;27 min on the one rising
    cell). What works is a <b>front-loading fraction</b> F = bouts in the first 5 minutes / bouts in
    the first 15: bounded, model-free, length-invariant, per-observation. Flat &rarr; 0.33, strong
    decay &rarr; higher.</p>
    <div class="scroll"><table>
      <thead><tr><th>&Delta;F</th><th>nt &middot; fear</th><th>nt &middot; social</th><th>nn &middot; fear</th><th>nn &middot; social</th></tr></thead>
      <tbody>
        <tr><td>H &rarr; O &nbsp;(odour ON)</td><td class="hi">&minus;0.40*</td><td>&minus;0.12</td><td class="hi">&minus;0.19*</td><td class="hi">&minus;0.11*</td></tr>
        <tr><td>O &rarr; P &nbsp;(odour OFF)</td><td>+0.19</td><td class="hi">+0.31*</td><td>+0.04</td><td class="hi">+0.27*</td></tr>
      </tbody></table></div>
    <p><b>Every sign is negative turning the odour on and positive turning it off</b> (* = resolves;
    5 of 8 do). The exposure flattens the habituation curve and withdrawing it restores fast
    habituation &mdash; not &ldquo;how much behaviour the odour triggers&rdquo; but &ldquo;how long
    it holds attention&rdquo;. F is undefined where an observation has no bouts in the window, so n
    falls to 17&ndash;24 by cell.</p>
    <p><b>Recommendation.</b> Report the <b>level</b> on a stated matched window and <b>&Delta;F</b>
    as the decay effect, per transition. Do not swap the level for a decay-corrected amplitude:
    extrapolating to t&nbsp;=&nbsp;0 multiplies minute 29.5 by ~25 and resolves 3 of 8 with
    intervals 2&ndash;4&times; wider &mdash; a small bias traded for a large variance.</p>
  </div>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">04 &middot; Model</p>
  <h2>A frame classifier for the 84 unlabelled pools</h2></div>
  <p>Only 24 of 108 pools are annotated, and none of v2. Everything in this section exists to put
  a number on the other 84: a per-frame detector whose per-observation aggregate can stand in for a
  human score.</p>
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

  <h3 style="margin-top:28px">Three metrics, and what each one is allowed to decide</h3>
  <p>Everything below is a comparison, so the measuring stick comes first. The three are not
  interchangeable, and the easiest one to compute is not the one that decides.</p>
  <div class="scroll"><table>
    <thead><tr><th>metric</th><th>what it measures</th><th>what it may decide</th></tr></thead>
    <tbody>
      <tr><td><b>macro AP</b></td><td>frame-level average precision, mean over the two
        behaviours, threshold-free. What training monitors.</td>
        <td>shortlisting only. Seed-noise floor <b>0.0089</b> from two identical runs, so a gap
        under ~0.01 is not a gap.</td></tr>
      <tr><td><b>event F1</b></td><td>bout-level: a predicted run of frames counts as a hit if it
        overlaps a true bout <em>at all</em>. The right resolution when {nnf}% of nn bouts last one
        frame.</td>
        <td>whether the detector finds the right <em>events</em> rather than the right frames.</td></tr>
      <tr><td><b>r&Delta;</b></td><td>correlation between true and predicted <em>within-pool phase
        differences</em>.</td>
        <td class="hi">the ranking. PPI's variance reduction is a function of r&Delta; and nothing
        else. It is <em>not</em> what training selects on.</td></tr>
    </tbody></table></div>
  <div class="note warnbox"><b>r&Delta; is the ranking key and also the noisiest number here.</b>
  On the standing 4-pool split it rests on <b>16 points</b> (4 pools &times; 2 exposures &times; 2
  transitions) at a threshold fitted to those same pools &mdash; a screen, not a measurement: enough
  to reject an arm, not to crown one. Only the cross-fitted folds in section 05, which hold out all
  24 annotated pools, give it an honest denominator. So every table below reports AP <em>and</em>
  r&Delta;, and the Spearman correlation between them across the
  {M['meta']['n_candidates']} candidate runs is only
  <b>{M['meta']['spearman_ap_vs_rdelta']:+.2f}</b>.</div>

  <h3 style="margin-top:28px">What actually moves the number</h3>
  <p>Every lever tried, in the order the subsections take them.</p>
  <div class="scroll"><table>
    <thead><tr><th>#</th><th>lever</th><th>what varies</th><th>&Delta; macro AP</th><th>verdict</th></tr></thead>
    <tbody>
      <tr><td>1</td><td>annotate more pools</td><td>data</td><td class="hi">+0.076 per doubling</td><td class="hi">the binding constraint, no plateau</td></tr>
      <tr><td>2</td><td>adapt the encoder (BitFit)</td><td>encoder</td><td class="hi">+0.112</td><td class="hi">recalibration, at 1/602nd the params</td></tr>
      <tr><td>3</td><td>SSL on unlabelled frames</td><td>encoder, label-free</td><td class="hi">+0.033</td><td class="hi">what we deploy; also reaches v2</td></tr>
      <tr><td>4</td><td>224 &rarr; 448 px input</td><td>input</td><td class="hi">+0.132</td><td>real, and now saturated</td></tr>
      <tr><td>5</td><td>vREx</td><td>objective</td><td class="lo">+0.008 at best, &minus;0.094 at &beta;=100</td><td class="lo">no help, and harmful when pushed</td></tr>
      <tr><td>6</td><td>DERM</td><td>objective</td><td>&mdash;</td><td class="v-part">four arms in training now</td></tr>
      <tr><td>&mdash;</td><td>head capacity (self-attn, multi-query)</td><td>head</td><td>&minus;0.003 to +0.014</td><td class="lo">flat</td></tr>
    </tbody></table></div>
  <div class="note warnbox"><b>One structural limit, before any of it.</b> The regime overfits
  &mdash; training loss falls monotonically while validation AP plateaus near epoch 24. That is why
  longer schedules and extra head capacity do nothing (row 4's saturation and the last row), and
  why the leverage sits in rows 1 and 2.</div>
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
  <div class="note"><b>~0.076 macro AP per doubling, and the curve has not bent.</b> Every
  modelling intervention below is worth between &minus;0.09 and +0.11, the best label-free one
  +0.033 &mdash; so twenty more annotated pools would beat all of them combined.</div>

  <div class="sub">
    <p class="q">ablation &middot; encoder</p>
    <h3>Adapting the encoder <span class="verdict v-yes">yes &mdash; +0.11 AP, for 70k params</span></h3>
    <p>Unfreezing the last DINOv2 blocks is the largest modelling gain measured: macro AP 0.4289
    frozen &rarr; 0.4889 at two blocks &rarr; 0.5243 at six, against a 0.0089 seed floor.</p>
    <p><b>What it is doing is recalibration, not new computation.</b> BitFit &mdash; training only
    biases, LayerNorm gains and LayerScale gains, and nothing that can form a new function of two
    patch features &mdash; matches and then beats full fine-tuning: <b>{run('res448_k2_bit6_d4')['ap']:.4f}
    with 70,656 trainable encoder parameters against 0.5243 with 42.5 M</b>, a 602&times; cut. Two
    head-capacity arms (patch self-attention, multi-query pooling) were flat. It also carries the
    best r&Delta; of any arm on nose-to-tail apart from full fine-tuning
    ({run('res448_k2_bit6_d4')['rd_nt']:.3f}), so this is not an AP-only win.</p>
    <div class="note"><b>Two checks on that comparison.</b> The BitFit arms ran with
    <code>d4</code> augmentation against a <code>d4_photo</code> control; against the
    <em>matched</em> <code>d4</code> control (0.4187) BitFit-6 is <b>+0.122</b>, so the mismatch
    understates the gain rather than manufacturing it. And encoder learning rate is decisive here,
    not incidental: BitFit reads 0.4509 at 1e-5 against 0.4902 at 1e-3.</div>
  </div>

  <div class="sub">
    <p class="q">ablation &middot; unlabelled frames</p>
    <h3>Self-supervised adaptation <span class="verdict v-yes">yes &mdash; and it is what we deploy</span></h3>
    <p>data2vec-style masked patch-feature regression against an EMA teacher, on <b>374,400</b>
    unlabelled frames over 104 pools spanning v1 and v2 &mdash; including the 216 v2 observations no
    supervised arm can reach. Stage&nbsp;B <b>freezes that encoder and trains the head only</b>, so
    the comparison is against the stock-DINOv2 frozen control. Drift after adaptation is 0.10, so
    the encoder genuinely moved and a null here would have been a result, not a failed run.</p>
    <div class="scroll"><table>
      <thead><tr><th>arm</th><th>encoder</th><th>macro AP</th><th>r&Delta; nt / nn</th><th>read</th></tr></thead>
      <tbody>
        <tr><td>frozen control (seed 1)</td><td>stock DINOv2</td><td>0.4200</td>
          <td>{rr('res448_k2_frozen_d4photo_decay30_seed1')}</td><td>reference</td></tr>
        <tr><td>SSL, 2 blocks, stride 10</td><td>adapted</td><td class="hi">0.4622</td>
          <td class="hi">{rr('res448_k2_frozen_d4photo_sslinit')}</td>
          <td class="hi">+0.033 with zero labels; best nn r&Delta; of any run</td></tr>
        <tr><td>&mdash; 2&times; the frames, matched compute</td><td>adapted</td><td>0.4524</td>
          <td>{rr('res448_k2_frozen_d4photo_ssl_s5_b2B')}</td><td>neutral</td></tr>
        <tr><td>&mdash; 6 blocks instead of 2</td><td>adapted</td><td class="lo">0.3831</td>
          <td>{rr('res448_k2_frozen_d4photo_ssl_s10_b6B')}</td><td class="lo">harmful</td></tr>
        <tr><td>BitFit 6 blocks</td><td>stock DINOv2</td><td class="hi">0.5409</td>
          <td class="hi">{rr('res448_k2_bit6_d4')}</td><td>best AP overall</td></tr>
        <tr><td>BitFit 6 blocks <b>on the SSL encoder</b></td><td>adapted</td><td>0.5127</td>
          <td>{rr('res448_k2_bit6_d4_sslinit')}</td><td class="lo">&minus;0.028 against BitFit alone</td></tr>
      </tbody></table></div>
    <p><b>+0.033 macro AP is 3.7&times; the seed spread</b>, bought with <em>zero labels</em>, and
    it is the only intervention that also reaches v2 &mdash; which is why it is the encoder every
    number in section 05 rests on. Note that the AP leader and the nn-r&Delta; leader are different
    arms; section 05 is about that disagreement.</p>
    <div class="note warnbox"><b>Two caveats of unequal strength.</b> That SSL and BitFit do not
    <em>stack</em> rests on a single pair, 0.5127 against 0.5409, one draw each &mdash; suggestive
    that both buy the same domain recalibration and therefore substitute, but not established; two
    seeds would settle it for ~3 GPU-hours. What is established, each a clean single-variable
    change: 2&times; the corpus at matched compute is neutral, and six adapted blocks is clearly
    harmful. So <em>scaling</em> the corpus is closed; SSL itself is not.</div>
  </div>

  <div class="sub">
    <p class="q">ablation &middot; input</p>
    <h3>Resolution against tokens <span class="verdict v-part">token-bound, now saturated</span></h3>
    <p>Going from 224 px to 448 px changes tokens and pixel detail at once. Capping the source
    pixels while holding the token count separates them: <b>tokens carried most of it</b>.</p>
    <div class="scroll"><table>
      <thead><tr><th>step</th><th>tokens</th><th>pixel detail</th><th>patches per mouse</th><th>macro AP</th><th>&Delta;</th></tr></thead>
      <tbody>
        <tr><td>224 px input</td><td>256</td><td>224 px</td><td>1.1</td><td>0.2966</td><td>&mdash;</td></tr>
        <tr><td>4&times; the tokens, detail held at 224 px</td><td>1,024</td><td>224 px</td><td>2.2</td>
          <td>0.3996</td><td class="hi">+0.103</td></tr>
        <tr><td>full pixels, tokens held</td><td>1,024</td><td>448 px</td><td>2.2</td>
          <td>0.4289</td><td class="hi">+0.029</td></tr>
        <tr><td>push tokens further (504 px)</td><td>1,296</td><td>504 px</td><td>2.5</td>
          <td>0.4315</td><td class="lo">+0.003</td></tr>
        <tr><td>224 px crop around a pair &mdash; <b>untried</b></td><td>256</td><td>native</td>
          <td class="hi">10</td><td>&mdash;</td><td class="hi">4.5&times; the detail at &frac14; the cost</td></tr>
      </tbody></table></div>
    <p>Whole-frame resolution is spent &mdash; a mouse is 2.2 patches at 448 px, and the full
    2060 px frame would cost 21,609 tokens and 466 GB of embeddings. <b>Cropping is the only lever
    left</b>, and it is the same fix any per-animal outcome needs. (The token step is confounded by
    +0.010 of photometric augmentation the 224 px arm lacked; and the pipeline resamples twice,
    2060&rarr;512&rarr;448, keeping 76% of the detail of a direct 2060&rarr;448 &mdash; free to
    recover by extracting frames at the working size.)</p>
  </div>

  <div class="sub">
    <p class="q">in progress &middot; deconfounding</p>
    <h3>DERM <span class="verdict v-part">in training &mdash; no result yet</span></h3>
    <p><b>The problem it targets.</b> Phase predicts <em>prevalence</em> here &mdash; the odour port
    visibly changes the scene &mdash; so a classifier can score a frame by which phase it
    <em>looks like</em> rather than by what the mice are doing. A bias that moves with the treatment
    is what corrupts an effect estimated without a rectifier, which is exactly the situation on v2.</p>
    <p><b>What DERM does.</b> It reweights every training sample by
    Var(<i>Y</i>|<i>E</i>)&thinsp;/&thinsp;P(<i>Y</i>,<i>E</i>) over a set of environments
    <i>E</i>. For a binary label that reduces to
    <code>w(y=1,&nbsp;e) = (1&minus;p<sub>e</sub>)/P(e)</code> and
    <code>w(y=0,&nbsp;e) = p<sub>e</sub>/P(e)</code>, which gives positives and negatives
    <b>equal mass inside every environment</b>: a raw prevalence spread of 3.5&times; across
    environments becomes exactly 0.5 in each. The mean weight is normalised to 1, so the effective
    step size is unchanged and the comparison against unweighted training is not confounded by a
    different learning rate. It never asks the model to be invariant to phase &mdash; only to stop
    the label carrying information about which phase it came from.</p>
    <p><b>Four arms.</b> Environments = the 3 <b>phases</b> (the estimand's own treatment
    variable), the 6 <b>phase &times; exposure</b> cells, a <b>seed replicate</b> of the phase arm,
    and <b>annotator</b> as a falsification test. That last one matters: annotator is exactly
    balanced across phase &mdash; every scorer took all six observations of each pool they touched,
    so H/O/P = 48/48/48 &mdash; so it <em>already cancels</em> in a within-pool phase contrast. If
    deconfounding against annotator moves r&Delta; as much as deconfounding against phase does,
    the mechanism is not the one described above.</p>
    <div class="note warnbox"><b>Status: mid-schedule, nothing to report.</b> All four arms are at
    epoch 18&ndash;24 of 30. They will be screened on <b>r&Delta;</b>, and whichever clears the
    controls will be cross-fitted over the three folds before any interval is quoted, because only
    out-of-fold predictions license a shrinkage number. No DERM figure appears anywhere in this
    report until then.</div>
  </div>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">05 &middot; Results</p>
  <h2>The model we run, and what it buys</h2></div>
  <p><b>Shortlist on frame AP; choose on the causal quantity.</b> Across the
  {M['meta']['n_candidates']} candidates the two rank models only loosely together (Spearman
  {M['meta']['spearman_ap_vs_rdelta']:+.2f}), and the disagreement is not academic &mdash; the
  best-AP arm of an earlier ablation was among the worst for the estimate. <b>Hover a point for the
  whole recipe behind it</b>: encoder, unlabelled-frame adaptation, fine-tuning, head and parameter
  count, augmentation, objective.</p>
</div>
  <div class="figwrap">{MODELS}</div>
<div class="measure">
  <div class="scroll"><table>
    <thead><tr><th>candidate</th><th>macro AP</th><th>event F1 nt / nn</th><th>r&Delta; nt</th><th>r&Delta; nn</th></tr></thead>
    <tbody>
      {cand_rows}
      <tr><td><b>cross-fitted deployment</b> (mean of 3 folds)</td><td>{xf['ap']:.3f}</td>
        <td>{xf['f1_nt']:.3f} / {xf['f1_nn']:.3f}</td><td>{xf['rd_nt']:.3f}</td>
        <td>{xf['rd_nn']:.3f}</td></tr>
    </tbody></table></div>
  <p class="defn"><b>Read the last row differently.</b> The first four are scored on the standing
  4-pool split &mdash; 24 observations, too few to separate close models, which is why r&Delta;
  cannot rank them. The last row is the <b>cross-fitted</b> deployment, and that needs defining.</p>
  <div class="note"><b>Cross-fitting.</b> Split the 24 annotated pools into three folds of 8; train
  on the other 16 and score only that fold. Every annotated pool ends up scored by a model that
  never saw it &mdash; a harder split than the standing one, hence the lower AP.
  <br><br><b>PPI++ is what makes it necessary.</b> Its rectifier is the gap between predictions on
  annotated pools and the truth there. A model trained on those pools predicts them too well, so
  the gap it measures is not the gap that applies to the 48 pools it corrects.
  <br><br><b>It is not needed to predict the unannotated pools</b> &mdash; no label is involved, so
  any model can. Their predictions are averaged over the three folds only so both sides of PPI++
  share one expectation. <b>PPCI uses no labels at all, so it is not bound by this</b> and would be
  better served by the single strongest model.</div>
  <div class="note warnbox"><b>The deployed configuration is not the strongest one measured.</b>
  Cross-fitting ran on the SSL-adapted frozen encoder with a plain 5.03 M head, not on BitFit-6,
  which leads on every accuracy axis. That choice bought label-free adaptation covering v2 as well,
  and every estimate here rests on it &mdash; but running BitFit-6 over the three folds, and over the
  unannotated pools for PPCI, is the cheapest outstanding improvement to the estimate at roughly
  18 GPU-hours.</div>

  <h3 style="margin-top:26px">Where it is right, where it is wrong, and what it sees
  where nobody looked</h3>
  <p>Pick a model, a set of frames and a behaviour. On the annotated pools every frame has a ground
  truth, so the four confusion buckets are meaningful and carry their counts. On the 84 unannotated
  pools there is no ground truth at all, so the only honest buckets are <em>confident yes</em> and
  <em>confident no</em> &mdash; that panel shows no counts and makes no accuracy claim.</p>
</div>
  <div class="figwrap">{EXAMPLES}</div>
<div class="measure">
  <div class="note warnbox"><b>The errors are not mostly boundary disagreements.</b> A natural
  reading of a confident false positive is that the model fired a frame or two outside a real bout.
  The figure's <b>d</b> column tests that directly &mdash; frames to the nearest scored bout in the
  same recording &mdash; and it does not hold: only <b>{dist('nn','FP','le2'):.0f}%</b> of
  nose-to-nose false positives and <b>{dist('nt','FP','le2'):.0f}%</b> of nose-to-tail ones sit
  within two frames of one, and the median is <b>{dist('nn','FP','median')}</b> and
  <b>{dist('nt','FP','median')}</b> frames respectively &mdash;
  {dist('nn','FP','median')/5:.0f} and {dist('nt','FP','median')/5:.0f} seconds from any scored
  behaviour. Most confident false positives are frames the model reads as contact where the
  annotator scored none, not mistimed edges. False negatives behave the same way
  ({dist('nn','FN','le2'):.0f}% and {dist('nt','FN','le2'):.0f}% within two frames of a detection),
  so a confidently missed bout is one the model saw nothing in anywhere nearby. That is a
  detection problem, and only double-annotation can say how much of it is genuine model error
  rather than label disagreement.</div>
  <p>What does hold: the confident detections on v2 &mdash; a cohort recorded months later with no
  annotations anywhere &mdash; do show genuine contact.</p>
  <div class="scroll"><table>
    <thead><tr><th>set</th><th>pools</th><th>observations</th><th>predicted nt</th><th>predicted nn</th><th>read</th></tr></thead>
    <tbody>
      <tr><td>v1 labelled (out-of-fold)</td><td>24</td><td>144</td><td>2.9&ndash;7.2%</td><td>8.3&ndash;13.1%</td><td>true 0.5&ndash;1.4% / 1.2&ndash;2.9%</td></tr>
      <tr><td>v1 unlabelled</td><td>48</td><td>288</td><td>3.9&ndash;10.1%</td><td>5.5&ndash;8.5%</td><td>plausible</td></tr>
      <tr><td>v2 (target cohort)</td><td>36</td><td>216</td><td>5.8&ndash;11.4%</td><td>8.1&ndash;12.1%</td><td>plausible, shifted up</td></tr>
    </tbody></table></div>
  <p>Ranges are over the six phase &times; exposure cells. Nothing collapses or saturates, and the
  v2 detections show genuine contact. Predicted occupancy runs about <b>5&times; above truth</b>
  throughout &mdash; part calibration offset, part the deliberate prior shift in training &mdash;
  so these must never be read as behaviour rates directly. For PPI++ that costs nothing, because
  &lambda; absorbs the scale. <b>For PPCI it is the whole caveat</b>: PPCI reports this scale, not
  the behaviour's, which is exactly why the figure gives it a separate axis and why nothing on this
  page reads a PPCI magnitude as a rate.</p>
</div>

<div class="measure">
  <div class="note warnbox"><b>CI and PPI++ do not target quite the same population.</b>
  Annotation is <b>3:1 het-enriched</b> (18 het / 6 wt against a 36/36 design), so CI estimates the
  effect <em>in the annotated pools</em> while PPI++ pulls in 48 unannotated ones that are
  wt-enriched (30 wt / 18 het) and targets the full 72. They coincide only if the phase effect does
  not vary with genotype, and it varies a little: nose-to-nose under fear reads about +0.79 across
  the wt strata against +0.60 across the het strata, moving the target roughly +0.05 (~7%). Small
  next to these intervals, but it is a difference in <em>estimand</em> rather than precision, so it
  does not shrink with more data. The fix is to estimate within stratum and recombine with design
  weights.</div>
</div></section>


<section><div class="measure">
  <div class="sechead"><p class="eyebrow">06 &middot; Next</p><h2>What to do next</h2></div>
  <div class="scroll"><table>
    <thead><tr><th></th><th>action</th><th>why now</th></tr></thead>
    <tbody>
      <tr><td>1</td><td>annotate ~20 more v1 pools</td><td>+0.076 AP per doubling and no plateau &mdash; worth more than every modelling change combined</td></tr>
      <tr><td>2</td><td>annotate 4&ndash;6 v2 pools</td><td>the only way v2 gets a CI or a PPI++ estimate at all; today it has PPCI and nothing to check it against</td></tr>
      <tr><td>3</td><td>fix the observation window on biological grounds</td><td>largest single lever on the headline number; currently inherited, not chosen</td></tr>
      <tr><td>4</td><td>land the four DERM arms and screen them on r&Delta;</td><td>in training now; the one objective-level idea this dataset actually motivates</td></tr>
      <tr><td>5</td><td>run BitFit-6 over the three folds and the unannotated pools</td><td>~18 GPU-h to put every estimate on the configuration that leads on accuracy &mdash; and PPCI needs no cross-fitting at all, so it can use the single best model</td></tr>
      <tr><td>6</td><td>per-animal crops from the 2060 px source</td><td>the only resolution lever left, and a prerequisite for any per-animal outcome</td></tr>
      <tr><td>7</td><td>record which animal in each v2 cage is the heterozygote</td><td>without it the within-pool genotype contrast is not identified no matter how good the vision gets</td></tr>
      <tr><td>8</td><td>double-annotate 15&ndash;20 observations</td><td>the only clean way to bound irreducible label noise, which caps everything above</td></tr>
    </tbody></table></div>
  <div class="note warnbox"><b>Closed.</b> Scaling the SSL corpus (2&times; the frames at matched
  compute is neutral; six adapted blocks is harmful) and vREx (four arms across two environment
  definitions, best +0.008). Everything else above is open.</div>
</div></section>

<div class="measure"><footer>
  All intervals 95%, clustered on pool (n = 24 labelled + 48 unlabelled on v1, 36 on v2). Built by
  <code>build_report.py</code> from four JSON payloads regenerated from the runs and the labels at
  build time &mdash; <code>build_estimates.py</code> (every effect), <code>build_models.py</code>
  (every scored run and its recipe), <code>build_outcome.py</code> (the outcome-unit numbers) and
  <code>build_decay.py</code> (the within-phase curves). Numbers quoted in the prose are read from
  those same payloads, so the text cannot drift from the tables. Figures still rendered ahead of
  time by <code>story_figures.py</code> and <code>event_eval.py</code>.
</footer></div>

</div>
'''
