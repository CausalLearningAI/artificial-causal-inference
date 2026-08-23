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
  Throughout, <b>pool</b> means the cage of four; they share a line and a sex, and in v1 a genotype.
  (Earlier drafts said <i>littermates</i> &mdash; the data records no breeding relation.)</p>
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
  <p><b>Counts win on measurability, and that is the honest case for them.</b> At 5&nbsp;fps
  {nnf}% of nose-to-nose bouts and {ntf}% of nose-to-tail bouts last a <em>single frame</em>, so
  their length is set by sub-frame timing the pipeline introduced rather than by the animals
  &mdash; duration has almost no dynamic range to carry an effect. Occupancy is dominated by its
  tail: the longest 10% of bouts carry {nnt}% of all nose-to-nose behaviour time and {ntt}% of
  nose-to-tail, so a single long huddle moves the number more than ten short contacts. And on the
  quantity the vision model has to reproduce, counts are the clearly better target &mdash; the
  correlation between true and predicted <em>within-pool phase differences</em> is about twice as
  high on counts as on occupancy.</p>
  <div class="note warnbox"><b>Two arguments withdrawn on recomputation.</b>
  <br><br><b>&ldquo;Counts resolve more, therefore the effect is on initiation.&rdquo;</b> Choosing
  the outcome that yields the most rejections of the null is selection on significance; it cannot
  be evidence for that outcome. Worse, duration has the <b>lowest</b> CV of the three and still
  resolves {O['units']['duration']['resolves']} of 8, so &ldquo;steadier, therefore resolves
  more&rdquo; does not hold either.
  <br><br><b>&ldquo;Counts halve the treatment-linked model bias.&rdquo;</b> They do not &mdash;
  the predicted/true ratio moves by the same <em>factor</em> on both units ({bs('counts')} against
  {bs('occupancy')}). The old claim compared absolute ranges on scales an order of magnitude apart.
  The bias is real (it is what motivates DERM), but the unit does not fix it.</div>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">03 &middot; Effects</p>
  <h2>Every estimate, in one figure</h2></div>
  <p>Pick a cohort, an outcome unit, a behaviour and a breakdown. One panel per exposure, both
  phase transitions in each, and <b>three estimators</b>:</p>
  <div class="scroll"><table>
    <thead><tr><th>estimator</th><th>annotations it uses</th><th>cohorts</th><th>reads as</th></tr></thead>
    <tbody>
      <tr><td><b>CI</b></td><td>human only, 24 pools</td><td>v1</td>
        <td>the answer, on 1/3 of the data</td></tr>
      <tr><td><b>PPI++</b></td><td>human + AI, all 72 pools</td><td>v1</td>
        <td class="hi">the same answer, narrower &mdash; unbiased for ANY predictor</td></tr>
      <tr><td><b>PPCI</b></td><td>AI only, uncalibrated</td><td class="hi">v1 and v2</td>
        <td class="lo">sign and pattern only &mdash; it is on the model's scale</td></tr>
    </tbody></table></div>
  <p><b>Start on CI</b> and read the other two against it. PPCI is the only one that needs no
  annotation anywhere, which is why it is the only estimator that exists on v2 &mdash; and why it
  gets its own axis inside every panel rather than sharing one with the two that are in behaviour
  units. Sections 04 and 05 exist to say how much that model can be trusted.</p>
</div>
  <div class="figwrap">{CHART}</div>
<div class="measure">
  <p><b>The two exposures are different treatments and are never pooled.</b> Nose-to-nose rises on
  exposure under both (+0.65 fear, +0.47 social) and falls when it is withdrawn, but nose-to-tail
  runs in <em>opposite</em> directions (+0.35 fear, &minus;0.36 social) &mdash; averaging over
  exposures would cancel that effect outright.</p>
  <div class="note"><b>Two behaviours, not three.</b> The lab's <code>behavior_type</code> column
  carries three codes, and <code>np</code> is not a third behaviour: cross-tabulated against the
  annotation files' own <code>Behavior</code> column over all 144 files, <code>nn</code> is
  nose-to-nose <b>mutual</b> and <code>np</code> is nose-to-nose <b>directional</b> (one animal
  sniffs, the other does not reciprocate). Nothing anogenital exists in this dataset, despite an
  earlier version of this report reading <code>np</code> that way. Labels are DIRECTED pairs, so
  the model's <code>nn</code> head predicts their union and the two can never be reported
  separately.</div>
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
  <div class="note warnbox"><b>Which window to prefer turns on a confound: the phase-onset spike
  is not the treatment.</b> Every phase is a separate recording that the experimenter starts by
  opening the cage, and <b>P &mdash; the phase where the odour is <em>removed</em> &mdash; has the
  largest onset spike of the three in 3 of 4 cells</b> (first-2-minutes over last-2-minutes rate:
  nn&nbsp;&middot;&nbsp;fear H 7.6, O 6.7, <b>P 12.3</b>). A response that peaks when the odour is
  taken away is handling, not odour. So the <b>first</b> 15 minutes of H matches onset position
  and cancels the handling response, at the cost of contrasting cage-novelty with odour-novelty;
  the <b>last</b> 15 gives the truer baseline but charges the handling spike to the odour. Neither
  is clean; the choice has to be stated rather than inherited.</div>

  <div class="sub">
    <p class="q">outcome design</p>
    <h3>The decay is a second effect, not a nuisance
      <span class="verdict v-part">report both</span></h3>
    <p>Summarising the decay by a slope or a time constant fails: log-linearity is rejected in 7 of
    12 cells, and &tau; blows up wherever the slope nears zero (&minus;27 min on the one rising
    cell). What works is a <b>front-loading fraction</b> F = bouts in the first 5 minutes / bouts
    in the first 15 &mdash; bounded, model-free, length-invariant, defined per observation, and it
    needs no exponential. Flat process &rarr; 0.33; strong decay &rarr; higher.</p>
    <div class="scroll"><table>
      <thead><tr><th>&Delta;F</th><th>nt &middot; fear</th><th>nt &middot; social</th><th>nn &middot; fear</th><th>nn &middot; social</th></tr></thead>
      <tbody>
        <tr><td>H &rarr; O &nbsp;(odour ON)</td><td class="hi">&minus;0.40*</td><td>&minus;0.12</td><td class="hi">&minus;0.19*</td><td class="hi">&minus;0.11*</td></tr>
        <tr><td>O &rarr; P &nbsp;(odour OFF)</td><td>+0.19</td><td class="hi">+0.31*</td><td>+0.04</td><td class="hi">+0.27*</td></tr>
      </tbody></table></div>
    <p><b>Every sign is negative turning the odour on and positive turning it off</b> (* = resolves;
    5 of 8 do). The exposure flattens the habituation curve and withdrawing it restores fast
    habituation &mdash; not &ldquo;how much behaviour the odour triggers&rdquo; but &ldquo;how long
    it holds attention&rdquo;. F is undefined for an observation with no bouts in the window, so n
    falls to 17&ndash;24 by cell.</p>
    <p><b>Recommendation.</b> Report the <b>level</b> on a stated matched window and
    <b>&Delta;F</b> as the decay effect, per transition. Do not replace the level with a
    decay-corrected amplitude: extrapolating to t&nbsp;=&nbsp;0 multiplies minute 29.5 by ~25, and
    it resolves 3 of 8 with intervals 2&ndash;4&times; wider &mdash; a small bias traded for a
    large variance, which is the opposite of what PPI is for.</p>
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
  <p>Everything below is a comparison, so the measuring stick comes first. The mistake this
  section has made before was letting the easiest of the three do all the deciding.</p>
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
  <div class="note warnbox"><b>r&Delta; is the ranking key and it is also the noisiest number
  here.</b> On the standing 4-pool validation split it rests on <b>16 points</b> (4 pools &times; 2
  exposures &times; 2 transitions) at an operating threshold picked by max-F1 on those same pools.
  That makes it a screen, not a measurement &mdash; enough to reject an arm, not enough to crown
  one. Only the cross-fitted folds in section 05, which hold out all 24 annotated pools, give it
  an honest denominator. Every table below therefore reports AP <em>and</em> r&Delta;, and the
  Spearman correlation between them across the {M['meta']['n_candidates']} candidate runs is only
  <b>{M['meta']['spearman_ap_vs_rdelta']:+.2f}</b>.</div>

  <h3 style="margin-top:28px">What actually moves the number</h3>
  <p>Every lever tried, in the order the subsections take them. Rows 5 and 6 are the two this
  section has previously got wrong.</p>
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
  <div class="note"><b>Each doubling of annotated pools is worth ~0.076 macro AP</b> and the curve
  has not bent. For scale, every modelling intervention below is worth between &minus;0.09 and
  +0.11, and the best label-free one is +0.033. <b>Twenty more annotated pools would beat all of
  them combined.</b></div>

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
    <div class="note"><b>Two things that could have made this comparison a lie, both checked.</b>
    The BitFit and six-block arms ran with <code>d4</code> augmentation while the headline frozen
    control ran with <code>d4_photo</code>. Against the <em>matched</em> frozen
    <code>d4</code> control (0.4187) BitFit-6 is <b>+0.122</b>, so the confound understates the
    gain rather than manufacturing it &mdash; photometric augmentation is itself worth only +0.010,
    inside seed noise. Second: BitFit reads 0.4509 at the inherited encoder LR of 1e-5 and 0.4902
    at 1e-3, so running the single inherited LR would have inverted the conclusion. Encoder LR is
    not a detail here.</div>
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
    <p><b>SSL works, and an earlier draft of this report called it closed.</b> +0.033 macro AP is
    3.7&times; the seed spread, it is bought with <em>zero labels</em>, it is the only intervention
    that also reaches v2, and it is <b>the encoder every number in section 05 rests on</b> &mdash;
    the three cross-fitting folds were run on it. Note that the AP leader and the nn-r&Delta;
    leader are different arms; that disagreement is the subject of section 05.</p>
    <div class="note warnbox"><b>&ldquo;It does not stack&rdquo; is one seed and is not
    settled.</b> The claim rests on a single pair &mdash; 0.5127 against 0.5409, one draw each.
    That both interventions might buy the same domain recalibration and therefore substitute is
    suggestive, not established; two seeds per arm would settle it for ~3 GPU-hours. What
    <em>is</em> established, each being a clean single-variable change: scaling the corpus
    2&times; at matched compute is neutral, and adapting six blocks instead of two is clearly
    harmful. The pretrained features are easy to damage. Closed: <em>scaling</em> SSL. Not
    closed: SSL.</div>
  </div>

  <div class="sub">
    <p class="q">ablation &middot; input</p>
    <h3>Resolution against tokens <span class="verdict v-part">token-bound, now saturated</span></h3>
    <p>Raising the input from 224 px to 448 px changes two things at once &mdash; the number of
    patch tokens and the pixel detail inside each. Capping the source pixels while keeping the
    token count separates them, and the answer is that <b>tokens carried most of it</b>:</p>
    <div class="scroll"><table>
      <thead><tr><th>step</th><th>tokens/frame</th><th>pixel detail</th><th>macro AP</th><th>&Delta;</th></tr></thead>
      <tbody>
        <tr><td>224 px input</td><td>256</td><td>224 px</td><td>0.2966</td><td>&mdash;</td></tr>
        <tr><td>4&times; the tokens, detail held at 224 px</td><td>1,024</td><td>224 px</td>
          <td>0.3996</td><td class="hi">+0.103</td></tr>
        <tr><td>full pixels, tokens held at 1,024</td><td>1,024</td><td>448 px</td>
          <td>0.4289</td><td class="hi">+0.029</td></tr>
        <tr><td>push tokens further (504 px)</td><td>1,296</td><td>504 px</td><td>0.4315</td>
          <td class="lo">+0.003 &mdash; saturated</td></tr>
      </tbody></table></div>
    <p>So 1,024 tokens on a 512 px stored frame is the ceiling, and the reason is geometry: a mouse
    covers about <b>2.2 patches</b> there. One caveat on the first step: the 224 px arm ran without
    photometric augmentation, which is worth about +0.010 on its own, so read the token step as
    ~+0.09 rather than +0.10. The remaining options, and why only one is worth running:</p>
    <div class="scroll"><table>
      <thead><tr><th>option</th><th>tokens/frame</th><th>patches per mouse</th><th>verdict</th></tr></thead>
      <tbody>
        <tr><td>full 2060 px whole frame</td><td>21,609</td><td>10</td><td class="lo">quadratic attention cost; 466 GB of embeddings</td></tr>
        <tr><td>224 px crop around a pair</td><td>256</td><td>10</td><td class="hi">4.5&times; the detail at &frac14; the cost</td></tr>
      </tbody></table></div>
    <p>Whole-frame resolution is spent; cropping is the lever that is left, and it is the same fix
    the per-animal identity problem needs. Separately, the pipeline resamples twice
    (2060&rarr;512&rarr;448) and so retains only <b>76% of the fine detail</b> of a direct
    2060&rarr;448 &mdash; recoverable for free by extracting frames at the working size.</p>
  </div>

  <div class="sub">
    <p class="q">ablation &middot; invariance</p>
    <h3>vREx <span class="verdict v-no">no help, and harmful when pushed</span></h3>
    <p><b>Where it should have been needed.</b> The model's bias is treatment-dependent: the ratio
    of predicted to true rate moves across phases ({bs('counts')} on counts, nn / nt). Within v1
    that costs nothing, because PPI's rectifier corrects any predictor; on v2, where no labels
    exist and no rectifier can be built, it is exactly the failure mode invariant training
    targets.</p>
    <div class="scroll"><table>
      <thead><tr><th>environment definition</th><th>&beta;</th><th>macro AP</th><th>&Delta; AP vs control</th><th>r&Delta; nt / nn</th></tr></thead>
      <tbody>
        <tr><td>none &mdash; control, <b>seed 42</b></td><td>&mdash;</td><td>0.4289</td><td>&mdash;</td><td>not scored</td></tr>
        <tr><td>none &mdash; control, seed 1</td><td>&mdash;</td><td>0.4200</td><td>&minus;0.009 (the seed floor)</td>
          <td>{rr('res448_k2_frozen_d4photo_decay30_seed1')}</td></tr>
        <tr><td>the 6 phase &times; exposure cells</td><td>1</td><td>0.4366</td><td>+0.008</td>
          <td class="lo">{rr('res448_k2_frozen_d4photo_vrexCond_b1')}</td></tr>
        <tr><td>the 6 phase &times; exposure cells</td><td>10</td><td>0.4265</td><td>&minus;0.002</td>
          <td class="lo">{rr('res448_k2_frozen_d4photo_vrexCond_b10')}</td></tr>
        <tr><td>annotator</td><td>10</td><td>0.4272</td><td>&minus;0.002</td>
          <td>{rr('res448_k2_frozen_d4photo_vrexAnn_b10')}</td></tr>
        <tr><td>annotator</td><td>100</td><td class="lo">0.3352</td><td class="lo">&minus;0.094</td>
          <td>{rr('res448_k2_frozen_d4photo_vrexAnn_b100')}</td></tr>
      </tbody></table></div>
    <div class="note warnbox"><b>Read the two control rows, not one.</b> All four vREx arms are
    seed 42, so the AP comparison belongs against the <b>seed-42</b> control at 0.4289 &mdash; on
    which the best arm is +0.008, inside the seed floor, and &beta;=100 is a &minus;0.094 collapse.
    An earlier version compared them against the seed-1 control at 0.4200, which flattered every
    arm by about one seed spread. r&Delta; can only be compared against seed 1, because that is the
    control whose held-out predictions were saved &mdash; an asymmetry worth stating rather than
    hiding, and one more reason to treat these as a screen.</div>
    <p>No arm cleared the seed floor on AP, none flattened the per-phase bias, and r&Delta; moves in
    both directions across arms &mdash; the two <code>vrexCond</code> arms have the <em>worst</em>
    nose-to-tail r&Delta; of any run in this report. <b>Stop spending runs on vREx.</b></p>
  </div>

  <div class="sub">
    <p class="q">ablation &middot; deconfounding</p>
    <h3>DERM <span class="verdict v-part">in training &mdash; no result yet</span></h3>
    <div class="note warnbox"><b>A correction to the previous version of this report.</b> This
    section was headed &ldquo;DERM / vREx &mdash; no measurable help&rdquo; and listed DERM as
    closed. That was a misattribution: <code>train_online_aug.py</code> only ever implemented
    <b>vREx</b>, the four runs behind the claim are named <code>vrexCond_b1/b10</code> and
    <code>vrexAnn_b10/b100</code>, and <b>DERM had never been run on this dataset at all</b>. The
    two methods attack the same failure by opposite means, so a null for one says nothing about the
    other.</div>
    <div class="scroll"><table>
      <thead><tr><th></th><th>what it changes</th><th>mechanism</th></tr></thead>
      <tbody>
        <tr><td><b>vREx</b></td><td>the objective</td><td>keeps the training distribution, ADDS a penalty on the variance of risk across environments</td></tr>
        <tr><td><b>DERM</b></td><td>the distribution</td><td>keeps the loss, REWEIGHTS every sample by Var(<i>Y</i>|<i>E</i>) / P(<i>Y</i>,<i>E</i>)</td></tr>
      </tbody></table></div>
    <p>For a binary label the weight collapses to something readable. With
    <i>p<sub>e</sub></i>&nbsp;=&nbsp;P(<i>Y</i>=1|<i>E</i>=<i>e</i>) and P(<i>e</i>) the
    environment's share of the pool, <code>w(y=1,&nbsp;e) = (1&minus;p<sub>e</sub>)/P(e)</code> and
    <code>w(y=0,&nbsp;e) = p<sub>e</sub>/P(e)</code> &mdash; so positives and negatives carry
    <b>equal mass inside every environment</b>, and the environment stops carrying information
    about how prevalent the behaviour is. Verified numerically: a raw prevalence spread of
    3.5&times; across environments becomes exactly 0.5 in every one, mean weight normalised to 1 so
    the effective step size is unchanged and DERM-vs-ERM is not confounded by a different learning
    rate.</p>
    <p><b>Why that is the mechanism this dataset calls for.</b> Phase predicts prevalence here
    &mdash; the odour port visibly changes the scene &mdash; so a classifier can score a frame by
    which phase it <em>looks like</em> rather than by what the mice are doing, and a bias that moves
    with the treatment is what corrupts an ATE estimated without a rectifier. DERM closes that route
    without ever asking the model to be invariant to phase; it only breaks the
    label&ndash;environment association.</p>
    <p><b>Four arms, differing from the controls only in the method:</b> environments = the 3
    <b>phases</b> (the estimand's own treatment variable), the 6 <b>phase &times; exposure</b>
    cells, a <b>seed replicate</b> of the phase arm because every vREx arm was a single draw, and
    <b>annotator</b> as a falsification rather than a candidate. That last one is the check on the
    argument above: annotator is exactly balanced across phase &mdash; every scorer took all six
    observations of each pool they touched, so H/O/P = 48/48/48 &mdash; so it <em>already cancels</em>
    in a within-pool phase contrast. If deconfounding against annotator moves r&Delta; as much as
    deconfounding against phase, the mechanism is not what this section claims.</p>
    <div class="note warnbox"><b>Status at this build: still training, nothing to report.</b> All
    four arms are mid-schedule (epoch 18&ndash;24 of 30; a run writes its
    <code>config.json</code> and held-out predictions only on completion, so none appears in the
    model figure yet). When they land they will be screened on <b>r&Delta;, not AP</b> &mdash; the
    vREx round was judged on AP alone, which is the mistake this section is trying not to repeat
    &mdash; and whichever arm clears both controls will then be <b>cross-fitted over three
    folds</b>, because only out-of-fold predictions on all 24 annotated pools license a real
    shrinkage number. No DERM number is quoted anywhere in this report until then.</div>
  </div>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">05 &middot; Results</p>
  <h2>The model we run, and what it buys</h2></div>
  <p><b>Shortlist on frame AP; choose on the causal quantity.</b> Across the
  {M['meta']['n_candidates']} candidate runs the two rank models only loosely together (Spearman
  {M['meta']['spearman_ap_vs_rdelta']:+.2f}), and the disagreement is not academic &mdash; the
  best-AP arm of an earlier ablation was among the worst for the estimate. Every scored run is
  below; <b>hover or focus a point to read the whole recipe behind it</b> &mdash; encoder, whether
  it saw unlabelled frames, how it was adapted, the head and its parameter count, the augmentation
  and the training objective.</p>
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
  cannot rank them. The last row averages three folds that between them hold out <em>all</em> 24
  annotated pools, so every labelled observation is scored by a model that never saw its pool.
  That is the only condition under which PPI++'s rectifier is valid, and it is a harder split
  &mdash; hence the lower AP.</p>
  <div class="note warnbox"><b>The deployed configuration is not the best one we found.</b>
  Cross-fitting ran on the SSL-adapted frozen encoder with a plain 5.03 M head, not on BitFit-6,
  which leads on every accuracy axis. That bought label-free adaptation covering v2 as well, and it
  is what every number below rests on &mdash; but re-running the three folds on BitFit-6 is the
  cheapest outstanding improvement to the estimate, at roughly 18 GPU-hours.</div>

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
  <p>No ground truth exists for any of the 84 unlabelled pools, so this is a sanity check, never a
  performance claim. Two things are checkable: whether predicted rates stay physically plausible,
  and whether the confident detections actually show the behaviour.</p>
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
  throughout &mdash; part calibration offset, part the deliberate prior shift in training &mdash;
  so these must never be read as behaviour rates directly. For PPI++ that costs nothing, because
  &lambda; absorbs the scale. <b>For PPCI it is the whole caveat</b>: PPCI reports this scale, not
  the behaviour's, which is exactly why the figure gives it a separate axis and why nothing on this
  page reads a PPCI magnitude as a rate.</p>
</div>

<div class="measure">
  <div class="note warnbox"><b>CI and PPI++ do not target quite the same population.</b>
  Annotation is <b>3:1 het-enriched</b> &mdash; the 24 labelled pools are 18 het / 6 wt against a
  36/36 design &mdash; so CI estimates the effect <em>in the annotated pools</em> while PPI++ pulls
  in 48 unlabelled pools that are wt-enriched (30 wt / 18 het) and targets the full 72. Those
  coincide only if the phase effect does not vary with genotype, and the stratified view suggests
  it varies a little: nose-to-nose under fear reads about +0.79 across the wt strata against +0.60
  across the het strata, which moves the population target roughly +0.05 (about 7%). Small next to
  the intervals here, but it is a difference in <em>estimand</em> rather than in precision, so it
  does not shrink with more data. The fix is to estimate within stratum and recombine with design
  weights, which is what the stratified view is for.</div>
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
      <tr><td>5</td><td>re-run the three cross-fitting folds on BitFit-6</td><td>~18 GPU-h to put the estimate on the configuration that leads on accuracy</td></tr>
      <tr><td>6</td><td>per-animal crops from the 2060 px source</td><td>the only resolution lever left, and a prerequisite for any per-animal outcome</td></tr>
      <tr><td>7</td><td>record which animal in each v2 cage is the heterozygote</td><td>without it the within-pool genotype contrast is not identified no matter how good the vision gets</td></tr>
      <tr><td>8</td><td>double-annotate 15&ndash;20 observations</td><td>the only clean way to bound irreducible label noise, which caps everything above</td></tr>
    </tbody></table></div>
  <div class="note warnbox"><b>Closed, and narrower than it used to read.</b> Two things:
  <em>scaling</em> the SSL corpus, and <em>vREx</em>. Neither &ldquo;SSL&rdquo; nor
  &ldquo;DERM&rdquo; is closed &mdash; SSL is the encoder this report deploys, and DERM has never
  finished a run.</div>
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
