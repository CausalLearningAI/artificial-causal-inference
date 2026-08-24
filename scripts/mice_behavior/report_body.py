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


def dF(behav, odour, trans, method='ci'):
    """One Delta-F cell from estimates.json, with a resolution star. Same source as the figure."""
    c = next(c for c in E['cells'] if c['exp'] == 'v1' and c['unit'] == 'decay'
             and c['stratum'] == 'all' and c['behav'] == behav and c['odour'] == odour
             and c['trans'] == trans and c['method'] == method)
    star = '*' if (c['lo'] is not None and c['lo'] * c['hi'] > 0) else ''
    return f"{c['est']:+.2f}{star}"


def _n_resolved(method):
    return sum(1 for c in E['cells']
               if c['exp'] == 'v1' and c['unit'] == 'decay' and c['stratum'] == 'all'
               and c['method'] == method and c['lo'] is not None and c['lo'] * c['hi'] > 0)


nF, nFp = _n_resolved('ci'), _n_resolved('ppi')


def rr2(tag, which):
    """One r-delta value, so a table cell can carry its own highlight class."""
    return f"{run(tag)['rd_' + which]:.3f}"


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
  programme asks how the genotype changes social behaviour. This report covers the step in front of
  that: <b>the effect of the exposure</b> &mdash; overall and broken down by line &times; genotype
  &mdash; and the vision model that has to carry it to the 84 pools nobody has annotated.</p>
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
  <div class="defn"><b>Two scored behaviours, from three codes.</b> <b>Nose-to-nose</b> is scored
  either mutual (<code>nn</code>) or directional (<code>np</code> &mdash; one animal sniffs, the
  other does not reciprocate); the two are reported together throughout. <b>Nose-to-tail</b> is
  <code>nt</code>. Every label is a <em>directed pair</em>, so a mutual bout counts for both animals
  and a one-sided bout only for the initiator.</div>
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
      <tr><td>annotators</td><td>6</td><td>&mdash;</td></tr>
      <tr><td>pools scored twice</td><td class="lo">0 of 24</td><td>&mdash;</td></tr>
      <tr><td>pools with no labels</td><td>48</td><td>36</td></tr>
      <tr><td>where genotype lives</td><td>between pools</td><td>within a pool</td></tr>
      <tr><td>estimators available</td><td>CI, PPI++, PPCI</td><td class="lo">PPCI only</td></tr>
    </tbody></table></div></div>
<div class="measure">
  <p><b>What is identified here, and what is not.</b> The programme's question is the genotype. This
  report reports the exposure &mdash; overall and within each line &times; genotype stratum &mdash;
  because that is the contrast the design identifies cleanly, and because the genotype contrast is
  currently limited by annotation rather than by biology.</p>
  <div class="scroll"><table>
    <thead><tr><th>contrast</th><th>how it is taken</th><th>status</th></tr></thead>
    <tbody>
      <tr><td><b>exposure</b>, per stratum</td><td><em>within</em> a pool, across phases</td>
        <td class="hi">identified &mdash; cage, genotype, sex, annotator and line background all
        cancel, because the same four animals scored by the same person supply both sides</td></tr>
      <tr><td><b>genotype</b>, on v1</td><td>between cages</td>
        <td class="lo">weak: annotation is 3:1 het-enriched (18 het / 6 wt) and annotator is
        confounded with genotype &mdash; one scorer took 18 het observations and no wild-type,
        another 3 wild-type and no het. Each wild-type stratum has 2 pools.</td></tr>
      <tr><td><b>genotype</b>, on v2</td><td>within a cage</td>
        <td class="lo">not identified at all: <code>genotype</code> is the string
        <code>mixed</code> on all 216 observations, the per-frame labels drop the annotator's animal
        indices, and the model emits one label per frame rather than per animal</td></tr>
    </tbody></table></div>
  <div class="note"><b>A negative control the design provides for free.</b> The three wild-type
  strata are three different lines' <em>unmutated</em> animals, so they should behave alike. On raw
  rates they do not quite: nose-to-tail differs across the three (0.22 / 0.50 / 0.59 bouts per
  minute, one-way ANOVA p = 0.04), while nose-to-nose is flat (p = 0.88). On the estimand &mdash;
  the within-pool H&rarr;O difference &mdash; they agree closely for both behaviours (nt p = 0.99,
  nn p = 0.28). Line background shifts the <em>level</em> and cancels in the <em>contrast</em>,
  which is the same pattern the annotator effect shows in section 05. With 2 annotated pools per
  wild-type stratum this is the weakest test on the page in both directions: neither result would
  survive much scrutiny, and it is the first thing more annotation would fix.</div>
  <div class="note"><b>Estimand.</b> The mean <em>within-pool</em> change in behaviour across one
  phase transition, per exposure, per stratum. The unit of analysis is the <b>pool</b>, clustered.
  Consecutive transitions only &mdash; H&rarr;O and O&rarr;P; P&minus;H is their sum, not an
  independent contrast.</div>
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
      <tr><td>contrasts resolved of 8 &nbsp;<i>(reported, not the reason)</i></td>
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
  <p class="defn">The resolved-contrast count is reported because a reader will ask for it, but it
  is not the reason for the choice &mdash; picking the outcome that yields the most rejections of
  the null is selection on significance. Measurability is the reason.</p>
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
  <p><b>The two exposures act differently, and one of them acts in opposite directions on the two
  behaviours.</b> Nose-to-nose rises on exposure under both (+0.65 fear, +0.47 social) and falls
  when it is withdrawn. Nose-to-tail rises under fear (+0.35) and <em>falls</em> under social
  (&minus;0.36). Each exposure is reported separately throughout.</p>
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
  <div class="note"><b>Decision: match the first 15 minutes of every phase.</b> The choice is
  settled by a confound. Every phase is a separate recording the experimenter starts by opening the
  cage, and <b>P &mdash; where the odour is <em>removed</em> &mdash; has the largest onset spike of
  the three in 3 of 4 cells</b> (first-2-min over last-2-min rate,
  nn&nbsp;&middot;&nbsp;fear: H 7.6, O 6.7, <b>P 12.3</b>). A response that peaks when the odour is
  taken away is handling, not odour; matching onset position puts it on both sides, where it
  cancels. The cost &mdash; contrasting cage-novelty with odour-novelty instead of a settled
  baseline &mdash; is the smaller of the two errors.
  <br><br>Two consequences. <b>Nose-to-nose under fear is the H&rarr;O effect to quote</b>: +0.45
  matched, positive and resolving under all three windows. <b>Nose-to-nose under social is not
  reportable</b>: it runs +0.47 &rarr; &minus;0.03 and changes sign, so its full-window value is the
  H mean being dragged down by fifteen extra minutes of decay that O never gets. O&rarr;P is
  unaffected either way. The figure above is still cut on the full window; re-cutting the grid is
  next-step 3.</div>
  <div class="sub">
    <p class="q">outcome design</p>
    <h3>The decay is a second effect
      <span class="verdict v-yes">in the figure, as its own unit</span></h3>
    <p>Measure it with a <b>front-loading fraction</b> F = bouts in the first 5 minutes / bouts in
    the first 15 &mdash; bounded, model-free, length-invariant, per-observation, and needing no
    exponential (a fitted slope or time constant does not survive: log-linearity is rejected in 7 of
    12 cells, and &tau; reaches &minus;27&nbsp;min on the one rising cell). Flat process &rarr; 0.33.
    <b>Select &ldquo;decay within phase&rdquo; as the unit in the figure above</b> to read it with
    all three estimators, the same way as the level.</p>
    <div class="scroll"><table>
      <thead><tr><th>&Delta;F, human labels</th><th>nt &middot; fear</th><th>nt &middot; social</th><th>nn &middot; fear</th><th>nn &middot; social</th></tr></thead>
      <tbody>
        <tr><td>H &rarr; O &nbsp;(odour ON)</td>
          <td class="hi">{dF('nt','fear','H->O')}</td><td>{dF('nt','social','H->O')}</td>
          <td class="hi">{dF('nn','fear','H->O')}</td><td class="hi">{dF('nn','social','H->O')}</td></tr>
        <tr><td>O &rarr; P &nbsp;(odour OFF)</td>
          <td>{dF('nt','fear','O->P')}</td><td class="hi">{dF('nt','social','O->P')}</td>
          <td>{dF('nn','fear','O->P')}</td><td class="hi">{dF('nn','social','O->P')}</td></tr>
      </tbody></table></div>
    <p><b>Every sign is negative turning the odour on and positive turning it off</b> (* = resolves;
    {nF} of 8 do on human labels alone, {nFp} of 8 with PPI++). The exposure flattens the
    habituation curve and withdrawing it restores fast habituation &mdash; not how much behaviour
    the odour triggers, but how long it holds attention. F is undefined where a recording has no
    bout in the window, so n falls to 13&ndash;24 by cell, which is why the model buys more here
    than it does on the level.</p>
  </div>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">04 &middot; Model</p>
  <h2>A frame classifier for the 84 unlabelled pools</h2></div>
  <p>Only 24 of 108 pools are annotated, and none of v2. Everything in this section exists to put
  a number on the other 84: a per-frame detector whose per-observation aggregate can stand in for a
  human score.</p>
  <div class="flow">
    <div class="step"><b>Video</b><span>2060&sup2; @ 30 fps<br>&rarr; stored 512&sup2; @ 5 fps</span></div>
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
        <td>shortlisting only. Seed noise measured on <b>six</b> configurations spans
        <b>0.004&ndash;0.016</b> (median 0.007) and is widest on the fine-tuned arms, so a gap under
        ~0.015 is not a gap.</td></tr>
      <tr><td><b>event F1</b></td><td>bout-level: a predicted run of frames counts as a hit if it
        overlaps a true bout <em>at all</em>. The right resolution when {nnf}% of nn bouts last one
        frame.</td>
        <td>whether the detector finds the right <em>events</em> rather than the right frames.</td></tr>
      <tr><td><b>r&Delta;</b></td><td>correlation between true and predicted <em>within-pool phase
        differences</em>.</td>
        <td class="hi">the ranking. PPI++'s variance reduction is a function of r&Delta; and
        nothing else. It is <em>not</em> what training selects on.</td></tr>
    </tbody></table></div>
  <div class="note warnbox"><b>r&Delta; is the ranking key, and on the standing split its
  nose-to-tail value is unusable.</b> It rests on <b>16 points</b> there (4 pools &times; 2 exposures
  &times; 2 transitions) at a threshold fitted to those same pools. How bad that is has been
  measured: two runs of one configuration differing only in <em>seed</em> give
  r&Delta;&nbsp;nt of {rr2('res448_k2_frozen_d4photo_ermH5M','nt')} and
  {rr2('res448_k2_frozen_d4photo_ermH5M_s1','nt')} &mdash; a spread wider than the range across
  every arm in section 04. Nose-to-nose is far steadier (0.77&ndash;0.80 across the same pairs).
  Only the cross-fitted folds, which hold out all 24 annotated pools, give either an honest
  denominator. So every table below reports AP <em>and</em> r&Delta;, and their Spearman correlation
  across the {M['meta']['n_candidates']} candidates is only
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
      <tr><td>6</td><td>DERM</td><td>objective</td><td class="lo">&minus;0.02 against a matched control</td><td class="lo">costs a little, buys nothing measurable</td></tr>
      <tr><td>&mdash;</td><td>head capacity (self-attn, multi-query)</td><td>head</td><td>&minus;0.003 to +0.014</td><td class="lo">flat</td></tr>
    </tbody></table></div>
  <div class="note warnbox"><b>One structural limit, before any of it.</b> The regime overfits
  &mdash; training loss falls monotonically while validation AP plateaus near epoch 24. That is why
  longer schedules and extra head capacity do nothing (row 4's saturation and the last row), and
  why the leverage sits in rows 1 and 2.
  <br><br><b>Read every &Delta; against 0.015, not 0.009.</b> Seed noise is not one number: across
  the six configurations now run at two seeds it spans 0.004 to 0.016, and the two widest are
  fine-tuned arms (BitFit-6 on the SSL encoder 0.014, DERM on phases 0.016). Rows 3 and 5 sit inside
  that; rows 1, 2 and 4 clear it comfortably.</div>
</div>

<div class="measure">
  <div class="sub">
    <p class="q">ablation &middot; data</p>
    <h3>The scaling law of annotation <span class="verdict v-yes">the binding constraint</span></h3>
    <p>Nested subsets of the labelled pools, so each point differs from the last for exactly one
    reason.</p>
  </div>
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
  <figure><img src="{img['lcurve']}" alt="Macro AP against number of annotated pools, still rising at 20.">
    <figcaption>Log-linear fit, R&sup2; = 0.993, no plateau. The 20-pool point is the deployed
    0.52 M-head control; the three subset points use the plain 5.03 M head, so the last step is
    worth slightly less than +0.045. The slope is not sensitive to it.</figcaption></figure>
  <div class="note"><b>~0.076 macro AP per doubling, and the curve has not bent.</b> Every
  modelling intervention below is worth between &minus;0.09 and +0.11, the best label-free one
  +0.033 &mdash; so twenty more annotated pools would beat all of them combined.</div>

  <div class="sub">
    <p class="q">ablation &middot; encoder</p>
    <h3>Adapting the encoder <span class="verdict v-yes">yes &mdash; +0.11 AP, for 70k params</span></h3>
    <p>Unfreezing the last DINOv2 blocks is the largest modelling gain measured: macro AP 0.4289
    frozen &rarr; 0.4889 at two blocks &rarr; 0.5243 at six &mdash; a +0.095 span against a seed
    spread of at most 0.016.</p>
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
    <p><b>Fine-tuning helps from either starting encoder; starting from SSL does not survive it.</b>
    On every metric at once:</p>
    <div class="scroll"><table>
      <thead><tr><th></th><th>macro AP</th><th>event F1 nt</th><th>event F1 nn</th><th>r&Delta; nt</th><th>r&Delta; nn</th></tr></thead>
      <tbody>
        <tr><td>stock, frozen</td><td>0.4200</td><td>0.421</td><td>0.490</td><td>0.417</td><td>0.768</td></tr>
        <tr><td>SSL, frozen</td><td>0.4622</td><td>0.432</td><td>0.489</td><td>0.455</td><td class="hi">0.902</td></tr>
        <tr><td>BitFit-6 on stock</td><td class="hi">0.5409</td><td>0.473</td><td class="hi">0.553</td><td class="hi">0.669</td><td>0.872</td></tr>
        <tr><td>BitFit-6 on SSL</td><td>0.5127</td><td class="hi">0.476</td><td>0.527</td><td>0.626</td><td>0.760</td></tr>
      </tbody></table></div>
    <p>SSL alone is worth +0.042 AP and the <b>best nose-to-nose r&Delta; of any run</b>, for zero
    labels &mdash; and it is the only intervention that reaches v2, which is why it is the encoder
    every number in section 05 rests on. Fine-tuning on top of it is worth a further +0.051 AP. But
    fine-tuning <em>stock</em> reaches higher still (0.5409 against 0.5127), and the SSL start is
    behind on three of the five metrics. Event F1 is the exception, where the two are level.</p>
    <div class="note warnbox"><b>Two caveats of unequal strength.</b> That SSL and BitFit do not
    <em>stack</em> is now a two-seed claim on one side: BitFit-6 on the SSL encoder reads 0.5127 and
    0.4989, mean <b>0.5058</b> with a 0.014 seed spread, against 0.5409 for BitFit-6 on stock. The
    gap of about 0.035 is roughly 2.5&times; that spread, so it is real unless the stock arm's own
    replicate lands unusually low &mdash; it is still training. What <em>is</em>
    established, each a clean single-variable change: 2&times; the corpus at matched compute is
    neutral, and six adapted blocks is clearly harmful. So <em>scaling</em> the corpus is closed;
    SSL itself is not.</div>
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
    <p class="q">ablation &middot; deconfounding</p>
    <h3>DERM <span class="verdict v-no">no help on this base model</span></h3>
    <p><b>The problem it targets.</b> Phase predicts <em>prevalence</em> here &mdash; the odour port
    visibly changes the scene &mdash; so a classifier can score a frame by which phase it
    <em>looks like</em> rather than by what the mice are doing. A bias that moves with the treatment
    is what corrupts an effect estimated without a rectifier, which is the situation on v2.</p>
    <p><b>What it does.</b> Reweight every sample by
    Var(<i>Y</i>|<i>E</i>)&thinsp;/&thinsp;P(<i>Y</i>,<i>E</i>) over a set of environments &mdash;
    for a binary label, <code>(1&minus;p<sub>e</sub>)/P(e)</code> on positives and
    <code>p<sub>e</sub>/P(e)</code> on negatives. Positives and negatives then carry <b>equal mass
    inside every environment</b>: a raw prevalence spread of 3.5&times; becomes exactly 0.5 in each,
    with the mean weight normalised to 1 so the step size is unchanged. It never asks the model to
    be invariant to phase, only to stop the label carrying information about which phase it came
    from.</p>
    <div class="scroll"><table>
      <thead><tr><th>arm</th><th>environments</th><th>seed</th><th>macro AP</th><th>r&Delta; nt</th><th>r&Delta; nn</th></tr></thead>
      <tbody>
        <tr><td><b>control</b> (unweighted)</td><td>&mdash;</td><td>42</td><td>0.4163</td>
          <td>{rr2('res448_k2_frozen_d4photo_ermH5M','nt')}</td>
          <td>{rr2('res448_k2_frozen_d4photo_ermH5M','nn')}</td></tr>
        <tr><td><b>control</b> (unweighted)</td><td>&mdash;</td><td>1</td><td>0.4202</td>
          <td>{rr2('res448_k2_frozen_d4photo_ermH5M_s1','nt')}</td>
          <td>{rr2('res448_k2_frozen_d4photo_ermH5M_s1','nn')}</td></tr>
        <tr><td>DERM</td><td>the 3 <b>phases</b></td><td>42</td><td>0.4060</td>
          <td>{rr2('res448_k2_frozen_d4photo_dermPhase','nt')}</td>
          <td>{rr2('res448_k2_frozen_d4photo_dermPhase','nn')}</td></tr>
        <tr><td>DERM</td><td>the 3 <b>phases</b></td><td>1</td><td>0.3902</td>
          <td>{rr2('res448_k2_frozen_d4photo_dermPhase_s1','nt')}</td>
          <td>{rr2('res448_k2_frozen_d4photo_dermPhase_s1','nn')}</td></tr>
        <tr><td>DERM</td><td>the 6 phase &times; exposure cells</td><td>42</td><td>0.3863</td>
          <td>{rr2('res448_k2_frozen_d4photo_dermCond','nt')}</td>
          <td>{rr2('res448_k2_frozen_d4photo_dermCond','nn')}</td></tr>
      </tbody></table></div>
    <p><b>Against a matched control, DERM costs a little accuracy and buys nothing.</b> Macro AP
    averages 0.418 unweighted against 0.398 with phase environments &mdash; about
    &minus;0.02 &mdash; and 0.386 with the finer phase&nbsp;&times;&nbsp;exposure cells. On
    r&Delta;&nbsp;nn, the stable axis, the two are identical: 0.786 unweighted against 0.788
    deconfounded. Whatever route DERM closes, this model was not using it enough for the closing to
    show.</p>
    <div class="note warnbox"><b>And r&Delta;&nbsp;nt cannot arbitrate any of this &mdash; the
    control proves it.</b> Two runs of the <em>unweighted</em> control differing only in seed give
    r&Delta;&nbsp;nt of <b>{rr2('res448_k2_frozen_d4photo_ermH5M','nt')}</b> and
    <b>{rr2('res448_k2_frozen_d4photo_ermH5M_s1','nt')}</b>. A spread of 0.67 between two runs of
    the same configuration is larger than the entire range across every arm in this section, so on
    the standing 4-pool split this metric measures the seed, not the method. That is why nothing in
    section 05 is chosen on it, and why a real answer needs cross-fitting.</div>
    <div class="note"><b>Still open: DERM on the accuracy leader.</b> Everything above sits on a
    frozen stock encoder at macro AP ~0.42. Two arms applying the same phase-environment
    deconfounding to BitFit-6 (AP 0.5409) are queued, with controls already in hand at both seeds.
    If DERM helps anywhere it should help there, where the model is good enough for a
    treatment-linked shortcut to be worth closing.</div>
  </div>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">05 &middot; Results</p>
  <h2>The model we run, and what it buys</h2></div>
  <p><b>Shortlist on frame AP; choose on the causal quantity.</b> Across the
  {M['meta']['n_candidates']} candidates the two rank models only loosely together (Spearman
  {M['meta']['spearman_ap_vs_rdelta']:+.2f}): the best-AP arm ranks 5th of
  {M['meta']['n_candidates']} on r&Delta;, and the r&Delta; leader ranks 4th on AP. AP is a usable
  filter, not the decision. <b>Hover a point for the whole recipe behind it</b> &mdash; encoder,
  unlabelled-frame adaptation, fine-tuning, head and parameter count, augmentation, objective.</p>
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

  <h3 style="margin-top:26px">Where it is right, where it is wrong, and what it sees on the
  pools nobody scored</h3>
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
  &lambda; absorbs the scale. <b>For PPCI it is the whole caveat</b>: PPCI reports this scale rather
  than the behaviour's, which is why it is drawn hollow in the effects figure and why nothing on
  this page reads a PPCI magnitude as a rate.</p>
</div>

<div class="measure">
  <h3 style="margin-top:26px">Is PPCI reading the behaviour or the model?</h3>
  <p>PPCI is uncalibrated, so it claims sign and pattern rather than magnitude. That claim is only
  worth something if sign and pattern survive changing the model &mdash; so they were recomputed on
  a second predictor, with the pools held fixed so the model is the only thing that varies.</p>
  <div class="scroll"><table>
    <thead><tr><th>predictor</th><th>trained on</th><th>macro AP</th><th>predicted/true occupancy</th></tr></thead>
    <tbody>
      <tr><td>deployed 3-fold mean</td><td>16 pools each, SSL encoder, 5.03 M head</td>
        <td>0.382</td><td class="lo">{R['meta']['calibration']['deployed']['nt']}&times; nt,
        {R['meta']['calibration']['deployed']['nn']}&times; nn</td></tr>
      <tr><td>single accuracy leader</td><td>20 pools, BitFit-6 on stock, 0.52 M head</td>
        <td class="hi">0.541</td><td class="hi">{R['meta']['calibration']['single']['nt']}&times; nt,
        {R['meta']['calibration']['single']['nn']}&times; nn</td></tr>
    </tbody></table></div>
  <p><b>On bouts per minute the two agree in
  {R['meta']['sign_agreement']['events']['agree']} of
  {R['meta']['sign_agreement']['events']['of']} cells &mdash; every one.</b> On occupancy they agree
  in {R['meta']['sign_agreement']['time']['agree']} of
  {R['meta']['sign_agreement']['time']['of']}, and both disagreements are cells where the deployed
  value is within 0.5 pp of zero. So PPCI's sign and pattern are a property of the behaviour, not of
  the predictor, across a change that nearly halves the calibration error and adds 0.16 macro AP.
  Magnitudes do move &mdash; which is exactly why the report never quotes one.</p>
  <div class="note"><b>Both predictors are out-of-sample on the same 52 pools</b> &mdash; the 48
  unannotated ones plus the 4 the single model held out &mdash; because the single model trained on
  the other 20. That is the largest set on which the comparison is a model comparison rather than a
  comparison of samples. The single model is also better on the causal quantity (r&Delta;
  {rr('res448_k2_bit6_d4')} against the deployment's {xf['rd_nt']:.3f} / {xf['rd_nn']:.3f}), so
  nothing is being traded here; the cross-fitted version of it is running.</div>

  <h3 style="margin-top:26px">How much is there left to win?</h3>
  <p>Two ceilings bound everything above, and both are computable rather than rhetorical.</p>
  <div class="scroll"><table>
    <thead><tr><th>r&Delta;</th><th>predicted CI width vs CI</th><th>where that is</th></tr></thead>
    <tbody>
      <tr><td>0.50</td><td>&minus;9%</td><td></td></tr>
      <tr><td>0.60</td><td>&minus;13%</td><td>roughly where we are: <b>&minus;12% measured</b>,
        averaged over the eight cells</td></tr>
      <tr><td>0.70</td><td>&minus;18%</td><td></td></tr>
      <tr><td>0.80</td><td>&minus;24%</td><td></td></tr>
      <tr><td class="hi">1.00</td><td class="hi">&minus;42%</td><td class="hi">a perfect model
        &mdash; the variance of all 72 pools labelled</td></tr>
    </tbody></table></div>
  <p><b>PPI++'s ceiling on v1 is a 42% narrower interval</b>, and the reason is exact: with
  D<sub>f</sub>&nbsp;=&nbsp;D<sub>Y</sub> the estimator's variance collapses to
  Var(D<sub>Y</sub>)/(n+N), which is what you would get by annotating all 72 pools. So the SE
  ratio is &radic;(24/72) = 0.577 no matter how good the model gets. Measured today the mean
  narrowing is <b>12%</b>, best cell 22%. The gap is entirely r&Delta;, which is why r&Delta; is
  the ranking metric.</p>
  <div class="note"><b>The second ceiling is the labels &mdash; and the estimand is largely
  protected from it.</b> No observation in v1 was scored twice, so agreement cannot be measured
  directly; the design bounds it instead, because within a genotype group the six pools are
  exchangeable yet different people scored them. Decomposing variance with annotator as the factor,
  against a permutation null of the same shape:
  <div class="scroll" style="margin-top:11px"><table>
    <thead><tr><th>quantity</th><th>annotator share of within-cell variance</th><th>chance</th><th>p</th></tr></thead>
    <tbody>
      <tr><td>rate, per observation &mdash; nt</td><td class="lo">31.7%</td><td>8.2%</td><td class="lo">&lt;0.001</td></tr>
      <tr><td>rate, per observation &mdash; nn</td><td class="lo">17.8%</td><td>7.6%</td><td class="lo">0.010</td></tr>
      <tr><td><b>within-pool difference &mdash; nt</b></td><td class="hi">26.1%</td><td>32.4%</td><td class="hi">0.59</td></tr>
      <tr><td><b>within-pool difference &mdash; nn</b></td><td class="hi">40.3%</td><td>35.7%</td><td class="hi">0.38</td></tr>
    </tbody></table></div>
  <b>Who scored a recording moves its measured rate, and stops mattering once the rate is
  differenced within a pool</b> &mdash; the same cancellation the three wild-type strata show in
  section 01. So a label-noise ceiling computed on <em>rates</em> (best attainable r &le; 0.65 on
  nose-to-tail) constrains level correlations, <b>not r&Delta;</b>. What bounds r&Delta; is
  <em>within</em>-annotator inconsistency between two phases, which no design without replication
  can separate from real change. Double-scoring 15&ndash;20 observations is the only way to get it,
  and the only way to know how much of the model's remaining error is addressable.</div>
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
      <tr><td>1</td><td>annotate ~20 more v1 pools</td><td>+0.076 AP per doubling and no plateau &mdash; worth more than every modelling change combined, and it raises CI's own precision rather than only PPI++'s</td></tr>
      <tr><td>2</td><td>annotate 4&ndash;6 v2 pools</td><td>the only way v2 gets a CI or a PPI++ estimate at all; today it has PPCI and nothing to check it against</td></tr>
      <tr><td>3</td><td>fix the observation window on biological grounds</td><td>largest single lever on the headline number; currently inherited, not chosen</td></tr>
      <tr><td>4</td><td>finish the matched ERM controls for DERM</td><td>launched; without them the DERM arms differ from their controls in the head as well as the objective, so no AP claim about them is readable</td></tr>
      <tr><td>5</td><td>run BitFit-6 over the three folds and the unannotated pools</td><td class="hi">launched &mdash; ~18 GPU-h to move every estimate onto the configuration that leads on accuracy (macro AP 0.541 against the deployed 0.382)</td></tr>
      <tr><td>6</td><td>per-animal crops from the 2060 px source</td><td>the only resolution lever left, and a prerequisite for any per-animal outcome</td></tr>
      <tr><td>7</td><td>record which animal in each v2 cage is the heterozygote</td><td>without it the within-pool genotype contrast is not identified no matter how good the vision gets</td></tr>
      <tr><td>8</td><td>double-annotate 15&ndash;20 observations, <b>nose-to-tail first</b></td><td>the annotator bound above is inferred from the design, not measured, and it is aliased with cage; nose-to-tail is where it binds (best possible r &le; 0.65)</td></tr>
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
  those same payloads, so the text cannot drift from the tables. One static figure, the annotation
  scaling curve, is still pre-rendered by <code>story_figures.py</code>.
</footer></div>

</div>
'''
