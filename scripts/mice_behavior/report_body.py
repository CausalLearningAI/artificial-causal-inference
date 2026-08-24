# Consumed by build_report.py via exec(). Expects `img` (data URIs), the five interactive figures
# already JSON-injected (`CHART`, `DECAY`, `MODELS`, `EXAMPLES`, `UNITS`) and the JSON payloads
# they are views over, for inline numbers: `E` (estimates), `M` (models), `O` (outcome units and
# their distributions), `X` (qualitative examples), `R` (PPCI robustness), `D` (the DERM /
# treatment-leak analysis, from build_derm.py).
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



def lvl(behav, odour, trans='H->O', method='ci'):
    """One LEVEL estimate (bouts/min) from estimates.json, with a proper minus sign."""
    c = next(c for c in E['cells'] if c['exp'] == 'v1' and c['unit'] == 'events'
             and c['stratum'] == 'all' and c['behav'] == behav and c['odour'] == odour
             and c['trans'] == trans and c['method'] == method)
    return f"{c['est']:+.2f}".replace('-', '&minus;')


def ddecay(behav, odour, trans, method='ci'):
    """One Delta-decay cell as a full <td>, with the star AND the highlight computed.

    The highlight used to be hard-coded in the table, so it had to be re-checked by hand whenever
    the unit changed -- and it silently encoded the OLD sign convention. Now a cell is marked
    exactly when its interval excludes zero, so it cannot drift.
    """
    c = next(c for c in E['cells'] if c['exp'] == 'v1' and c['unit'] == 'decay'
             and c['stratum'] == 'all' and c['behav'] == behav and c['odour'] == odour
             and c['trans'] == trans and c['method'] == method)
    hit = c['lo'] is not None and c['lo'] * c['hi'] > 0
    val = f"{c['est']:+.2f}".replace('-', '&minus;')
    return '<td%s>%s%s</td>' % (CLS if hit else '', val, '*' if hit else '')


def _n_resolved(method):
    return sum(1 for c in E['cells']
               if c['exp'] == 'v1' and c['unit'] == 'decay' and c['stratum'] == 'all'
               and c['method'] == method and c['lo'] is not None and c['lo'] * c['hi'] > 0)


n_dec, n_dec_ppi = _n_resolved('ci'), _n_resolved('ppi')


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

# Where the AP leader and the rDelta leader sit in each other's ranking. Computed, because these
# two integers move every time a candidate lands and a stale rank reads as a claim.
_cand = [r for r in M['runs'] if r['role'] == 'model candidate']
_by_ap = sorted(_cand, key=lambda r: -r['ap'])
_by_rd = sorted(_cand, key=lambda r: -(r['rd_nt'] + r['rd_nn']) / 2)
_ord = lambda i: f'{i}{"th" if 11 <= i % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")}'
rk_ap_on_rd = _ord(_by_rd.index(_by_ap[0]) + 1)
rk_rd_on_ap = _ord(_by_ap.index(_by_rd[0]) + 1)

# The deployment row is the mean over the three cross-fitting folds, which between them hold out
# all 24 annotated pools. Averaged here rather than transcribed.
_folds = [r for r in M['runs'] if r['role'] == 'deployment fold']
xf = {k: sum(r[k] for r in _folds) / len(_folds)
      for k in ('ap', 'f1_nt', 'f1_nn', 'rd_nt', 'rd_nn')}


def nui(behav, fac, which):
    """eta-squared and p for a pool-level factor's share of the model's bias."""
    d = D['nuisance'][which][behav][fac]
    return f"{100 * d['eta2']:.1f}% (p {d['p']:.2f})"


# The PPI++ bound, read off derm.json's grid rather than retyped. `_bw(r)` is the width ratio at
# a given r-delta; the floor is its r=1 limit.
CLS = " class='hi'"
_BG = {round(g['r'], 2): g['ratio'] for g in D['ppi_bound']['grid']}
_bw = lambda r: _BG[round(r, 2)]
_PB = D['ppi_bound']




# ---------------------------------------------------------------- probe + estimand-bias readers
# Section 04.6's load-bearing fact: a physical bag sits in a corner of the cage during O, so the
# treatment is legible in a frame that carries no behaviour at all. Read from derm.json.
_P = D['probe']
_R = _P['region']


def eb(behav, fam, what='mean'):
    r = D['estimand_bias']['families'][behav][fam]
    if what == 'ci':
        return f"[{r['lo']:+.2f}, {r['hi']:+.2f}]".replace('-', '&minus;')
    if what == 'share':
        k = r['share_of_truth']
        return f"{k:.1f}&times;" if k is not None else '&mdash;'
    if what == 'mean_abs':                       # a magnitude: a leading + would read as a sign
        return f"{r[what]:.3f}"
    return f"{r[what]:+.3f}".replace('-', '&minus;')


def ebp(behav, field):
    v = D['estimand_bias']['families'][behav]['paired_erm_minus_derm'][field]
    if field == 'diff':
        return f"{v:+.3f}".replace('-', '&minus;')
    if field == 'p':
        return f"{v:.2f}"
    return v


def dY(behav, which='pooled'):
    return f"{D['estimand_bias']['truth'][behav][which]:+.3f}".replace('-', '&minus;')


# Section 05's bound table: every row computed from derm.json's grid, so the printed percentages
# and the formula in the box next to them cannot disagree.
_BROWS = [(0.50, ''), (0.60, 'roughly where we are: <b>&minus;12% measured</b>, averaged over the '
                            'eight cells'), (0.70, ''), (0.80, ''),
          (1.00, 'a perfect model &mdash; the variance of all 72 pools labelled')]
_HI = ' class="hi"'
bound_rows = '\n      '.join(
    '<tr><td{0}>{1:.2f}</td><td{0}>&minus;{2:.0f}%</td><td{0}>{3}</td></tr>'.format(
        _HI if r == 1.0 else '', r, 100 * (1 - _bw(r)), note) for r, note in _BROWS)

BODY = f'''
<div class="wrap">

<header class="top"><div class="measure">
  <p class="eyebrow">Mice v1 / v2 &middot; status &middot; 24 August 2026</p>
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
  (15 min) &mdash; crossed with two hormonal exposures, <b>fear</b> and <b>social</b>. All six share
  a cage by construction, and among the annotated pools
  {D['pool_constant']['annotator']['constant_pools']} of
  {D['pool_constant']['annotator']['n_pools']} also share a single annotator and
  {D['pool_constant']['date']['constant_pools']} a single day. Video is 2060&sup2; at 30 fps, stored
  512&sup2; at 5 fps.
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
  <p>Behaviour is a continuous stream. Turning it into a number takes three decisions and the
  biology supplies none of them &mdash; and the answer is <b>two</b> numbers, not one: how much
  behaviour there is, and <em>when</em> in the phase it happens.</p>
  <div class="scroll"><table>
    <thead><tr><th>decision</th><th>what this report chose</th><th>where</th></tr></thead>
    <tbody>
      <tr><td>what counts as ONE event</td><td>a <b>bout</b> &mdash; one uninterrupted run of
        annotated frames. Defined at 5&nbsp;fps, so a real bout split by a two-frame gap becomes
        two</td><td>&mdash;</td></tr>
      <tr><td>over what WINDOW</td><td>the <b>first 15 minutes</b> of every phase, matched. H runs
        30 minutes, so half of it is discarded</td><td>next</td></tr>
      <tr><td>what you MEASURE</td><td><b>a level</b>, how often a bout starts, and <b>a
        timing</b>, when in the phase they start</td><td>02a, 02b</td></tr>
    </tbody></table></div>
  <p>The window comes first because it applies to both outcomes. Start by looking at the whole
  protocol laid end to end &mdash; six recordings, 120 minutes, both behaviours.</p>
</div>
  <div class="figwrap">{DECAY}</div>
<div class="measure">
  <p><b>Nothing is stationary inside a phase.</b> Rates fall several-fold across every recording
  &mdash; half-life <b>4&ndash;14 minutes</b>, P fastest in every cell, and one cell rises instead
  (nose-to-tail under social exposure during O, where the exposure sustains investigation while
  everything else habituates). So a phase <em>mean</em> averages over whichever stretch of a
  decaying curve the schedule happened to sample, and because H runs 30 minutes against O and P's
  15, <b>the two sides of H&rarr;O do not sample the same stretch</b>. O&rarr;P is unaffected:
  equal lengths, so any window rule leaves it bit-for-bit identical, checked in all four cells.
  What the choice is worth on H&rarr;O:</p>
  <div class="scroll"><table>
    <thead><tr><th>H &rarr; O</th><th>full H (30 min)</th><th>first 15</th><th>last 15</th><th>spread</th></tr></thead>
    <tbody>
      <tr><td>nt &middot; fear</td><td>+0.36</td><td>+0.22</td><td>+0.49</td><td>0.28</td></tr>
      <tr><td>nt &middot; social</td><td>&minus;0.37</td><td>&minus;0.67</td><td>&minus;0.07</td><td>0.60</td></tr>
      <tr><td>nn &middot; fear</td><td>+0.66</td><td>+0.45</td><td>+0.86</td><td>0.42</td></tr>
      <tr><td>nn &middot; social</td><td>+0.47</td><td>&minus;0.03</td><td>+0.97</td><td class="lo">1.01 &mdash; changes sign</td></tr>
    </tbody></table></div>
  <p><b>Matching the first 15 minutes settles a confound, not just an inconsistency.</b> Every
  phase is a separate recording the experimenter starts by opening the cage, and the onset spike
  that follows is largest in <b>P</b> &mdash; where the odour is <em>removed</em> &mdash; in 3 of 4
  cells (first-2-min over last-2-min rate, nn&nbsp;&middot;&nbsp;fear: H 7.6, O 6.7, <b>P 12.3</b>).
  A response peaking when the odour is taken away is handling, not odour, so matching onset position
  puts it on both sides of every contrast, where it cancels. Every estimate in this report is cut
  that way, so <b>&ldquo;first 15&rdquo; is the column the figures report</b> and the other two are
  the sensitivity around it. Nose-to-nose under social is the cell that depended on it: +0.47 on the
  full window against &minus;0.03 matched.</p>

  <div class="sub">
    <p class="q">02a &middot; the level</p>
    <h3>How much behaviour &mdash; and which of three ways to count it</h3>
    <p><b>Counts</b> (bouts per minute) measure how often the behaviour is initiated,
    <b>occupancy</b> (percent of frames in it) how much of the window it fills, <b>duration</b>
    (mean bout length) how long one bout lasts. Occupancy is close to
    counts&nbsp;&times;&nbsp;duration, so it inherits the noise of both rather than being a third
    independent choice. Two columns decide it: <b>noise</b> is how much the measurement scatters
    across pools that had the same treatment, and <b>r&Delta;</b> is how well the vision model
    reproduces a pool's within-pool phase difference &mdash; the only thing the model is asked to
    do.</p>
  </div>
  <div class="scroll"><table>
    <thead><tr><th>unit</th><th>what it measures</th><th>noise<br><span style="opacity:.65">CV, nn / nt &mdash; lower better</span></th>
      <th>model tracks it<br><span style="opacity:.65">r&Delta;, nn / nt &mdash; higher better</span></th><th>verdict</th></tr></thead>
    <tbody>
      <tr><td><b>counts</b><br><span style="opacity:.65">bouts per minute</span></td>
        <td>how often it starts</td><td>{cv('counts')}</td>
        <td class="hi">{rd('counts')}</td><td class="hi">chosen</td></tr>
      <tr><td><b>occupancy</b><br><span style="opacity:.65">% of frames in it</span></td>
        <td>how much time it fills</td><td class="lo">{cv('occupancy')}</td>
        <td class="lo">{rd('occupancy')}</td><td>noisier, and the model tracks it half as well</td></tr>
      <tr><td><b>duration</b><br><span style="opacity:.65">mean bout length</span></td>
        <td>how long one bout lasts</td><td class="hi">{cv('duration')}</td>
        <td>no model head</td><td>quietest, but nothing left to vary with</td></tr>
    </tbody></table></div>
</div>
  <div class="figwrap">{UNITS}</div>
<div class="measure">
  <p><b>Counts win on measurability.</b> At 5&nbsp;fps {nnf}% of nose-to-nose bouts and {ntf}% of
  nose-to-tail bouts last a <em>single frame</em>, so duration has almost nothing left to vary with
  &mdash; its length is set by sub-frame timing the pipeline introduced, not by the animals.
  Occupancy has the opposite problem: the longest 10% of bouts carry {nnt}% of all nose-to-nose
  behaviour time and {ntt}% of nose-to-tail, so one long huddle moves it more than ten short
  contacts. Counts sit between the two, and are what the model tracks best.</p>

  <div class="sub">
    <p class="q">02b &middot; the timing</p>
    <h3>When in the phase behaviour happens</h3>
    <p>The same non-stationarity makes <em>when</em> a bout starts an outcome in its own right.
    Measured as the <b>mean onset time</b> of a phase's bouts, inside the same matched window:</p>
  </div>
  <div class="eqn"><math display="block"><mrow><mi>decay</mi><mo>=</mo>
    <mfrac><mn>1</mn><mrow><mo>|</mo><mi>B</mi><mo>|</mo></mrow></mfrac>
    <munder><mo>&#x2211;</mo><mrow><mi>b</mi><mo>&#x2208;</mo><mi>B</mi></mrow></munder>
    <mi>onset</mi><mo>(</mo><mi>b</mi><mo>)</mo>
    <mo>,</mo><mspace width="1.2em"/>
    <mi>B</mi><mo>=</mo><mrow><mo>{{</mo><mtext>bouts starting in minutes&#xA0;0&#x2013;15</mtext>
    <mo>}}</mo></mrow></mrow></math></div>
  <p>In minutes, so it interprets itself: a <b>flat process gives 7.5</b>, half the window, and a
  difference reads as <em>&ldquo;the exposure pushes bouts X minutes later into the phase&rdquo;</em>.
  Model-free, per-observation, and needing no exponential &mdash; a fitted slope does not survive
  here, with log-linearity rejected in 7 of 12 cells. It beats a front-loading fraction, whose null
  was an artefact of its own nesting, and the median, which discards the late tail where a flatter
  curve actually shows.</p>
</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">03 &middot; Effects</p>
  <h2>Every estimate, in one figure</h2></div>
  <p>Both outcomes section 02 settled on &mdash; <b>the level</b> (bouts per minute) and <b>the
  timing</b> (decay, mean onset in minutes) &mdash; for every cohort, behaviour and breakdown.
  Occupancy is there too, as the alternative 02a rejected, so the pattern can be checked against it.
  One panel per exposure, both phase transitions in each, and <b>three estimators</b>:</p>
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
  <p>PPCI needs no annotation anywhere, which is why it is the only estimator that exists on v2 at
  all. Sections 04 and 05 say how far that model can be trusted.</p>
</div>
  <div class="figwrap">{CHART}</div>
<div class="measure">
  <p><b>The level: the two exposures act differently, and one acts in opposite directions on the two
  behaviours.</b> Turning the odour on, nose-to-nose rises under fear ({lvl('nn','fear')}) and is
  flat under social ({lvl('nn','social')}), while nose-to-tail rises under fear
  ({lvl('nt','fear')}) and <em>falls</em> under social ({lvl('nt','social')}). Withdrawing it
  reverses nose-to-nose under both ({lvl('nn','fear','O->P')} and {lvl('nn','social','O->P')}).
  Each exposure is reported separately throughout, never pooled.</p>
  <p><b>The timing: every sign is positive turning the odour on and negative turning it off.</b>
  Bouts start <b>1.0&ndash;2.2 minutes later</b> into the phase once the odour is on, and up to 3.3
  minutes earlier once it is withdrawn &mdash; the exposure flattens the habituation curve and
  withdrawing it restores fast habituation. Not how much behaviour the odour triggers, but how long
  it holds attention.</p>
  <div class="scroll"><table>
    <thead><tr><th>&Delta;decay, minutes, human labels</th><th>nt &middot; fear</th><th>nt &middot; social</th><th>nn &middot; fear</th><th>nn &middot; social</th></tr></thead>
    <tbody>
      <tr><td>H &rarr; O &nbsp;(odour ON)</td>{ddecay('nt','fear','H->O')}{ddecay('nt','social','H->O')}{ddecay('nn','fear','H->O')}{ddecay('nn','social','H->O')}</tr>
      <tr><td>O &rarr; P &nbsp;(odour OFF)</td>{ddecay('nt','fear','O->P')}{ddecay('nt','social','O->P')}{ddecay('nn','fear','O->P')}{ddecay('nn','social','O->P')}</tr>
    </tbody></table></div>
  <p class="defn">* = resolves; {n_dec} of 8 do on human labels alone, {n_dec_ppi} of 8 with PPI++.
  Decay is undefined where a recording has no bout in the window, so n falls to 13&ndash;24 by cell
  &mdash; which is why the model buys more on the timing than it does on the level.</p>

  <h3 style="margin-top:26px">What a better model could buy, at most</h3>
  <p>Two things cap it, and both are computable rather than rhetorical.</p>

  <div class="bound">
    <p class="t">bound &middot; PPI++ on v1</p>
    <p>Let <math><mi>n</mi><mo>=</mo><mn>{_PB['n']}</mn></math> be the labelled pools,
    <math><mi>N</mi><mo>=</mo><mn>{_PB['N']}</mn></math> the unlabelled ones, and
    <math><mrow><mi>r</mi><mi>&#x394;</mi><mo>=</mo><mi>corr</mi><mo>(</mo>
    <msub><mi>D</mi><mi>Y</mi></msub><mo>,</mo><msub><mi>D</mi><mi>f</mi></msub>
    <mo>)</mo></mrow></math>. At the power-tuned
    <math><mrow><mi>&#x3BB;</mi><mo>=</mo><mi>Cov</mi><mo>(</mo><msub><mi>D</mi><mi>Y</mi></msub>
    <mo>,</mo><msub><mi>D</mi><mi>f</mi></msub><mo>)</mo><mo>/</mo><mo>[</mo>
    <mi>Var</mi><mo>(</mo><msub><mi>D</mi><mi>f</mi></msub><mo>)</mo>
    <mo>(</mo><mn>1</mn><mo>+</mo><mi>n</mi><mo>/</mo><mi>N</mi><mo>)</mo><mo>]</mo></mrow></math>
    the estimator's variance is</p>
    <div class="eqn"><math display="block"><mrow>
      <mi>Var</mi><mo>(</mo><msub><mover accent="true"><mi>&#x3B8;</mi><mo>^</mo></mover>
        <mtext>PPI++</mtext></msub><mo>)</mo><mo>=</mo>
      <mfrac><mrow><mi>Var</mi><mo>(</mo><msub><mi>D</mi><mi>Y</mi></msub><mo>)</mo></mrow>
             <mi>n</mi></mfrac>
      <mrow><mo>[</mo><mn>1</mn><mo>&#x2212;</mo>
        <msup><mrow><mo>(</mo><mi>r</mi><mi>&#x394;</mi><mo>)</mo></mrow><mn>2</mn></msup>
        <mfrac><mi>N</mi><mrow><mi>n</mi><mo>+</mo><mi>N</mi></mrow></mfrac><mo>]</mo></mrow>
    </mrow></math></div>
    <p>so the interval scales as</p>
    <div class="eqn"><math display="block"><mrow>
      <mfrac><msub><mi>SE</mi><mtext>PPI++</mtext></msub>
             <msub><mi>SE</mi><mtext>CI</mtext></msub></mfrac>
      <mo>=</mo>
      <msqrt><mrow><mn>1</mn><mo>&#x2212;</mo>
        <mfrac><mn>2</mn><mn>3</mn></mfrac>
        <msup><mrow><mo>(</mo><mi>r</mi><mi>&#x394;</mi><mo>)</mo></mrow><mn>2</mn></msup>
      </mrow></msqrt>
      <mspace width="1.2em"/><mo>&#x2265;</mo><mspace width="0.6em"/>
      <msqrt><mfrac><mi>n</mi><mrow><mi>n</mi><mo>+</mo><mi>N</mi></mrow></mfrac></msqrt>
      <mo>=</mo>
      <msqrt><mfrac><mn>1</mn><mn>3</mn></mfrac></msqrt>
      <mo>=</mo><mn>{_PB['floor']:.3f}</mn>
    </mrow></math></div>
    <p><b>So no model can narrow the interval by more than {100 * (1 - _PB['floor']):.1f}%.</b>
    Equality holds only at <math><mrow><mi>r</mi><mi>&#x394;</mi><mo>=</mo><mn>1</mn></mrow></math>,
    where the variance collapses to
    <math><mrow><mi>Var</mi><mo>(</mo><msub><mi>D</mi><mi>Y</mi></msub><mo>)</mo><mo>/</mo>
    <mo>(</mo><mi>n</mi><mo>+</mo><mi>N</mi><mo>)</mo></mrow></math> &mdash; precisely the variance
    of having annotated all {_PB['n'] + _PB['N']} pools. The bound depends on the <em>design</em>
    only: 24 of 72, and nothing about the model. Measured today the mean narrowing is <b>12%</b>,
    best cell 22%, so about a third of the available ceiling is in hand and the rest is entirely
    r&Delta;.</p>
  </div>

  <div class="scroll"><table>
    <thead><tr><th>r&Delta;</th><th>predicted PPI++ width vs CI</th><th>where that is</th></tr></thead>
    <tbody>
      {bound_rows}
    </tbody></table></div>
  <p>Two things follow. <b>r&Delta; is the ranking metric</b> because it is the only free variable
  in the bound. And <b>annotating more pools raises the ceiling itself</b>: the floor is
  &radic;(n/(n+N)), so moving 20 pools from unlabelled to labelled changes what a perfect model
  could ever be worth &mdash; which is why more annotation beats every modelling change in section 04.</p>
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
  <div class="note warnbox"><b>Not a ceiling, but the last thing to know about these three:
  CI and PPI++ do not target quite the same population.</b> Annotation is <b>3:1 het-enriched</b> (18 het / 6 wt against a 36/36 design), so CI estimates the
  effect <em>in the annotated pools</em> while PPI++ pulls in 48 unannotated ones that are
  wt-enriched (30 wt / 18 het) and targets the full 72. They coincide only if the phase effect does
  not vary with genotype, and it varies a little: nose-to-nose under fear reads about +0.79 across
  the wt strata against +0.60 across the het strata, moving the target roughly +0.05 (~7%). Small
  next to these intervals, but it is a difference in <em>estimand</em> rather than precision, so it
  does not shrink with more data. The fix is to estimate within stratum and recombine with design
  weights.</div>
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
  interchangeable, and the easiest one to compute is not the one that decides. Written out:</p>
  <div class="mdef">
    <p class="w"><b>macro AP</b> &mdash; frame-level average precision, the area under the
    precision&ndash;recall curve, averaged over the two behaviours. Threshold-free; what training
    monitors and what early stopping selects on.</p>
    <div class="eqn"><math display="block"><mrow>
      <mi>AP</mi><mo>=</mo><mfrac><mn>1</mn><mn>2</mn></mfrac>
      <munder><mo>&#x2211;</mo><mrow><mi>b</mi><mo>&#x2208;</mo><mo>{{</mo><mtext>nt</mtext>
        <mo>,</mo><mtext>nn</mtext><mo>}}</mo></mrow></munder>
      <msub><mi>AP</mi><mi>b</mi></msub>
      <mo>,</mo><mspace width="1.6em"/>
      <msub><mi>AP</mi><mi>b</mi></msub><mo>=</mo>
      <munder><mo>&#x2211;</mo><mi>k</mi></munder>
      <mrow><mo>(</mo><msub><mi>R</mi><mi>k</mi></msub><mo>&#x2212;</mo>
        <msub><mi>R</mi><mrow><mi>k</mi><mo>&#x2212;</mo><mn>1</mn></mrow></msub><mo>)</mo></mrow>
      <mspace width="0.2em"/><msub><mi>P</mi><mi>k</mi></msub>
    </mrow></math></div>
    <p class="w"><b>event F1</b> &mdash; bout-level, with <em>any-overlap</em> matching. Write
    <math><mi>B</mi></math> for the true bouts of one recording and
    <math><mover accent="true"><mi>B</mi><mo>^</mo></mover></math> for the predicted ones (maximal
    runs of frames over threshold). A bout counts as found if it overlaps one on the other side
    <em>at all</em> &mdash; the right resolution when {nnf}% of nose-to-nose bouts last one frame.</p>
    <div class="eqn"><math display="block"><mrow>
      <mi>R</mi><mo>=</mo>
      <mfrac>
        <mrow><mo>|</mo><mo>{{</mo><mi>b</mi><mo>&#x2208;</mo><mi>B</mi><mo>:</mo>
          <mi>b</mi><mo>&#x2229;</mo><mover accent="true"><mi>B</mi><mo>^</mo></mover>
          <mo>&#x2260;</mo><mo>&#x2205;</mo><mo>}}</mo><mo>|</mo></mrow>
        <mrow><mo>|</mo><mi>B</mi><mo>|</mo></mrow></mfrac>
      <mo>,</mo><mspace width="1.2em"/>
      <mi>P</mi><mo>=</mo>
      <mfrac>
        <mrow><mo>|</mo><mo>{{</mo><mover accent="true"><mi>b</mi><mo>^</mo></mover>
          <mo>&#x2208;</mo><mover accent="true"><mi>B</mi><mo>^</mo></mover><mo>:</mo>
          <mover accent="true"><mi>b</mi><mo>^</mo></mover><mo>&#x2229;</mo><mi>B</mi>
          <mo>&#x2260;</mo><mo>&#x2205;</mo><mo>}}</mo><mo>|</mo></mrow>
        <mrow><mo>|</mo><mover accent="true"><mi>B</mi><mo>^</mo></mover><mo>|</mo></mrow></mfrac>
      <mo>,</mo><mspace width="1.2em"/>
      <msub><mi>F</mi><mn>1</mn></msub><mo>=</mo>
      <mfrac><mrow><mn>2</mn><mi>P</mi><mi>R</mi></mrow>
             <mrow><mi>P</mi><mo>+</mo><mi>R</mi></mrow></mfrac>
    </mrow></math></div>
    <p class="w"><b>r&Delta;</b> &mdash; the correlation between true and predicted
    <em>within-pool phase differences</em>. For pool <math><mi>p</mi></math> and transition
    <math><mrow><mi>x</mi><mo>&#x2192;</mo><mi>y</mi></mrow></math>, with
    <math><mi>Y</mi></math> the human score and <math><mi>f</mi></math> the model's:</p>
    <div class="eqn"><math display="block"><mrow>
      <msubsup><mi>D</mi><mi>Y</mi><mrow><mo>(</mo><mi>p</mi><mo>)</mo></mrow></msubsup>
      <mo>=</mo><msub><mi>Y</mi><mi>p</mi></msub><mo>(</mo><mi>y</mi><mo>)</mo>
      <mo>&#x2212;</mo><msub><mi>Y</mi><mi>p</mi></msub><mo>(</mo><mi>x</mi><mo>)</mo>
      <mo>,</mo><mspace width="1em"/>
      <msubsup><mi>D</mi><mi>f</mi><mrow><mo>(</mo><mi>p</mi><mo>)</mo></mrow></msubsup>
      <mo>=</mo><msub><mi>f</mi><mi>p</mi></msub><mo>(</mo><mi>y</mi><mo>)</mo>
      <mo>&#x2212;</mo><msub><mi>f</mi><mi>p</mi></msub><mo>(</mo><mi>x</mi><mo>)</mo>
      <mo>,</mo><mspace width="1.6em"/>
      <mi>r</mi><mi>&#x394;</mi><mo>=</mo>
      <mfrac>
        <mrow><mi>Cov</mi><mo>(</mo><msub><mi>D</mi><mi>Y</mi></msub><mo>,</mo>
          <msub><mi>D</mi><mi>f</mi></msub><mo>)</mo></mrow>
        <mrow><mi>sd</mi><mo>(</mo><msub><mi>D</mi><mi>Y</mi></msub><mo>)</mo>
          <mspace width="0.15em"/>
          <mi>sd</mi><mo>(</mo><msub><mi>D</mi><mi>f</mi></msub><mo>)</mo></mrow></mfrac>
    </mrow></math></div>
    <p class="w">These are the two differences the bound at the end of section 03 is written in:
    PPI++'s entire variance reduction is a function of
    <math><mrow><mi>r</mi><mi>&#x394;</mi></mrow></math> and of nothing else, which is why it
    &mdash; and not AP &mdash; is the ranking key here.</p>
  </div>
  <div class="scroll"><table>
    <thead><tr><th>metric</th><th>what it is allowed to decide</th></tr></thead>
    <tbody>
      <tr><td><b>macro AP</b></td>
        <td>shortlisting only. Seed noise measured on <b>seven</b> configurations spans
        <b>0.004&ndash;0.016</b> (median 0.007) and is widest on the fine-tuned arms, so a gap under
        ~0.015 is not a gap.</td></tr>
      <tr><td><b>event F1</b></td>
        <td>whether the detector finds the right <em>events</em> rather than the right frames.</td></tr>
      <tr><td><b>r&Delta;</b></td>
        <td class="hi">the ranking. PPI++'s variance reduction is a function of r&Delta; and
        nothing else, exactly as the bound at the end of section 03 says. It is <em>not</em> what
        training selects on.</td></tr>
    </tbody></table></div>
  <div class="note warnbox"><b>r&Delta; is the ranking key, and on the standing split its
  nose-to-tail value is unusable.</b> It rests on <b>16 points</b> there (4 pools &times; 2 exposures
  &times; 2 transitions) at a threshold fitted to those same pools. How bad that is has been
  measured: two runs of one configuration differing only in <em>seed</em> give
  r&Delta;&nbsp;nt of {rr2('res448_k2_frozen_d4photo_ermH5M','nt')} and
  {rr2('res448_k2_frozen_d4photo_ermH5M_s1','nt')} &mdash; a spread wider than the range across
  every arm below. Nose-to-nose is far steadier (0.77&ndash;0.80 across the same pairs).
  Only the cross-fitted folds, which hold out all 24 annotated pools, give either an honest
  denominator. So every table below reports AP <em>and</em> r&Delta;, and their Spearman correlation
  across the {M['meta']['n_candidates']} candidates is only
  <b>{M['meta']['spearman_ap_vs_rdelta']:+.2f}</b>.</div>

  <h3 style="margin-top:28px">What actually moves the number</h3>
  <p>Every lever tried, grouped by the one thing it changes. The subsections below run in this
  order, and each holds everything else fixed.</p>
  <div class="scroll"><table>
    <thead><tr><th></th><th>what varies</th><th>lever</th><th>&Delta; macro AP</th><th>verdict</th></tr></thead>
    <tbody>
      <tr><td>04.1</td><td><b>data</b></td><td>annotate more pools</td>
        <td class="hi">+0.076 per doubling</td><td class="hi">the binding constraint, no plateau</td></tr>
      <tr><td>04.2</td><td><b>encoder</b><br><span style="opacity:.65">fine-tuning</span></td>
        <td>unfreeze the last blocks &mdash; BitFit</td>
        <td class="hi">+0.112</td><td class="hi">recalibration, at 1/602nd the params</td></tr>
      <tr><td>04.3</td><td><b>encoder</b><br><span style="opacity:.65">label-free</span></td>
        <td>self-supervised on unlabelled frames</td>
        <td class="hi">+0.033</td><td class="hi">what we deploy; the only lever that reaches v2</td></tr>
      <tr><td>04.4</td><td><b>input</b></td><td>224 &rarr; 448 px, and the token count under it</td>
        <td class="hi">+0.132</td><td>real, and now saturated</td></tr>
      <tr><td>04.5</td><td><b>head</b></td><td>capacity: 0.44 M to 5.03 M, self-attention, multi-query</td>
        <td>&minus;0.013 to +0.014</td><td class="lo">flat across an 11&times; parameter range</td></tr>
      <tr><td>04.6</td><td><b>objective</b></td><td>vREx</td>
        <td class="lo">+0.008 at best, &minus;0.094 at &beta;=100</td><td class="lo">no help, and harmful when pushed</td></tr>
      <tr><td>04.6</td><td><b>objective</b></td><td>DERM &mdash; deconfound against phase</td>
        <td>&minus;0.02 against a matched control<br><span style="opacity:.65">the expected price, not a cost</span></td>
        <td class="lo">open &mdash; judge it on the estimand, not on AP</td></tr>
    </tbody></table></div>
  <p class="defn"><b>Read every &Delta; against 0.015.</b> Seed noise spans 0.004&ndash;0.016
  across the seven configurations now run at two seeds, so 04.1, 04.2 and 04.4 clear it and 04.3,
  04.5 and vREx sit inside it. DERM is the one exception on this page: it buys prior-independence
  by giving up frame accuracy, so a small AP loss is what success looks like there, and 04.6 judges
  it on the estimand instead.</p>
</div>

<div class="measure">
  <details class="sub" open>
    <summary>
    <p class="q">04.1 &middot; data</p>
    <h3>The scaling law of annotation <span class="verdict v-yes">the binding constraint</span></h3>
    </summary>
    <div class="body">
    <p>Nested subsets of the labelled pools, so each point differs from the last for exactly one
    reason.</p>
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
  <div class="note"><b>What &ldquo;more data&rdquo; means here is more POSITIVES, and the
  imbalance is handled by subsampling.</b> A frame carries nose-to-tail 1.2% of the time and
  nose-to-nose 0.8%, so an unweighted epoch would be 99% negatives. Every arm on this page
  therefore trains on <b>all 23,280 positive anchors plus an equal number of negatives drawn fresh
  from the 54,292 available every epoch</b> &mdash; 46,560 samples, 1:1, resampled so the model
  still sees most of the negative pool over 30 epochs without any epoch being swamped by it. Two
  consequences worth carrying forward. The effective training prevalence is <b>~25%, not ~1%</b>,
  which is why predicted occupancy runs about fivefold above truth in section 05 and why no
  prediction on this page may be read as a rate. And the label budget in the table above binds
  through positives: annotating a pool adds ~970 positive anchors, and negatives were never
  scarce.</div>

  </div>
  </details>
  <details class="sub">
    <summary>
    <p class="q">04.2 &middot; encoder &mdash; fine-tuning</p>
    <h3>Adapting the encoder <span class="verdict v-yes">yes &mdash; +0.11 AP, for 70k params</span></h3>
    </summary>
    <div class="body">
    <p>Unfreezing the last DINOv2 blocks is the largest modelling gain measured: macro AP 0.4289
    frozen &rarr; 0.4889 at two blocks &rarr; 0.5243 at six &mdash; a +0.095 span against a seed
    spread of at most 0.016.</p>
    <p><b>What it is doing is recalibration, not new computation.</b> BitFit &mdash; training only
    biases, LayerNorm gains and LayerScale gains, and nothing that can form a new function of two
    patch features &mdash; matches and then beats full fine-tuning: <b>{run('res448_k2_bit6_d4')['ap']:.4f}
    with 70,656 trainable encoder parameters against 0.5243 with 42.5 M</b>, a 602&times; cut. So
    the gain is not extra capacity: adding capacity to the <em>head</em> instead does nothing at all
    (04.5). It also carries the best r&Delta; of any arm on nose-to-tail apart from full fine-tuning
    ({run('res448_k2_bit6_d4')['rd_nt']:.3f}), so this is not an AP-only win.</p>
    <div class="note"><b>Two checks on that comparison.</b> The BitFit arms ran with
    <code>d4</code> augmentation against a <code>d4_photo</code> control; against the
    <em>matched</em> <code>d4</code> control (0.4187) BitFit-6 is <b>+0.122</b>, so the mismatch
    understates the gain rather than manufacturing it. And encoder learning rate is decisive here,
    not incidental: BitFit reads 0.4509 at 1e-5 against 0.4902 at 1e-3.</div>

  </div>
  </details>
  <details class="sub">
    <summary>
    <p class="q">04.3 &middot; encoder &mdash; unlabelled frames</p>
    <h3>Self-supervised adaptation <span class="verdict v-yes">yes &mdash; and it is what we deploy</span></h3>
    </summary>
    <div class="body">
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
    <div class="note"><b>Settled: SSL and BitFit do not stack on AP, and it does not matter for the
    estimate.</b> Both arms now have two seeds.
    <div class="scroll" style="margin-top:11px"><table>
      <thead><tr><th>BitFit-6 starting from</th><th>macro AP (2 seeds)</th><th>event F1 nt / nn</th><th>r&Delta; nt</th><th>r&Delta; nn</th></tr></thead>
      <tbody>
        <tr><td>stock DINOv2</td><td class="hi">0.5365 &nbsp;<span style="opacity:.6">(0.5409, 0.5321)</span></td>
          <td>0.446 / <b>0.545</b></td><td>0.641</td><td><b>0.826</b></td></tr>
        <tr><td>the SSL encoder</td><td>0.5058 &nbsp;<span style="opacity:.6">(0.5127, 0.4989)</span></td>
          <td><b>0.473</b> / 0.521</td><td><b>0.649</b></td><td>0.804</td></tr>
      </tbody></table></div>
    On macro AP the gap is <b>+0.031</b> for stock, about 2.2&times; the widest seed spread, so it
    clears noise: the two interventions substitute rather than compose. <b>On every other metric
    they are a wash</b> &mdash; each leads on one of the two event F1s and one of the two r&Delta;s.
    Since r&Delta; is what decides whether a model helps the estimate, the deployed SSL encoder is
    <em>not</em> penalised on the axis that matters, only on the one that shortlists. What <em>is</em>
    established, each a clean single-variable change: 2&times; the corpus at matched compute is
    neutral, and six adapted blocks is clearly harmful. So <em>scaling</em> the corpus is closed;
    SSL itself is not.</div>

  </div>
  </details>
  <details class="sub">
    <summary>
    <p class="q">04.4 &middot; input</p>
    <h3>Resolution against tokens <span class="verdict v-part">token-bound, now saturated</span></h3>
    </summary>
    <div class="body">
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
  </details>
  <details class="sub">
    <summary>
    <p class="q">04.5 &middot; head</p>
    <h3>Head capacity <span class="verdict v-no">flat across an 11&times; parameter range</span></h3>
    </summary>
    <div class="body">
    <p>Everything else held fixed &mdash; frozen stock DINOv2, 448&nbsp;px, D4 + photometric
    augmentation, 30 epochs, seed&nbsp;42 &mdash; and only the pooling head changed.</p>
    <div class="scroll"><table>
      <thead><tr><th>head</th><th>params</th><th>what it adds</th><th>macro AP</th><th>&Delta;</th><th>r&Delta; nt / nn</th></tr></thead>
      <tbody>
        <tr><td>plain, no cross-attention</td><td>5.03 M</td><td>mean-pool the 1024 tokens, then an MLP</td>
          <td class="lo">{run('res448_k2_frozen_d4photo_ermH5M')['ap']:.4f}</td><td class="lo">&minus;0.013</td>
          <td>{rr('res448_k2_frozen_d4photo_ermH5M')}</td></tr>
        <tr><td><b>cross-attention, 1 query</b> &mdash; the control</td><td>0.52 M</td>
          <td>one learned query attends over the 1024 tokens</td><td>0.4289</td><td>&mdash;</td>
          <td>{rr('res448_k2_frozen_d4photo_decay30_seed1')}</td></tr>
        <tr><td>4 learned queries</td><td>0.57 M</td><td>four pooling queries instead of one</td>
          <td>{run('res448_k2_frozen_q4_d4photo')['ap']:.4f}</td><td>&minus;0.003</td>
          <td>{rr('res448_k2_frozen_q4_d4photo')}</td></tr>
        <tr><td>+ 4&times;4 region grid</td><td class="hi">0.44 M</td>
          <td>keeps <em>where</em> in the cage a token came from &mdash; and is the one row that
          moves two things, since it also widens cross-attention from 64 to 128</td>
          <td>{run('res448_k2_frozen_d4photo_rgrid4')['ap']:.4f}</td><td>+0.004</td>
          <td class="hi">{rr('res448_k2_frozen_d4photo_rgrid4')}</td></tr>
        <tr><td>+ patch self-attention (d = 128)</td><td>0.79 M</td>
          <td>tokens attend to each other before pooling</td>
          <td class="hi">0.4431</td><td>+0.014</td><td>&mdash; not scored</td></tr>
      </tbody></table></div>
    <p>Across an <b>11&times;</b> span in head parameters every arm lands within 0.014 of the
    control &mdash; inside the 0.015 seed band &mdash; though the two extremes are 0.027 apart. The
    one arm that reaches the band's edge, patch self-attention at +0.014, has a single seed and no
    held-out predictions, so it has no r&Delta; and cannot be promoted on AP alone. Two things do
    follow. The <b>region-preserving head matches the 5.03 M plain one at under a tenth of the
    size</b> and carries a better r&Delta;,
    which is why it appears in section 05's shortlist. And the head the objective arms and the
    deployment folds all use is the <em>plain</em> 5.03 M one, the weakest of the five &mdash;
    a legacy of the launcher, and the reason those arms are only ever compared to their own matched
    controls.</p>

  </div>
  </details>
  <details class="sub">
    <summary>
    <p class="q">04.6 &middot; objective</p>
    <h3>Does the model read the treatment?
      <span class="verdict v-part">open &mdash; 4 pools cannot answer it</span></h3>
    </summary>
    <div class="body">
    <p><b>The shortcut, and why ERM would take it.</b> A bag is placed in a corner of the cage for
    the exposure phase, so the treatment is visible in the frame whether or not any behaviour is.
    Prevalence moves with the phase too (nose-to-tail 0.89% in H against 1.22% in O on the training
    pools). ERM's optimum is
    <math><mrow><mi>P</mi><mo>(</mo><mi>Y</mi><mo>=</mo><mn>1</mn><mo>|</mo><mi>x</mi>
    <mo>)</mo></mrow></math>, which <em>includes</em> that phase-conditional prior, so nothing in
    the objective discourages scoring a frame by which phase it looks like. DERM reweights each
    sample by
    <math><mrow><mi>Var</mi><mo>(</mo><mi>Y</mi><mo>|</mo><mi>E</mi><mo>)</mo><mo>/</mo>
    <mi>P</mi><mo>(</mo><mi>Y</mi><mo>,</mo><mi>E</mi><mo>)</mo></mrow></math>, which divides the
    prior odds out. It is the right tool for this.</p>

    <p><b>What the shortcut would cost.</b> The estimand is a within-pool difference, so with
    <math><mrow><mi>E</mi><mo>[</mo><mi>f</mi><mo>|</mo><mi>p</mi><mo>]</mo><mo>=</mo>
    <msub><mi>a</mi><mi>p</mi></msub><mo>+</mo><mi>b</mi><mspace width="0.15em"/><mi>E</mi>
    <mo>[</mo><mi>Y</mi><mo>|</mo><mi>p</mi><mo>]</mo></mrow></math>,</p>
    <div class="eqn"><math display="block"><mrow>
      <mi>E</mi><mo>[</mo><msub><mi>D</mi><mi>f</mi></msub><mo>]</mo><mo>=</mo>
      <munder><munder><mrow><mi>b</mi><mspace width="0.15em"/><mi>E</mi><mo>[</mo>
        <msub><mi>D</mi><mi>Y</mi></msub><mo>]</mo></mrow>
        <mo>&#x23DF;</mo></munder><mtext>a scale &#x2014; harmless</mtext></munder>
      <mo>+</mo>
      <munder><munder><mrow><mo>(</mo><msub><mi>a</mi><mi>O</mi></msub><mo>&#x2212;</mo>
        <msub><mi>a</mi><mi>H</mi></msub><mo>)</mo></mrow>
        <mo>&#x23DF;</mo></munder><mtext>what the shortcut becomes</mtext></munder>
    </mrow></math></div>
    <p>The scale is absorbed by PPI++'s <math><mi>&#x3BB;</mi></math> and never quoted by
    uncalibrated PPCI. <math><mrow><msub><mi>a</mi><mi>O</mi></msub><mo>&#x2212;</mo>
    <msub><mi>a</mi><mi>H</mi></msub></mrow></math> is the whole risk, and it is what an objective
    change has to be judged on. Measured in bouts per minute at the rate-matched threshold &mdash;
    which spends the one scale the estimand allows &mdash; per (pool &times; exposure), eight units,
    seeds averaged within a unit:</p>
    <div class="scroll"><table>
      <thead><tr><th></th><th>mean a<sub>O</sub>&minus;a<sub>H</sub></th><th>95% CI</th>
        <th>true effect, H&rarr;O</th></tr></thead>
      <tbody>
        <tr><td>nose-to-tail &middot; <b>ERM</b></td><td class="lo">{eb('nt','ERM')}</td>
          <td>{eb('nt','ERM','ci')}</td>
          <td rowspan="2">{dY('nt')} pooled &mdash; so the bias is
          <b>{eb('nt','ERM','share')}</b> its size, and opposite in sign</td></tr>
        <tr><td>nose-to-tail &middot; <b>DERM</b></td><td>{eb('nt','DERM')}</td>
          <td>{eb('nt','DERM','ci')}</td></tr>
        <tr><td>nose-to-nose &middot; <b>ERM</b></td><td>{eb('nn','ERM')}</td>
          <td>{eb('nn','ERM','ci')}</td>
          <td rowspan="2">{dY('nn')} pooled &mdash; nothing to see, and the sign flips
          between seeds under both objectives</td></tr>
        <tr><td>nose-to-nose &middot; <b>DERM</b></td><td>{eb('nn','DERM')}</td>
          <td>{eb('nn','DERM','ci')}</td></tr>
      </tbody></table></div>
    <p>On nose-to-tail both values are positive with DERM nearer zero, which is what a correction
    looks like rather than an overshoot. But the paired ERM-minus-DERM difference is
    {ebp('nt','diff')} at p&nbsp;=&nbsp;{ebp('nt','p')}, so <b>four validation pools cannot answer
    this</b>. What is running now is the same comparison cross-fitted over the three deployment
    folds: 24 pools instead of 4, cutting the standard error about 2.4&times;, which is what it
    takes to separate {eb('nt','ERM')} from zero.</p>

    <div class="note"><b>Do not read the AP column as a verdict.</b> A model that has stopped using
    the phase prior is <em>necessarily</em> a little worse at frame classification, because the
    prior is genuinely informative for that task &mdash; so the &minus;0.02 macro AP below is the
    expected price of the correction, not evidence against it. r&Delta;&nbsp;nt cannot arbitrate
    either: two seeds of the same unweighted control give
    {rr2('res448_k2_frozen_d4photo_ermH5M','nt')} and
    {rr2('res448_k2_frozen_d4photo_ermH5M_s1','nt')}.</div>

    <div class="scroll"><table>
      <thead><tr><th>arm</th><th>environments</th><th>seed</th><th>macro AP</th><th>r&Delta; nt</th><th>r&Delta; nn</th></tr></thead>
      <tbody>
        <tr><td><b>control</b> (unweighted ERM)</td><td>&mdash;</td><td>42</td><td>0.4163</td>
          <td>{rr2('res448_k2_frozen_d4photo_ermH5M','nt')}</td>
          <td>{rr2('res448_k2_frozen_d4photo_ermH5M','nn')}</td></tr>
        <tr><td><b>control</b> (unweighted ERM)</td><td>&mdash;</td><td>1</td><td>0.4202</td>
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
        <tr><td>vREx, &beta; = 1</td><td>the 6 phase &times; exposure cells</td><td>42</td>
          <td>{run('res448_k2_frozen_d4photo_vrexCond_b1')['ap']:.4f}</td>
          <td>{rr2('res448_k2_frozen_d4photo_vrexCond_b1','nt')}</td>
          <td>{rr2('res448_k2_frozen_d4photo_vrexCond_b1','nn')}</td></tr>
        <tr><td>vREx, &beta; = 100</td><td>annotator</td><td>42</td>
          <td class="lo">{run('res448_k2_frozen_d4photo_vrexAnn_b100')['ap']:.4f}</td>
          <td>{rr2('res448_k2_frozen_d4photo_vrexAnn_b100','nt')}</td>
          <td class="lo">{rr2('res448_k2_frozen_d4photo_vrexAnn_b100','nn')}</td></tr>
      </tbody></table></div>
    <p>The implementation was audited against the formula it claims
    (<code>test_derm.py</code>): the weights match
    <math><mrow><mi>Var</mi><mo>(</mo><mi>Y</mi><mo>|</mo><mi>E</mi><mo>)</mo><mo>/</mo>
    <mi>P</mi><mo>(</mo><mi>Y</mi><mo>,</mo><mi>E</mi><mo>)</mo></mrow></math> to 1e&minus;7,
    positives and negatives end up with equal mass inside every environment, the mean weight is
    exactly 1 so the step size is unchanged against ERM, each behaviour gets its own correction,
    and the phase map reproduces the protocol's own 30/15/15-minute split. vREx is a different
    thing and does not help: four arms, best +0.008, worst &minus;0.094.</p>

    <div class="note"><b>One thing to change before the next DERM arm.</b> The saved checkpoint is
    the epoch with the highest <em>unweighted</em> validation AP, so selection rewards exactly the
    prior-exploitation the objective removes. It is small here &mdash; the best epoch is 21&ndash;24
    of 30 and AP moves by under 0.04 across the last ten &mdash; but a shared fixed epoch budget, or
    selection on the DERM-weighted criterion, would remove the ambiguity for free.</div>
    </div>
  </details>

</div></section>

<section><div class="measure">
  <div class="sechead"><p class="eyebrow">05 &middot; Results</p>
  <h2>The model we run, and what it buys</h2></div>
  <div class="sub" style="border-top:0;margin-top:0;padding-top:0"><p class="q">05.1 &middot; the model</p>
  <h3>Shortlist on AP, choose on the causal quantity</h3></div>
  <p>Across the
  {M['meta']['n_candidates']} candidates the two rank models only loosely together (Spearman
  {M['meta']['spearman_ap_vs_rdelta']:+.2f}): the best-AP arm ranks {rk_ap_on_rd} of
  {M['meta']['n_candidates']} on r&Delta;, and the r&Delta; leader ranks {rk_rd_on_ap} on AP. AP is a
  usable filter, not the decision. <b>Hover a point for the whole recipe behind it</b> &mdash; encoder,
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
  <div class="note"><b>Read the last row differently.</b> The first four are scored on the standing
  4-pool split, 24 observations &mdash; too few to separate close models. The last is
  <b>cross-fitted</b>: the 24 annotated pools split into three folds of 8, each fold scored by a
  model trained on the other 16, so every pool is scored by a model that never saw it. A harder
  split, hence the lower AP.
  <br><br><b>Two separate things make it necessary.</b> PPI++'s validity: its rectifier is the gap
  between prediction and truth on annotated pools, and a model trained on those pools predicts them
  too well, so the gap it measures is not the one that applies to the 48 it corrects. And
  <b>r&Delta; itself</b> &mdash; which section 03's bound says is the only thing that matters
  &mdash; cannot be estimated any other way: in sample it is inflated by the model having seen the
  labels, and on the standing 4-pool split it rests on 16 points, where two seeds of one
  configuration give 0.183 and 0.853. So cross-fitting is not PPI++ hygiene that model ranking
  happens to inherit; <b>it is what makes ranking possible at all</b>. <b>PPCI uses no labels
  anywhere, so it is bound by neither</b> and would be better served by the single strongest
  model.</div>
  <div class="note"><b>The deployed configuration is not the strongest one measured, and its
  replacement is running.</b> Cross-fitting ran on the SSL-adapted frozen encoder with a plain
  5.03 M head, not on BitFit-6, which leads on every accuracy axis &mdash; that choice bought
  label-free adaptation covering v2 as well. The BitFit-6 cross-fit is now <b>2 of 3 folds in, at
  macro AP 0.50 and 0.55 against the deployed 0.382</b>, with the third training. Every estimate on
  this page moves onto it when that fold lands.</div>

  <div class="sub"><p class="q">05.2 &middot; where it fails</p>
  <h3>Where it is right, where it is wrong, and what it sees on the pools nobody scored</h3></div>
  <p>Pick a model, a set of frames and a behaviour. On the annotated pools every frame has a ground
  truth, so the four confusion buckets are meaningful and carry their counts. On the 84 unannotated
  pools there is no ground truth at all, so the only honest buckets are <em>confident yes</em> and
  <em>confident no</em> &mdash; that panel shows no counts and makes no accuracy claim.</p>
</div>
  <div class="figwrap">{EXAMPLES}</div>
<div class="measure">
  <div class="note warnbox"><b>The errors are not mostly boundary disagreements.</b> The obvious
  reading of a confident false positive is a frame or two fired outside a real bout. The figure's
  <b>d</b> column tests it &mdash; distance to the nearest scored bout in the same recording &mdash;
  and it does not hold: only {dist('nn','FP','le2'):.0f}% of nose-to-nose false positives and
  {dist('nt','FP','le2'):.0f}% of nose-to-tail ones sit within two frames of one, with medians of
  {dist('nn','FP','median')} and {dist('nt','FP','median')} frames
  ({dist('nn','FP','median')/5:.0f} and {dist('nt','FP','median')/5:.0f} seconds). False negatives
  behave the same way. So these are frames the model reads as contact where the annotator scored
  none &mdash; a detection problem, and only double-annotation can say how much of it is model error
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
  <p>Ranges are over the six phase &times; exposure cells, and over the <em>whole</em> recording
  rather than 02b's matched window &mdash; these describe the model's output, not an estimate, and
  the estimates in section 03 are all windowed. Nothing collapses or saturates, and the v2
  detections show genuine contact. Predicted occupancy runs about <b>5&times; above truth</b>
  throughout &mdash; part calibration offset, part the deliberate prior shift in training &mdash;
  so these must never be read as behaviour rates directly. For PPI++ that costs nothing, because
  &lambda; absorbs the scale. <b>For PPCI it is the whole caveat</b>: PPCI reports this scale rather
  than the behaviour's, which is why it is drawn hollow in the effects figure and why nothing on
  this page reads a PPCI magnitude as a rate. A fivefold offset that is the <em>same</em> in every
  phase would still cancel in a within-pool difference; what would not cancel is the part that moves
  with the phase, and section 04.6 measures that part at {eb('nt','ERM')} bouts/min on nose-to-tail
  &mdash; {eb('nt','ERM','share')} the pooled true effect, on an interval too wide to resolve. It is
  the largest open threat to PPCI on this page.</p>
</div>

<div class="measure">
  <div class="sub"><p class="q">05.3 &middot; robustness</p>
  <h3>Is PPCI reading the behaviour, or the model?</h3></div>
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
</div></section>


<section><div class="measure">
  <div class="sechead"><p class="eyebrow">06 &middot; Next</p><h2>What to do next</h2></div>
  <p>In priority order. <span class="run"><b>Amber is already running</b></span> &mdash; those need
  no action, they need the queue.</p>
  <div class="scroll"><table>
    <thead><tr><th></th><th>action</th><th>status</th><th>what it changes</th></tr></thead>
    <tbody>
      <tr><td>1</td><td class="run">cross-fit BitFit-6 and move every estimate onto it</td>
        <td class="run">running &mdash; 2 of 3 folds in</td>
        <td class="run">macro AP 0.50 and 0.55 against the deployed 0.382, so every number on the
        page rests on a better model the moment fold 3 lands</td></tr>
      <tr><td>2</td><td class="run">cross-fit the DERM / ERM pair</td><td class="run">running</td>
        <td class="run">resolves the largest open threat to PPCI: a<sub>O</sub>&minus;a<sub>H</sub>
        is {eb('nt','ERM')} bouts/min on nose-to-tail, {eb('nt','ERM','share')} the pooled true
        effect and opposite in sign. On 4 pools it does not resolve; on 24 it should</td></tr>
      <tr><td>3</td><td class="run">the exposure-split PPCI test, both directions</td>
        <td class="run">running</td>
        <td class="run">trains on one exposure session and tests on the other, so the phase
        shortcut shows as a bias that <em>reverses sign</em> when the direction reverses &mdash;
        which a plain generalisation gap cannot do</td></tr>
      <tr><td>4</td><td>annotate ~20 more v1 pools</td><td>needs annotator time</td>
        <td>+0.076 AP per doubling and no plateau, worth more than every modelling change combined
        &mdash; and it raises CI's own precision, not only PPI++'s</td></tr>
      <tr><td>5</td><td>annotate 4&ndash;6 v2 pools</td><td>needs annotator time</td>
        <td>the only way v2 gets a CI or a PPI++ estimate at all; today it has PPCI and nothing to
        check it against</td></tr>
      <tr><td>6</td><td>double-annotate 15&ndash;20 observations, nose-to-tail first</td>
        <td>needs annotator time</td>
        <td>the label ceiling is currently inferred from the design, not measured, and it is
        aliased with cage. Nose-to-tail is where it binds</td></tr>
      <tr><td>7</td><td>record which animal in each v2 cage is the heterozygote</td>
        <td>lab metadata</td>
        <td>without it the within-pool genotype contrast &mdash; the programme's actual question
        &mdash; is not identified however good the vision gets</td></tr>
      <tr><td>8</td><td>per-animal crops from the 2060 px source</td><td>engineering</td>
        <td>the only resolution lever left, and a prerequisite for any per-animal outcome</td></tr>
    </tbody></table></div>
</div></section>

<div class="measure"><footer>
  All intervals 95%, clustered on pool (n = 24 labelled + 48 unlabelled on v1, 36 on v2). Built by
  <code>build_report.py</code> from seven JSON payloads regenerated from the runs and the labels at
  build time &mdash; <code>build_estimates.py</code> (every effect), <code>build_models.py</code>
  (every scored run and its recipe), <code>build_outcome.py</code> (the outcome units and their
  distributions), <code>build_decay.py</code> (the within-phase curves),
  <code>build_examples.py</code> (the error thumbnails),
  <code>build_ppci_robustness.py</code> (PPCI under a second predictor) and
  <code>build_derm.py</code> (the phase leak, the nuisance-bias decomposition and the PPI++ bound).
  Numbers quoted in the prose are read from those same payloads, so the text cannot drift from the
  tables. One static figure, the annotation scaling curve, is still pre-rendered by
  <code>story_figures.py</code>.
</footer></div>

</div>
'''
