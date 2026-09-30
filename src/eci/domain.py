"""
Experimental domains of the ECI (Exploratory Causal Inference) NES pipeline.

A domain says where its frames, codes and results live, how to build its design table (one row per
video, in annotations.csv block order, with the half-open row range of the video in the per-frame
codes memmap) and which NES analyses run on it. Every script takes --domain (default mice); the mice
domain wraps the original mice v1 code (src/eci/contrasts.py load_design / TRANSITIONS / STAGES), so
mice runs are unchanged. Kept apart from src/domain/ (the PPCI domain handlers).

Analysis families (the unit is what gets resampled / tested, never a frame):
  A  paired within unit: the same unit at stage a and stage b (mice: one pool, one genotype);
     contrasts.paired, nes.paired_effect_search.
  B  two-sample: control vs treatment units (mice: het vs wt videos of one stage; ants: videos of one
     experiment with treatment value c vs t); contrasts.two_sample, nes.neural_effect_search.

Domains:
  mice  v1 (data/mice/DATA_STRUCTURE.md): 72 pools x 6 stages; A = 4 stage transitions x 2 genotypes,
        B = het vs wt at stages 1..6 (ids and order as the original runs, analysis_ids()).
  ants  v2 (ISTAnt, 44 videos, treatment 1 vs 2) and v3 (212 videos, treatments 2/4/6/7/8/9; the
        analyses use 2 vs 6 and 2 vs 8). All videos 3000 frames at 5 fps. Unit = video, family B
        only. v3 t = 8 was recorded only on day C (brighter arena) and t = 2 / 6 only on days A / B,
        so v3_2_vs_8 is confounded with the recording day (meta 'confound'). Per-frame table:
        dataset/ants/eci/annotations.csv, per-video table dataset/ants/eci/experiment.csv (both
        written by scripts/eci/ants_prepare.py). Primary nuisance: none.

Analysis sets (get_domain(name, analysis_set), runners' --analysis-set; default 'core' = the analyses above,
unchanged): ants 'pairs' = the 3 core analyses (same ids, meta and order) followed by every other pair of
treatment values within one experiment, control = the lower treatment number, treatment = the higher
(v2: 1 vs 2 only; v3: all 15 pairs of 2/4/6/7/8/9, 12 new ids v3_<c>_vs_<t>). A new pair's meta 'confound'
is computed from experiment.csv: chi-square test of recording day x arm, p < CONFOUND_ALPHA -> 'recording
day: ...' (the arms' day counts and p), else ''. analysis_table() gives n per arm, day counts, p and the flag.

Functions / classes:
    Analysis       one NES contrast: id, family, unit, summary.csv meta, direction labels; select()
    Domain         base class: paths, fps / n_match, load_design, analyses, subgroups, balance, texts
    MiceDomain     mice v1
    AntsDomain     ants v2 + v3
    get_domain     (name, analysis set) -> Domain instance (cached)
    DOMAINS        the domain names
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import contrasts as C

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = ('mice', 'ants')


@dataclass
class Analysis:
    """One NES contrast.

    id          analysis_id in summary.csv / result dirs
    family      'A' (paired within unit, stage a -> b) or 'B' (two-sample, control vs treatment)
    unit        design column of the unit ('pool' mice, 'observation_id' ants)
    meta        extra summary.csv columns of the analysis, in column order (after analysis_id, family)
    directions  (label of tau > 0, label of tau < 0)
    genotype, stages   family A: the genotype and (stage a, stage b) of the pairs
    where       family B: {design column: value} restricting the rows (e.g. {'stage': 2})
    arm, control, treatment   family B: design column and its values of the two arms (T = 1 treatment)
    matched     family A: the time-matched window ('last' frames of stage a vs all of stage b) applies
    """
    id: str
    family: str
    unit: str
    meta: dict
    directions: tuple
    genotype: str = None
    stages: tuple = None
    where: dict = field(default_factory=dict)
    arm: str = None
    control: tuple = ()
    treatment: tuple = ()
    matched: bool = False

    def select(self, design):
        """The design rows of this analysis, in design order. Family B rows get T = 1 (treatment) /
        0 (control) from `arm`; family A rows are both stages of `genotype`."""
        if self.family == 'A':
            return design[(design['genotype'] == self.genotype) & design['stage'].isin(self.stages)]
        m = design[self.arm].isin(self.control + self.treatment).values
        for k, v in self.where.items():
            m &= (design[k] == v).values
        d = design[m].copy()
        d['T'] = d[self.arm].isin(self.treatment).astype(int)
        return d


class Domain:
    """Base class. Paths are absolute (repo root = ROOT)."""
    name = ''
    title = ''            # 'mice v1' in report headers
    subjects = ''         # 'mice' / 'ants' in report prose
    ann_path = None       # per-frame annotations.csv (row order of every codes memmap)
    eci_dir = None        # dataset/.../eci: sae/, codes/, train_tokens/, fg448/background/
    nes_root = None       # results/.../eci/nes
    fps = 5.0
    n_match = 4500        # frames of the time-matched window (family A habituation -> odor)
    nuisance = 'none'     # primary nuisance of the NES runs ('none' | 'nfg')
    fg_rule = 'fg448'     # default foreground rule (src/eci/foreground.py RULES)
    subgroups = {}        # {'line': (...), 'sex': (...)}: restriction of the units before the search
    duration_col = ''     # design column of the stage_durations.csv groups
    null_two = None       # analysis id of the two-sample label-shuffle null
    null_paired = None    # analysis id of the paired within-unit swap null (None: no family A)
    frame_analysis = None  # analysis id of the frame- vs video-level pseudo-replication illustration
    frame_key = ''        # suffix of sanity 'pseudo_replication_<key>' and the C_frame_<key>/ dir
    balance_key = ''      # sanity.json key of balance()
    shuffle_word = ''     # '<null_two>_<word>_shuffle_n_selected' in the bouts sanity.json
    desc_by = ()          # design columns of the per-neuron descriptives groups (run_nes_bouts.py)
    meta_cols = ()        # per-video columns of video_meta() carried into src/eci/viz.py video blocks
    desc_table = ()       # (row column, row values, column column, column values) of their report tables
    text = {}             # report phrases: window, families_nes, families_bouts, null_two, frame,
    #                       null_two_bouts, null_paired_bouts
    analysis_sets = ('core',)  # names accepted by get_domain(name, analysis_set); 'core' = the default analyses

    def __init__(self, analysis_set='core'):
        if analysis_set not in self.analysis_sets:
            raise ValueError(f'domain {self.name} has no analysis set {analysis_set!r}; known: {self.analysis_sets}')
        self.analysis_set = analysis_set

    @property
    def codes_root(self):
        return self.eci_dir / 'codes'

    @property
    def ann_rel(self):
        """annotations.csv relative to the dataset dir (scripts take --dataset-dir)."""
        return self.ann_path.relative_to(ROOT / 'dataset')

    @property
    def eci_rel(self):
        return self.eci_dir.relative_to(ROOT / 'dataset')

    def load_design(self):
        raise NotImplementedError

    @property
    def analyses(self):
        raise NotImplementedError

    def analysis(self, aid):
        return {a.id: a for a in self.analyses}[aid]

    def analysis_ids(self):
        return [a.id for a in self.analyses]

    def subset(self, design, line='all', sex='all'):
        """The design restricted to one subgroup (obs_row kept: it indexes the full-cohort arrays)."""
        if (line, sex) != ('all', 'all'):
            raise SystemExit(f'domain {self.name} has no line / sex subgroups')
        return design.copy()

    def subgroup_info(self, design, line='all', sex='all'):
        return {'n_videos': int(len(design))}

    def describe(self, design):
        """One-line summary of the units, printed by the runners."""
        return f'{len(design)} videos'

    def balance(self, design):
        return {}

    def balance_report(self, sanity):
        return []

    def val_split(self, val_from=None, seed=0):
        """SAE validation split -> (held-out unit names, (n_rows,) bool: annotations.csv row held out)."""
        raise NotImplementedError

    def video_meta(self):
        raise NotImplementedError


class MiceDomain(Domain):
    """Mice v1, exactly the original code paths."""
    name, title, subjects = 'mice', 'mice v1', 'mice'
    ann_path = ROOT / 'dataset/mice/v1/annotations.csv'
    experiment_csv = ROOT / 'data/mice/v1/experiment.csv'
    data_dir = ROOT / 'data'
    eci_dir = ROOT / 'dataset/mice/v1/eci'
    nes_root = ROOT / 'results/vision/mice/eci/nes'
    n_match = 4500
    subgroups = {'line': C.LINES, 'sex': C.SEXES}
    duration_col = 'stage'
    null_two, null_paired, frame_analysis, frame_key = 'B_stage2', 'A_het_1to2', 'B_stage2', 'stage2'
    balance_key, shuffle_word = 'genotype_balance', 'genotype'
    desc_by = ('stage', 'genotype')
    meta_cols = ('pool', 'stage', 'genotype')
    desc_table = ('genotype', ('het', 'wt'), 'stage', tuple(range(1, 7)))  # row col, rows, column col, columns
    stage_label = {v: f'{p},{o}' for (o, p), v in C.STAGES.items()}  # 1 -> 'H,S'
    text = {'window': 'stage',
            'families_nes': 'Family A = paired stage transition within genotype (unit = pool, tau > 0 = increase from '
                            'stage a to b); family B = het vs wt within stage (unit = video, tau > 0 = higher in het).',
            'families_bouts': 'Family A = paired stage transition within genotype (unit = pool, tau > 0 = more '
                              'bouts/min at the later stage); family B = het vs wt within stage (tau > 0 = more '
                              'bouts/min in het).',
            'null_two': 'Genotype labels shuffled across pools (stage 2', 'frame': 'genotype at stage 2',
            'null_two_bouts': 'genotype shuffled across pools (B stage 2)',
            'null_paired_bouts': 'stage labels swapped within pool at random (A het 1->2)'}

    def load_design(self):
        return C.load_design(self.ann_path, self.experiment_csv)

    @property
    def analyses(self):
        out = []
        for g in ('het', 'wt'):
            for tr, (a, b) in C.TRANSITIONS.items():
                out.append(Analysis(f'A_{g}_{tr}', 'A', 'pool', {'genotype': g, 'stage': '', 'transition': tr},
                                    ('up', 'down'), genotype=g, stages=(a, b), matched=a in (1, 4)))
        for s in range(1, 7):
            out.append(Analysis(f'B_stage{s}', 'B', 'pool', {'genotype': 'het_vs_wt', 'stage': s, 'transition': ''},
                                ('het>wt', 'het<wt'), where={'stage': s}, arm='genotype', control=('wt',),
                                treatment=('het',)))
        return out

    def subset(self, design, line='all', sex='all'):
        return C.subset_design(design, line, sex)

    def subgroup_info(self, design, line='all', sex='all'):
        return {'line': line, 'sex': sex, 'n_pools': int(design['pool'].nunique()),
                'n_het_pools': int(design[design['T'] == 1]['pool'].nunique()), 'n_videos': int(len(design))}

    def describe(self, design):
        return (f'{design["pool"].nunique()} pools ({design[design["T"] == 1]["pool"].nunique()} het), '
                f'{len(design)} videos')

    def balance(self, design):
        """Pool-level het/wt counts per recording field, with a chi-square test of independence.
        'cage_pos' = the _1/_2/_3 suffix of the pool id (cage position on the recording day; 'none' if
        the pool id has no suffix), 'month' = recording month."""
        from scipy.stats import chi2_contingency
        e = pd.read_csv(self.experiment_csv)
        p = e[e['pool'].isin(set(design['pool']))].drop_duplicates('pool').copy()
        p['cage_pos'] = p['pool'].str.extract(r'_(\d)$')[0].fillna('none')
        p['month'] = p['date'].str[:7]
        p['annotator'] = p['annotator'].fillna('none')
        p['hour'] = p['time'].str[:2]
        out = {}
        for c in ('line', 'sex', 'seed', 'cage_pos', 'month', 'hour', 'annotator'):
            tab = pd.crosstab(p[c], p['genotype'])
            pval = float(chi2_contingency(tab.values)[1]) if tab.shape[0] > 1 else 1.0
            out[c] = {'counts': {str(k): {g: int(v) for g, v in r.items()} for k, r in tab.iterrows()}, 'chi2_p': pval}
        return out

    def balance_report(self, sanity):
        sg = sanity.get('subgroup', {})
        lines = [f'## Genotype balance across recording fields (pool level, {sg.get("n_het_pools", 36)} het / '
                 f'{sg.get("n_pools", 72) - sg.get("n_het_pools", 36)} wt)', '']
        for c, v in sanity.get(self.balance_key, {}).items():
            cnt = ', '.join(f'{k}: {d.get("het", 0)}/{d.get("wt", 0)}' for k, d in v['counts'].items())
            lines.append(f'- {c} (het/wt): {cnt}; chi-square p = {v["chi2_p"]:.3g}')
        return lines

    def val_split(self, val_from=None, seed=0):
        """The held-out pools of an existing SAE (val_from = its metrics.json 'val_pools')."""
        import json
        val_pools = json.loads(Path(val_from).read_text())['val_pools']
        ann = pd.read_csv(self.ann_path, usecols=['observation_id'])
        exp = pd.read_csv(self.experiment_csv).set_index('observation_id')
        pool = exp.loc[ann.observation_id.values, 'pool'].values
        codes, names = pd.factorize(pool)
        return val_pools, np.isin(np.array(list(names))[codes.astype(np.int16)], val_pools)

    def video_meta(self):
        from .viz import video_meta
        return video_meta(self.data_dir)


class AntsDomain(Domain):
    """Ants v2 + v3, unit = video, family B only."""
    name, title, subjects = 'ants', 'ants v2 + v3', 'ants'
    eci_dir = ROOT / 'dataset/ants/eci'
    ann_path = eci_dir / 'annotations.csv'
    experiment_csv = eci_dir / 'experiment.csv'
    nes_root = ROOT / 'results/vision/ants/eci/nes'
    sources = {'v2': ROOT / 'dataset/ants/v2/annotations.csv', 'v3': ROOT / 'dataset/ants/v3/annotations.csv'}
    raw_experiments = {'v2': ROOT / 'data/ants/v2/experiment.csv', 'v3': ROOT / 'data/ants/v3/experiment.csv'}
    design_cols = ('experiment', 'T', 'batch', 'position', 'annotator', 'recording_date', 'nestbox')
    balance_fields = ('batch', 'position', 'annotator', 'recording_date', 'nestbox')
    n_match = 3000  # = the whole 10 min video (no family A, the matched window is unused)
    fg_rule = 'ants'
    duration_col = 'experiment'
    null_two, null_paired, frame_analysis, frame_key = 'v2_1_vs_2', None, 'v2_1_vs_2', 'v2_1_vs_2'
    balance_key, shuffle_word = 'treatment_balance', 'treatment'
    desc_by = ('experiment', 'T')
    meta_cols = ('experiment', 'T')
    desc_table = ('experiment', ('v2', 'v3'), 'T', (1, 2, 4, 6, 7, 8, 9))
    text = {'window': 'video',
            'families_nes': 'Family B = treated vs control videos within one experiment (unit = video, tau > 0 = '
                            'higher in the treated videos); v3_2_vs_8 is confounded with the recording day (t=8 only '
                            'on day C, brighter arena).',
            'families_bouts': 'Family B = treated vs control videos within one experiment (unit = video, tau > 0 = '
                              'more bouts/min in the treated videos); v3_2_vs_8 is confounded with the recording day.',
            'null_two': 'Treatment labels shuffled across videos (v2_1_vs_2', 'frame': 'treatment in v2 (1 vs 2)',
            'null_two_bouts': 'treatment shuffled across videos (v2_1_vs_2)', 'null_paired_bouts': ''}
    # (experiment, control value, treatment value, confound)
    CONTRASTS = (('v2', 1, 2, ''), ('v3', 2, 6, ''),
                 ('v3', 2, 8, 'recording day: t=8 only on day C (brighter arena), t=2 only on days A/B'))
    analysis_sets = ('core', 'pairs')
    CONFOUND_ALPHA = 0.05  # 'pairs': recording day x arm chi-square p below this -> confounded
    text_pairs = {'families_nes': 'Family B = treated vs control videos within one experiment (unit = video, tau > 0 = '
                                  'higher in the treated videos = the higher treatment number); every pair of treatment '
                                  'values within an experiment, control = the lower number. Pairs whose arms differ in '
                                  'recording day (chi-square p < 0.05) are flagged in the confound column: in v3, '
                                  't=8 and t=9 were recorded only on day C (brighter arena), t=2/4/6/7 only on days A/B.',
                  'families_bouts': 'Family B = treated vs control videos within one experiment (unit = video, tau > 0 = '
                                    'more bouts/min in the treated videos = the higher treatment number); every pair of '
                                    'treatment values within an experiment; pairs confounded with the recording day are '
                                    'flagged in the confound column.'}

    def __init__(self, analysis_set='core'):
        super().__init__(analysis_set)
        if analysis_set == 'pairs':
            self.text = {**type(self).text, **self.text_pairs}

    def load_design(self):
        return C.load_design_ants(self.ann_path, self.experiment_csv)

    def contrasts(self):
        """(experiment, control, treatment, confound) of the analysis set: 'core' = CONTRASTS; 'pairs' = CONTRASTS,
        then the other within-experiment pairs (control < treatment), in experiment / control / treatment order."""
        if self.analysis_set == 'core':
            return self.CONTRASTS
        core = {(e, c, t) for e, c, t, _ in self.CONTRASTS}
        e = pd.read_csv(self.experiment_csv)
        extra = []
        for ex in sorted(e['experiment'].unique()):
            ts = sorted(int(t) for t in e.loc[e['experiment'] == ex, 'T'].unique())
            for i, c in enumerate(ts):
                for t in ts[i + 1:]:
                    if (ex, c, t) not in core:
                        extra.append((ex, c, t, self.day_confound(e, ex, c, t)[0]))
        return tuple(self.CONTRASTS) + tuple(extra)

    def day_confound(self, e, ex, c, t):
        """-> (confound text or '', chi-square p, {arm value: {day: videos}}) of recording day x arm in
        experiment ex (e = experiment.csv); one day in total -> p = 1."""
        from scipy.stats import chi2_contingency
        d = e[(e['experiment'] == ex) & e['T'].isin([c, t])]
        tab = pd.crosstab(d['recording_date'].astype(str), d['T'])
        p = float(chi2_contingency(tab.values)[1]) if tab.shape[0] > 1 and tab.shape[1] > 1 else 1.0
        days = {int(v): {str(k): int(n) for k, n in tab[v].items() if n > 0} for v in tab.columns}
        txt = ''
        if p < self.CONFOUND_ALPHA:
            on = {v: ('day ' if len(days[v]) == 1 else 'days ') + '/'.join(days[v]) for v in (c, t)}
            txt = f'recording day: t={c} on {on[c]}, t={t} on {on[t]} (chi-square p = {p:.2g})'
        return txt, p, days

    @property
    def analyses(self):
        return [Analysis(f'{e}_{c}_vs_{t}', 'B', 'observation_id',
                         {'experiment': e, 'control': c, 'treatment': t, 'confound': conf},
                         ('treated>control', 'treated<control'), where={'experiment': e}, arm='T',
                         control=(c,), treatment=(t,))
                for e, c, t, conf in self.contrasts()]

    def analysis_table(self):
        """One row per analysis of the set: id, experiment, control, treatment, n_control, n_treatment (videos),
        recording-day counts per arm, day_chi2_p, confounded (day_chi2_p < CONFOUND_ALPHA), confound (meta text),
        core (one of the 3 default analyses)."""
        e = pd.read_csv(self.experiment_csv)
        core = {(x, c, t) for x, c, t, _ in self.CONTRASTS}
        rows = []
        for an in self.analyses:
            m = an.meta
            ex, c, t = m['experiment'], m['control'], m['treatment']
            _, p, days = self.day_confound(e, ex, c, t)
            g = e[e['experiment'] == ex]
            rows.append({'analysis_id': an.id, 'experiment': ex, 'control': c, 'treatment': t,
                         'n_control': int((g['T'] == c).sum()), 'n_treatment': int((g['T'] == t).sum()),
                         'days_control': days.get(c, {}), 'days_treatment': days.get(t, {}), 'day_chi2_p': p,
                         'confounded': bool(p < self.CONFOUND_ALPHA), 'confound': m['confound'],
                         'core': (ex, c, t) in core})
        return rows

    def describe(self, design):
        n = design.groupby(['experiment', 'T']).size()
        return f'{len(design)} videos (' + ', '.join(f'{e} t={t}: {k}' for (e, t), k in n.items()) + ')'

    def balance(self, design):
        """Per analysis: control / treatment video counts per recording field, chi-square test of
        independence (fields with one value -> p = 1)."""
        from scipy.stats import chi2_contingency
        out = {}
        for an in self.analyses:
            d = an.select(design)
            arm = np.where(d['T'] == 1, 'treatment', 'control')
            out[an.id] = {}
            for c in self.balance_fields:
                tab = pd.crosstab(d[c].fillna('none').astype(str).values, arm)
                pval = float(chi2_contingency(tab.values)[1]) if tab.shape[0] > 1 and tab.shape[1] > 1 else 1.0
                out[an.id][c] = {'counts': {str(k): {g: int(v) for g, v in r.items()} for k, r in tab.iterrows()},
                                 'chi2_p': pval}
        return out

    def balance_report(self, sanity):
        lines = ['## Treatment balance across recording fields (video level, control/treatment)', '']
        for aid, fields in sanity.get(self.balance_key, {}).items():
            for c, v in fields.items():
                cnt = ', '.join(f'{k}: {d.get("control", 0)}/{d.get("treatment", 0)}' for k, d in v['counts'].items())
                lines.append(f'- {aid} {c}: {cnt}; chi-square p = {v["chi2_p"]:.3g}')
        return lines

    def val_split(self, val_from=None, seed=0, frac=0.1):
        """About frac of the videos of every (experiment, T) held out (at least 1), seeded."""
        e = pd.read_csv(self.experiment_csv)
        rng = np.random.default_rng(seed)
        val = []
        for _, g in e.sort_values('observation_id').groupby(['experiment', 'T'], sort=True):
            k = max(1, int(round(frac * len(g))))
            val += sorted(rng.choice(g['observation_id'].values, k, replace=False).tolist())
        ann = pd.read_csv(self.ann_path, usecols=['observation_id'])
        return val, ann['observation_id'].isin(val).values

    def video_meta(self):
        return pd.read_csv(self.experiment_csv)


@lru_cache(maxsize=None)
def get_domain(name='mice', analysis_set='core'):
    if name not in DOMAINS:
        raise ValueError(f'unknown domain {name!r}; known: {DOMAINS}')
    return {'mice': MiceDomain, 'ants': AntsDomain}[name](analysis_set)
