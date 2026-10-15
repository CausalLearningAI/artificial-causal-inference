"""
Checks of the ants 30-min ECI layout code (src/eci/domain.py AntsDomain, src/eci/foreground.py background windows,
scripts/eci/fg_encode_extend.py row map). CPU only; each check prints PASS / FAIL lines and exits non-zero on a FAIL.

    contrasts   synthetic experiment table (v2, v3, v4, v4_infested): analysis ids / order, the videos and arms each
                analysis selects, --day (within-experiment analyses keep the days holding both arms; infested vs
                healthy keeps every day), confound texts, analysis_table counts; the 10-min analyses select exactly
                what the previous code did (re-implemented here)
    windows     bg_windows on 3000 / 9000-frame videos; the relocatable time blocks (FgBackgrounds.segment) give the
                same block as the absolute segment_of for every row of real 10-min ants videos, and the same block
                after the video moves to other rows
    atlas       scripts/eci/build_explorer.py on the synthetic table: kept analyses (VIEW 'keep' / 'drop'), page metadata
                of a within and an infested vs healthy analysis (arms = the experiment|T groups of each arm), the
                contrast video masks
    rowmap      --new <annotations.csv of a 30-min layout> vs --old (default dataset/ants/eci/annotations.csv): rows
                reused per experiment and frame range, every reused row has frame_idx < 3000 and the same frame_path

Usage: python scripts/eci/ants30_checks.py contrasts | windows | atlas | rowmap --new <csv>
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
from src.eci.domain import AntsDomain  # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print(f'{"PASS" if ok else "FAIL"}  {name}{"  " + str(detail) if detail else ""}', flush=True)
    if not ok:
        FAIL.append(name)


def synthetic_design():
    """(experiment, T, recording_date, n videos) cells -> one row per video."""
    cells = [('v2', 1, '18.04.2024', 3), ('v2', 2, '18.04.2024', 3), ('v2', 2, '19.04.2024', 2),
             ('v3', 2, 'A', 3), ('v3', 6, 'A', 2), ('v3', 6, 'B', 3), ('v3', 2, 'B', 1), ('v3', 8, 'C', 4),
             ('v4', 2, 'D', 3), ('v4', 6, 'D', 4), ('v4', 2, 'F', 2),
             ('v4_infested', 2, 'E', 3), ('v4_infested', 6, 'E', 2), ('v4_infested', 2, 'F', 1),
             ('v4_infested', 6, 'F', 3)]
    rows = []
    for e, t, d, n in cells:
        for i in range(n):
            rows.append({'observation_id': f'{e}_{t}_{d}_{i}', 'experiment': e, 'T': t, 'batch': '1', 'position': i,
                         'annotator': 'A', 'recording_date': d, 'nestbox': 'INF' if e == 'v4_infested' else 'C',
                         'infested': int(e == 'v4_infested')})
    return pd.DataFrame(rows)


def old_select(design, ex, c, t, day):
    """The pre-2026-10-15 Analysis.select of ants analysis ex_c_vs_t (where experiment == ex, arm T)."""
    m = design['T'].isin((c, t)).values & (design['experiment'] == ex).values
    d = design[m].copy()
    d['T'] = d['T'].isin((t,)).astype(int)
    if day:
        arms = d.groupby(d['recording_date'].astype(str))['T'].nunique()
        d = d[d['recording_date'].astype(str).isin(arms[arms == 2].index)]
    return d


def run_contrasts():
    e = synthetic_design()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / 'experiment.csv'
        e.to_csv(p, index=False)
        D = AntsDomain('core')
        D.experiment_csv = p
        ids = D.analysis_ids()
        want = ['v2_1_vs_2', 'v3_2_vs_6', 'v3_2_vs_8', 'v4_2_vs_6', 'v4_infested_2_vs_6', 'infested_vs_healthy_t2',
                'infested_vs_healthy_t6', 'infested_vs_healthy_t2_6']
        check('analysis ids and order', ids == want, ids)
        for day in (False, True):
            D.day_nuisance = day
            for x, c, t in (('v2', 1, 2), ('v3', 2, 6), ('v3', 2, 8), ('v4', 2, 6), ('v4_infested', 2, 6)):
                a, b = D.analysis(f'{x}_{c}_vs_{t}').select(e), old_select(e, x, c, t, day)
                check(f'{x}_{c}_vs_{t} day={day}: same rows and T as the previous select',
                      a['observation_id'].tolist() == b['observation_id'].tolist() and a['T'].tolist() == b['T'].tolist(),
                      f'n={len(a)}')
            for aid, ts in (('infested_vs_healthy_t2', (2,)), ('infested_vs_healthy_t6', (6,)),
                            ('infested_vs_healthy_t2_6', (2, 6))):
                an = D.analysis(aid)
                d = an.select(e)
                exp = e[e['experiment'].isin(('v4', 'v4_infested')) & e['T'].isin(ts)]
                check(f'{aid} day={day}: v4 + v4_infested videos with T in {ts}, every day (never day-matched)',
                      d['observation_id'].tolist() == exp['observation_id'].tolist() and an.day_col is None
                      and not an.day_match, f'n={len(d)}')
                check(f'{aid} day={day}: T = 1 exactly on the infested videos',
                      (d['T'].values == (d['experiment'] == 'v4_infested').values).all())
        D.day_nuisance = True
        d = D.analysis('v4_2_vs_6').select(e)
        check('v4_2_vs_6 --day: day F (t=2 only) dropped', set(d['recording_date']) == {'D'}, sorted(set(d['recording_date'])))
        an = D.analysis('infested_vs_healthy_t2')
        check('infested confound text names both arms\' days', an.meta['confound'] ==
              'recording day (not day-matched: healthy colonies on days D/F, infested on days E/F)', an.meta['confound'])
        check('infested meta labels', (an.meta['experiment'], an.meta['control'], an.meta['treatment']) ==
              ('v4 infested vs healthy', 'healthy t=2', 'infested t=2'), an.meta)
        check('pooled meta label', D.analysis('infested_vs_healthy_t2_6').meta['treatment'] == 'infested t=2+6')
        tab = {r['analysis_id']: r for r in D.analysis_table()}
        r = tab['infested_vs_healthy_t2_6']
        check('analysis_table n for the pooled contrast', (r['n_control'], r['n_treatment']) == (9, 9),
              (r['n_control'], r['n_treatment']))
        check('analysis_table: infested contrasts flagged confounded, day_match False',
              all(tab[a]['confounded'] and not tab[a]['day_match'] for a in want[5:]))
        check('analysis_table: v4_2_vs_6 day counts', (tab['v4_2_vs_6']['days_control'], tab['v4_2_vs_6']['days_treatment'])
              == ({'D': 3, 'F': 2}, {'D': 4}), (tab['v4_2_vs_6']['days_control'], tab['v4_2_vs_6']['days_treatment']))
        P = AntsDomain('pairs')
        P.experiment_csv = p
        extra = [a for a in P.analysis_ids() if a not in want]
        check('pairs = core + the other within-experiment pairs', extra == ['v3_6_vs_8'], extra)
        e2 = e[e['experiment'].isin(('v2', 'v3'))]
        e2.to_csv(p, index=False)
        D2 = AntsDomain('core')
        D2.experiment_csv = p
        check('10-min layout (v2 + v3 only): the three original analyses',
              D2.analysis_ids() == ['v2_1_vs_2', 'v3_2_vs_6', 'v3_2_vs_8'], D2.analysis_ids())


def run_atlas():
    from types import SimpleNamespace
    import build_explorer as B
    e = synthetic_design()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / 'experiment.csv'
        e.to_csv(p, index=False)
        D = AntsDomain('pairs')
        D.experiment_csv = p
        view = B.VIEW['ants']
        kept = B.kept_analyses(D, view)
        check('kept: core minus v3_6_vs_8 (VIEW drop)', kept == [a for a in D.analysis_ids() if a != 'v3_6_vs_8'], kept)
        cfg = SimpleNamespace(dom=D, view=view, ameta={r['analysis_id']: r for r in D.analysis_table()})
        m = B.analysis_meta(cfg, 'v3_2_vs_6', {'n_units': 9})
        check('within meta: ints, arms t=2 / t=6, not cross',
              (m['control'], m['treatment'], m['arms'], 'cross' in m) == (2, 6, [['t=2', [['v3', 2]]], ['t=6', [['v3', 6]]]], False), m)
        m = B.analysis_meta(cfg, 'infested_vs_healthy_t2_6', {'n_units': 18})
        check('infested meta: string arms over v4 / v4_infested, cross, stratum, confound',
              m['arms'] == [['healthy t=2+6', [['v4', 2], ['v4', 6]]], ['infested t=2+6', [['v4_infested', 2], ['v4_infested', 6]]]]
              and m['cross'] and m['stratum'] == 't=2+6' and m['day_match'] is False
              and m['confound'].startswith('recording day (not day-matched'), m)
        c, t = B.contrast_groups(SimpleNamespace(dom=D), 'infested_vs_healthy_t6', e)
        check('contrast videos (clips): v4 t=6 vs v4_infested t=6',
              set(e[c]['experiment'] + e[c]['T'].astype(str)) == {'v46'} and set(e[t]['experiment'] + e[t]['T'].astype(str))
              == {'v4_infested6'}, (int(c.sum()), int(t.sum())))


def run_windows():
    from src.eci.foreground import FgBackgrounds, RULES, bg_windows, load_background, obs_rows, segment_of
    w = bg_windows({'a': (0, 3000), 'b': (3000, 12000), 'c': (12000, 12100)}, 3000)
    check('bg_windows: 10-min video one window under its own id, 30-min video three',
          w == {'a': (0, 3000), 'b': (3000, 6000), 'b__w1': (6000, 9000), 'b__w2': (9000, 12000), 'c': (12000, 12100)}, w)
    check('bg_windows: no window = the videos', bg_windows({'a': (0, 3000)}, None) == {'a': (0, 3000)})
    D = AntsDomain('core')
    bg = REPO / 'dataset/ants/eci/fg448/background'
    ranges = obs_rows(REPO / 'dataset/ants/eci/annotations.csv')
    rule = RULES['ants']
    picks = [o for o in sorted(ranges, key=lambda o: ranges[o][0])][::40]
    for o in picks:
        lo, hi = ranges[o]
        b = load_background(bg, o, rule)
        rows = np.arange(lo, hi)
        old = segment_of(rows, b['rows'], b['seg_bounds'])
        bgs = FgBackgrounds(bg, {o: (lo, hi)}, rule, 'cpu')
        new = bgs.segment(rows, 0, b)
        shift = 10 ** 6 + 17  # the same video at other rows of a new annotations.csv
        bgs2 = FgBackgrounds(bg, {o: (lo + shift, hi + shift)}, rule, 'cpu')
        moved = bgs2.segment(rows + shift, int(bgs2.window_index([lo + shift])[0]), b)
        check(f'{o}: relative time blocks = absolute ones (rows {lo}-{hi}), and after a move',
              (old == new).all() and (old == moved).all() and b['rows'][0] == lo, np.bincount(old).tolist())
    check('windows.json absent in the 10-min background dir (one background per video, as before)',
          FgBackgrounds(bg, {picks[0]: ranges[picks[0]]}, rule, 'cpu').window is None)
    del D


def run_rowmap(new, old):
    from fg_encode_extend import row_map, runs
    old_row, a = row_map(new, old)
    o = pd.read_csv(old, usecols=['frame_path'])['frame_path'].values
    exp = a['frame_path'].str.split('/').str[1].values
    fi = a['frame_idx'].values
    r = old_row >= 0
    for e in pd.unique(exp):
        m = exp == e
        print(f'  {e}: rows {m.sum():,}, reused {(m & r).sum():,} (frame_idx < 3000: {(m & (fi < 3000)).sum():,}), '
              f'to encode {(m & ~r).sum():,}')
    check('every reused row has the same frame_path in both layouts', (o[old_row[r]] == a['frame_path'].values[r]).all())
    check('every frame_idx < 3000 row of v2 / v3 is reused, nothing else is',
          np.array_equal(r, np.isin(exp, ('v2', 'v3')) & (fi < 3000)))
    R = runs(old_row)
    cover = np.zeros(len(old_row), bool)
    for nlo, olo, k in R:
        cover[nlo:nlo + k] = True
        assert (old_row[nlo:nlo + k] == np.arange(olo, olo + k)).all()
    check('the copy runs cover exactly the reused rows, each contiguous in both layouts (consecutive videos merge)',
          np.array_equal(cover, r), f'{len(R)} runs for {a.loc[r, "observation_id"].nunique()} videos')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('check', choices=('contrasts', 'windows', 'rowmap', 'atlas'))
    ap.add_argument('--new', default=None)
    ap.add_argument('--old', default=str(REPO / 'dataset/ants/eci/annotations.csv'))
    a = ap.parse_args()
    {'contrasts': run_contrasts, 'windows': run_windows, 'atlas': run_atlas, 'rowmap': lambda: run_rowmap(a.new, a.old)}[a.check]()
    print(f'{len(FAIL)} FAIL' + (f': {FAIL}' if FAIL else ''))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
