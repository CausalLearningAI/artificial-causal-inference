"""
CelebA DINES benchmark, step 3 (CPU job): metrics tables.

  --published   read the published NES selections (key 'Bonferroni_NES' of Tommaso's sneurons_{seed}.json, 26 MB each)
                once and keep only the selections in results/eci/dines_celeba/published_nes.json; print the sanity
                reproduction (precision at tau = 0.7, exact recovery pooled over tau = 0.1..0.8 at N = 5000).
  (default)     tables for every design from the per-trial JSONs (run_trials.py) -> results/eci/dines_celeba/
                summary_<design>.csv and the markdown tables in summary.md.

Metrics exactly as notebooks/rebuttal/SAE_ablation_update.ipynb (cell 10, _set_metrics; cell 22): per trial, set
precision (0 for an empty prediction), recall, IoU against the ground-truth atoms, then the mean over trials. Exact
recovery = prediction equal to the truth. Original design truth = {"38", "6051"} (hat, eyeglasses); new designs: the
principal atom(s) of the direct attribute(s) (designs_meta.json), plus the selection rates of the indirect atom B and
the spouse atom W. False-positive rate at tau = 0 = share of runs with a non-empty selection.
DINES is scored on its REPORTED set (kept + undetermined, theory Section 2.5) and, as '<arm>_kept', on kept only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path('/nfs/scistore19/locatgrp/tmencatt/projects/neuralscience/results/experiments-semi-synthetic')
OUT = Path('results/eci/dines_celeba')
TAUS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
GT_ORIGINAL = {'38', '6051'}
METHODS = ['DINES_d0.0', 'DINES_d0.02', 'DINES_d0.05', 'DINES_d0.0_kept', 'DINES_d0.02_kept', 'DINES_d0.05_kept',
           'NES_published', 'NES_repo']


def set_metrics(pred, gt):
    """_set_metrics of the notebook (cell 10)."""
    inter, union = len(pred & gt), len(pred | gt)
    p = inter / len(pred) if len(pred) else 0.0
    r = inter / len(gt) if len(gt) else 0.0
    return {'precision': p, 'recall': r, 'iou': inter / union if union else 0.0, 'exact': float(pred == gt),
            'size': len(pred), 'any': float(len(pred) > 0)}


def published():
    sel = {}
    for seed in range(420, 430):
        d = json.loads((SRC / f'sneurons_{seed}.json').read_text())
        for key, v in d.items():
            raw = v['Bonferroni_NES']
            sel[key] = [str(x) for x in (raw.get('neurons', []) if isinstance(raw, dict) else raw)]
    (OUT / 'published_nes.json').write_text(json.dumps(sel))
    rows = []
    for key, s in sel.items():
        _, ate, size, seed = key.split('_')
        rows.append({'tau': float(ate.split(':')[1]), 'N': int(size.split(':')[1]), 'seed': int(seed.split(':')[1]),
                     **set_metrics(set(s), GT_ORIGINAL)})
    df = pd.DataFrame(rows)
    print(f'{len(df)} published trials')
    print('tau = 0.7, mean over 10 seeds:')
    print(df[df.tau == 0.7].groupby('N')[['precision', 'recall', 'iou', 'size', 'exact']].mean().round(3).to_string())
    pooled = df[df.tau > 0].groupby('N').agg(precision=('precision', 'mean'), size=('size', 'mean'),
                                             exact=('exact', 'sum'), runs=('exact', 'size'))
    print('pooled over tau = 0.1..0.8:')
    print(pooled.round(3).to_string())
    print('tau = 0, runs with any selection:', df[df.tau == 0].groupby('N')['any'].sum().to_dict())


def load_trials(design):
    pub = json.loads((OUT / 'published_nes.json').read_text()) if design == 'original' else None
    meta = None if design == 'original' else json.loads((OUT / 'designs_meta.json').read_text())[design]
    gt = GT_ORIGINAL if meta is None else set(meta['roles']['direct'])
    rows = []
    for f in sorted((OUT / 'trials' / design).glob('*.json')):
        rec = json.loads(f.read_text())
        meths = dict(rec['methods'])
        for name in [k for k in meths if k.startswith('DINES') and 'kept' in meths[k]]:
            meths[f'{name}_kept'] = {**meths[name], 'selected': meths[name]['kept'], 'derived': True}
        if pub is not None:
            meths['NES_published'] = {'selected': pub[f"M3_ATE:{rec['tau']}_size:{rec['N']}_seed:{rec['seed']}"]}
        for name, m in meths.items():
            pred = set(m['selected'])
            row = {'design': design, 'tau': rec['tau'], 'N': rec['N'], 'seed': rec['seed'], 'method': name,
                   **set_metrics(pred, gt), 'false': ','.join(sorted(pred - gt)), 'error': 'error' in m, 'elapsed_s': m.get('elapsed_s', np.nan),
                   'forward_size': len(m['forward_set']) if 'forward_set' in m else np.nan,
                   'undetermined': len(m.get('undetermined', [])), 'set_aside': len(m.get('set_aside', [])),
                   'complete': m.get('terminal_complete', True),
                   'capped': m.get('forward_capped', False)}
            if meta is not None:
                for role in ('indirect', 'spouse'):
                    atoms = set(meta['roles'].get(role, []))
                    row[f'{role}_sel'] = float(len(atoms & pred) > 0) if atoms else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _pivot(df, value, agg='mean'):
    t = df.pivot_table(index='method', columns='N', values=value, aggfunc=agg)
    return t.reindex([m for m in METHODS if m in t.index])


def tables(design):
    df = load_trials(design)
    df.to_csv(OUT / f'summary_{design}.csv', index=False)
    md = [f'## design {design}', '', f'trials: {df.groupby("method").size().to_dict()}; errors: '
          f'{int(df["error"].sum())}; DINES terminal incomplete: {int((~df["complete"].astype(bool)).sum())}; '
          f'forward capped: {int(df["capped"].astype(bool).sum())}; runs with undetermined atoms: '
          f'{int((df["undetermined"] > 0).sum())}', '']
    t07 = df[np.isclose(df.tau, 0.7)]
    pooled = df[df.tau > 0.05]
    for title, sub in (('tau = 0.7', t07), ('pooled over tau = 0.1..0.8', pooled)):
        for value in ('precision', 'recall', 'iou', 'size'):
            md += [f'### {value}, {title} (mean over runs)', '', _pivot(sub, value).round(3).to_markdown(), '']
        md += [f'### exact recovery count, {title}', '', _pivot(sub, 'exact', 'sum').astype(int).to_markdown(), '',
               f'runs per cell: {int(sub.groupby(["method", "N"]).size().max())}', '']
        if 'indirect_sel' in sub:
            for role in ('indirect', 'spouse'):
                if sub[f'{role}_sel'].notna().any():
                    md += [f'### {role} atom selected (share of runs), {title}', '',
                           _pivot(sub, f'{role}_sel').round(3).to_markdown(), '']
    big = pooled[pooled.N == pooled.N.max()]
    md += [f'### most frequent false atoms at N = {int(pooled.N.max())}, pooled over tau = 0.1..0.8 (runs out of '
           f'{int(big.groupby("method").size().max())})', '']
    for meth in [m for m in METHODS if m in set(big.method)]:
        cnt = big[big.method == meth]['false'].str.split(',').explode()
        cnt = cnt[cnt.astype(bool)].value_counts().head(6)
        md.append(f'- {meth}: ' + ', '.join(f'{a} ({c})' for a, c in cnt.items()))
    md.append('')
    null = df[df.tau == 0]
    md += ['### tau = 0: share of runs with any selection (false-positive rate)', '', _pivot(null, 'any').round(3).to_markdown(), '']
    rt = df[(df.method != 'NES_published') & ~df.method.str.endswith('_kept')]
    md += ['### runtime per trial and method (s; median, max)', '',
           rt.pivot_table(index='method', columns='N', values='elapsed_s', aggfunc='median').round(1).to_markdown(), '',
           rt.pivot_table(index='method', columns='N', values='elapsed_s', aggfunc='max').round(1).to_markdown(), '']
    fw = df[df.method.str.startswith('DINES') & ~df.method.str.endswith('_kept')]
    md += ['### DINES forward-set size |F| (mean, all tau)', '', _pivot(fw, 'forward_size').round(2).to_markdown(), '',
           '### DINES undetermined atoms per run (mean, all tau)', '', _pivot(fw, 'undetermined').round(2).to_markdown(), '',
           '### DINES set-aside atoms per run (mean, all tau)', '', _pivot(fw, 'set_aside').round(2).to_markdown(), '',
           '### DINES runs with an incomplete terminal phase (count, all tau)', '',
           fw.assign(inc=~fw['complete'].astype(bool)).pivot_table(index='method', columns='N', values='inc', aggfunc='sum').to_markdown(), '']
    return '\n'.join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--published', action='store_true')
    ap.add_argument('--designs', nargs='*', default=['original', 'M', 'S', 'H', 'I'])
    args = ap.parse_args()
    if args.published:
        published()
        return
    parts = [tables(d) for d in args.designs if (OUT / 'trials' / d).exists()]
    (OUT / 'summary.md').write_text('\n\n'.join(parts))
    print('\n\n'.join(parts))


if __name__ == '__main__':
    main()
