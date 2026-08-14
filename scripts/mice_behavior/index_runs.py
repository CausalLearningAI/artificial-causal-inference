"""One table over every mice frame-classifier run in results/vision/mice/frame.

There are 70+ run directories in a flat namespace, written across several naming eras
('res448', 'd4_decay40_res448', '448_best_s1', 'ft_b4'), and a run's config and its scores
live only in its own config.json. So "which variant won, and what was it" is currently a
question you answer by opening files one at a time.

THE POINT OF THIS SCRIPT is the grouping, not the table. Those runs sit on FOUR DIFFERENT
VALIDATION SPLITS, and numbers from different splits are not comparable at all:

    current   rd11_2 rd13 rd14 rd18        <- src.mice_behavior.pools.VAL_POOLS_V1, use this
    5-pool    rd11_2 rd32 rd34 rd35_2 rd41_3
    4x4-era   rd14 rd19 rd29 rd35_3
    unknown   (val_pools never recorded)

Ranking across those groups is what makes this directory confusing -- e.g. the old '504 beats
448 by 4.7%' finding is a 5-pool-split result and does not transfer to the current split. The
table therefore ranks WITHIN each split and refuses to sort across them.

Two further honesty flags the raw numbers do not carry:
  * calib= column. rate_report fit its affine calibration on the EVALUATION fold's own labels
    until 2026-08-14, so 'oracle' rows have optimistic mae_calibrated / mae_vs_baseline.
  * auc~ is the per-epoch MONITOR value from history, not full-val -- full-val ROC-AUC is
    printed to the job log but never persisted into config.json.

Runs from before 2026-08-12 use an older config schema whose headline number is `best_ap`
(best MONITOR macro AP), not the full-val `ap_report` the current schema stores. That is a
different quantity on a different split, so it is shown in its own column, never merged.

Deliberately read-only: 10+ scripts resolve checkpoints as gsf.FRAME_DIR / '<run name>', so
renaming or moving these directories would break them. This adds a view, not a layout.

Usage:
    python scripts/mice_behavior/index_runs.py                  # grouped, best first
    python scripts/mice_behavior/index_runs.py --split current  # only comparable runs
    python scripts/mice_behavior/index_runs.py --pattern 'ft_*' # the fine-tuning sweep
    python scripts/mice_behavior/index_runs.py --sort r_nt      # rank by downstream r
    python scripts/mice_behavior/index_runs.py --full           # include bak/smoke/empty
"""
import argparse
import csv
import fnmatch
import json
from pathlib import Path

FRAME_DIR = Path('results/vision/mice/frame')
OUT_CSV = FRAME_DIR / 'runs_index.csv'
PREFIX = 'patchgrid256_dinov2_'

SPLITS = {
    ('rd11_2', 'rd13', 'rd14', 'rd18'): 'current',
    ('rd11_2', 'rd32', 'rd34', 'rd35_2', 'rd41_3'): '5-pool',
    ('rd14', 'rd19', 'rd29', 'rd35_3'): '4x4-era',
}
SPLIT_ORDER = ['current', '5-pool', '4x4-era', 'unknown']


def split_of(cfg: dict) -> str:
    return SPLITS.get(tuple(cfg.get('val_pools') or ()), 'unknown')


def classify(name: str, cfg: dict | None) -> str:
    if cfg is None:
        return 'no-results'
    if name.endswith('_bak') or '_before_' in name:
        return 'backup'
    if 'smoke' in name:
        return 'smoke'
    return 'run'


def get(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def describe(cfg: dict) -> str:
    """The config in one column -- only the axes that actually varied across these runs."""
    bits = [f"{cfg.get('input_size') or '?'}px", f"k{cfg.get('context_k', '?')}"]
    if (cfg.get('stride') or 1) != 1:
        bits.append(f"s{cfg['stride']}")
    nb = cfg.get('unfreeze_blocks') or 0
    mode = 'bit' if cfg.get('ft_mode') == 'bitfit' else 'ft'
    bits.append(f"{mode}{nb}@{cfg['encoder_lr']:g}" if nb else 'frozen')
    if cfg.get('patch_selfattn_dim'):
        bits.append(f"sa{cfg['patch_selfattn_dim']}")
    if (cfg.get('pool_queries') or 1) != 1:
        bits.append(f"q{cfg['pool_queries']}")
    if cfg.get('optimizer'):
        bits.append(cfg['optimizer'])
    aug = cfg.get('augment')
    if aug:
        bits.append(aug + (f"@{cfg.get('photo_strength', 1.0):g}" if aug == 'd4_photo' else ''))
    if cfg.get('warmup_epochs'):
        bits.append(f"wu{cfg['warmup_epochs']}")
    if cfg.get('neg_ratio') not in (None, 1):
        bits.append(f"neg{cfg['neg_ratio']}")
    return ' '.join(str(b) for b in bits)


def n_epochs(d: Path, cfg: dict):
    """Current schema stores history inline (list of dicts); the old one uses a column-oriented
    history.json. Return (n_epochs, best_epoch) from whichever exists."""
    hist = cfg.get('history')
    if isinstance(hist, list) and hist:
        best = max(range(len(hist)), key=lambda i: hist[i].get('monitor_ap', -1)) + 1
        return len(hist), best, hist[-1]
    hj = d / 'history.json'
    if hj.exists():
        h = json.load(open(hj))
        eps = h.get('epoch') or []
        key = next((k for k in ('macro_ap', 'monitor_ap', 'val_ap') if k in h), None)
        best = (max(range(len(h[key])), key=lambda i: h[key][i]) + 1) if key and h.get(key) else None
        return (len(eps) or None), best, {}
    return None, None, {}


def collect(full: bool, pattern: str | None) -> list[dict]:
    rows = []
    for d in sorted(FRAME_DIR.iterdir()):
        if not d.is_dir():
            continue
        name = d.name[len(PREFIX):] if d.name.startswith(PREFIX) else d.name
        if pattern and not fnmatch.fnmatch(name, pattern):
            continue
        cfg_f = d / 'config.json'
        cfg = json.load(open(cfg_f)) if cfg_f.exists() else None
        kind = classify(d.name, cfg)
        if not full and kind not in ('run',):
            continue
        if cfg is None:
            rows.append({'run': name, 'kind': kind, 'split': '-', 'config': '-',
                         'note': 'no config.json (running, failed, or an aggregate dir)'})
            continue
        eps, best_ep, last = n_epochs(d, cfg)
        rate = cfg.get('rate_report') or {}
        rows.append({
            'run': name, 'kind': kind, 'split': split_of(cfg), 'config': describe(cfg),
            'epochs': eps, 'best_ep': best_ep,
            'macroAP0': get(cfg, 'ap_report', 'macro/tol0', 'ap'),
            'macroAP1': get(cfg, 'ap_report', 'macro/tol1', 'ap'),
            'ap_nt': get(cfg, 'ap_report', 'nt/tol0', 'ap'),
            'ap_nn': get(cfg, 'ap_report', 'nn/tol0', 'ap'),
            'monAP': cfg.get('best_ap'),          # old schema headline: best MONITOR macro AP
            'auc_nt~': last.get('auc_nt'), 'auc_nn~': last.get('auc_nn'),
            'r_nt': get(rate, 'nt', 'pearson_r'), 'r_nn': get(rate, 'nn', 'pearson_r'),
            'p_nt': get(rate, 'nt', 'pearson_p'),
            'ppi_nt': get(rate, 'nt', 'ppi_variance_reduction'),
            'calib': (get(rate, 'nt', 'calibration_source') or ('oracle*' if rate else '-')).split(' ')[0],
            'note': '',
        })
    return rows


COLS = [('run', 34, 's'), ('config', 30, 's'), ('ep', 4, 'd'), ('bst', 4, 'd'),
        ('macroAP0', 9, '.4f'), ('macroAP1', 9, '.4f'), ('ap_nt', 7, '.4f'), ('ap_nn', 7, '.4f'),
        ('monAP', 7, '.4f'), ('auc_nt~', 8, '.3f'), ('r_nt', 7, '.3f'), ('r_nn', 7, '.3f'),
        ('ppi_nt', 7, '.2f'), ('calib', 9, 's')]
ALIAS = {'ep': 'epochs', 'bst': 'best_ep'}


def render(rows, sort_key):
    hdr = ' '.join(f'{c:>{w}}' for c, w, _ in COLS)
    print(hdr); print('-' * len(hdr))
    scored = [r for r in rows if isinstance(r.get(sort_key), (int, float))]
    rest = [r for r in rows if not isinstance(r.get(sort_key), (int, float))]
    scored.sort(key=lambda r: -r[sort_key])
    for r in scored + rest:
        cells = []
        for c, w, fmt in COLS:
            v = r.get(ALIAS.get(c, c))
            cells.append(f'{str(v):>{w}}'[:w] if fmt == 's' and v is not None
                         else f'{"-":>{w}}' if v is None else f'{v:>{w}{fmt}}')
        print(' '.join(cells))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sort', default='macroAP0')
    p.add_argument('--pattern', default=None)
    p.add_argument('--split', default=None, choices=SPLIT_ORDER)
    p.add_argument('--full', action='store_true')
    p.add_argument('--csv', nargs='?', const=str(OUT_CSV), default=None,
                    help='also write the table as CSV. Off by default: this is derived from the '
                         'config.json files and regenerates in under a second, so a copy left '
                         'lying in the results directory is just one more thing that can go '
                         'stale and be mistaken for a source of truth.')
    args = p.parse_args()

    rows = collect(args.full, args.pattern)
    if not rows:
        print('no runs matched'); return

    by_split = {}
    for r in rows:
        by_split.setdefault(r.get('split', '-'), []).append(r)

    for s in SPLIT_ORDER + [k for k in by_split if k not in SPLIT_ORDER]:
        if s not in by_split or (args.split and s != args.split):
            continue
        grp = by_split[s]
        tag = {'current': '  <- VAL_POOLS_V1, the only mutually comparable group',
               '5-pool': '  <- pre-2026-08-12, NOT comparable to current',
               '4x4-era': '  <- coarse 4x4 grid era, NOT comparable to current',
               'unknown': '  <- val_pools never recorded, provenance unclear'}.get(s, '')
        print(f'\n=== split: {s}  ({len(grp)} runs){tag}')
        render(grp, args.sort)

    print('\nranked WITHIN each split only -- cross-split comparison is invalid.')
    print('monAP = old-schema headline (best MONITOR macro AP), a different quantity from the '
          'current schema\'s full-val macroAP0. Never compare the two columns.')
    print('~ auc is the per-epoch monitor value from history, not full-val (never persisted).')
    print("calib=oracle -> that run's mae_calibrated/mae_vs_baseline were fit on the eval fold "
          "itself and are optimistic; 'oracle*' predates the flag entirely.")

    if args.csv:
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
        print(f'wrote {args.csv}')


if __name__ == '__main__':
    main()
