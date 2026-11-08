"""
ECI v2.7 scoring = the v2.5 code with paths redirected to v2.7 (pre-registration scripts/eci/v2/v27_prereg.md); rules auc /
dist / dist2. Models: PREDc / PREDd / PREDk / PREDh (seeds 2701-2703), V s1001-1006 and v1 s0 (v2.5 edge-excluded caches,
re-tagged with dist2), PRED26 = the stored v2.6 PRED s2601-2603 (v2.6 caches, readout-only rescore with dist2).
usage: v27_eval.py setup | <v25_eval step> --dom D [--key K] | codes / merge (v25_train steps)
"""
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v25_train as TR  # noqa: E402
import v27_train as T7  # noqa: E402

TR.BEE25, TR.OUT25 = T7.BEE27, T7.OUT27
for _a in T7.ARMS + ('PRED26',):
    TR.ARMS[_a], TR.DIN[_a] = ('N', 0.0), 1024
TR.SEEDS, TR.MAIN_ARMS = T7.SEEDS, T7.ARMS
import v25_eval as EV  # noqa: E402

EV.RULES = ('auc', 'dist', 'dist2')
EV.E.RULES = EV.RULES
EV.MODELS['PRED26'] = [f'PRED26_s{s}' for s in (2601, 2602, 2603)]
NEW = T7.ARMS
R3 = EV.RULES
EV.EXTRA_PAIRS_MICE = [(f'{x}|{r}', f'PREDc|{r}', True) for r in R3 for x in NEW if x != 'PREDc'] + \
    [(f'{x}|{r}', 'V|auc', False) for r in R3 for x in NEW + ('PRED26',)] + [(f'{x}|{r}', 'v1s0|auc', False) for r in R3 for x in NEW + ('PRED26',)] + \
    [(f'{x}|dist2', f'{x}|dist', True) for x in NEW + ('PRED26',)]
EV.EXTRA_PAIRS_ANTS = [(x, 'PREDc', True) for x in NEW if x != 'PREDc'] + [(x, 'V', False) for x in NEW] + \
    [(x, 'v1s0', False) for x in NEW] + [(x, 'v1', False) for x in NEW]


def setup():
    """rows / counts from v2.5; V / v1 s0 mice caches and v2.6 PRED caches (as PRED26) linked into v2.7."""
    b5 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_5')
    b6 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_6')
    for dom in ('mice', 'ants'):
        od = T7.BEE27 / dom / 'codes'
        od.mkdir(parents=True, exist_ok=True)
        for f in ('rows.parquet', 'rows_all.parquet'):
            shutil.copy(b5 / dom / 'codes' / f, od / f)
        T7.OUT27.mkdir(parents=True, exist_ok=True)
        shutil.copy(T7.REPO / 'results/vision/eci_v2_5' / f'{dom}_count.json', T7.OUT27 / f'{dom}_count.json')
    cd = T7.BEE27 / 'mice' / 'cache'
    cd.mkdir(parents=True, exist_ok=True)
    links = {f'{k}{suf}': b5 / 'mice' / 'cache' / f'{k}{suf}' for k in [f'V_s{s}' for s in range(1001, 1007)] + ['v1_s0'] for suf in ('_raw.npy', '_pon.npy')}
    links.update({f'PRED26_s{s}{suf}': b6 / 'mice' / 'cache' / f'PRED_s{s}{suf}' for s in (2601, 2602, 2603) for suf in ('_raw.npy', '_pon.npy')})
    for n, src in links.items():
        if src.exists() and not (cd / n).exists():
            (cd / n).symlink_to(src)
    sd = T7.BEE27 / 'mice' / 'sae'
    sd.mkdir(parents=True, exist_ok=True)
    for s in (2601, 2602, 2603):
        if not (sd / f'PRED26_s{s}').exists():
            (sd / f'PRED26_s{s}').symlink_to(b6 / 'mice' / 'sae' / f'PRED_s{s}')
    print('setup done')


if __name__ == '__main__':
    import argparse
    if sys.argv[1] == 'setup':
        setup()
    elif sys.argv[1] in ('codes', 'merge'):
        p = argparse.ArgumentParser()
        p.add_argument('step')
        p.add_argument('--dom')
        p.add_argument('--keys', default='')
        p.add_argument('--task', type=int, default=0)
        p.add_argument('--ntask', type=int, default=1)
        a = p.parse_args()
        (TR.cmd_codes if a.step == 'codes' else TR.cmd_merge)(a)
    else:
        p = argparse.ArgumentParser()
        p.add_argument('step')
        p.add_argument('--dom', default='mice')
        p.add_argument('--key', default='')
        a = p.parse_args()
        {'cache': EV.cmd_cache, 'tag': EV.cmd_tag, 'prep': EV.cmd_prep, 'report': EV.cmd_report, 'fwdcheck': EV.cmd_fwdcheck}[a.step](a)
