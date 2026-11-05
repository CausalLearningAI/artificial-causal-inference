"""
ECI v2.6 scoring = the v2.5 code with paths redirected to v2.6 (pre-registration scripts/eci/v2/v26_prereg.md):
v25_train.BEE25 / OUT25 -> eci_codes/v2_6, results/vision/eci_v2_6 (codes, merge, cache, tag, prep, report, fwdcheck, sheets);
arms N0wc / PRED / SMOOTH / RW / N0x registered as static inputs (x_t); seeds 2601-2603. Baselines V s1001-1006 / v1 s0:
the v2.5 edge-excluded mice caches + tags (linked / copied by step baselines); ants V / v1: v2.3 tags (unchanged frame set).
usage: v26_eval.py <v25_eval step> --dom D [--key K] | v26_eval.py baselines | v26_eval.py codes|merge ... (v25_train steps)
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v25_train as TR  # noqa: E402
import v26_train as T6  # noqa: E402

TR.BEE25, TR.OUT25 = T6.BEE26, T6.OUT26
for _a in T6.ARMS:
    TR.ARMS[_a], TR.DIN[_a] = ('N', 0.0), 1024
TR.SEEDS, TR.MAIN_ARMS = T6.SEEDS, T6.ARMS
import v25_eval as EV  # noqa: E402

NEW = T6.ARMS
EV.EXTRA_PAIRS_MICE = [(f'{x}|{r}', f'N0wc|{r}', True) for r in EV.RULES for x in NEW if x != 'N0wc'] + \
    [(f'{x}|{r}', 'V|auc', False) for r in EV.RULES for x in NEW] + [(f'{x}|{r}', 'v1s0|auc', False) for r in EV.RULES for x in NEW]
EV.EXTRA_PAIRS_ANTS = [(x, 'N0wc', True) for x in NEW if x != 'N0wc'] + [(x, 'V', False) for x in NEW] + \
    [(x, 'v1s0', False) for x in NEW] + [(x, 'v1', False) for x in NEW]


def baselines():
    """V / v1 s0 mice caches (symlinks), tags and prep pickles (copies) from v2.5 -> v2.6 folders."""
    b5 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_5/mice/cache')
    o5 = T6.REPO / 'results/vision/eci_v2_5/mice'
    (T6.BEE26 / 'mice' / 'cache').mkdir(parents=True, exist_ok=True)
    for k in [f'V_s{s}' for s in range(1001, 1007)] + ['v1_s0']:
        for suf in ('_raw.npy', '_pon.npy'):
            src, dst = b5 / f'{k}{suf}', T6.BEE26 / 'mice' / 'cache' / f'{k}{suf}'
            if src.exists() and not dst.exists():
                dst.symlink_to(src)
        for sub in ('tags', 'work'):
            (EV.MICE_OUT / sub).mkdir(parents=True, exist_ok=True)
            shutil.copy(o5 / sub / f'{k}.pkl', EV.MICE_OUT / sub / f'{k}.pkl')
    print('baselines linked / copied')


if __name__ == '__main__':
    if sys.argv[1] == 'baselines':
        baselines()
    elif sys.argv[1] in ('codes', 'merge'):
        sys.argv = [sys.argv[0]] + sys.argv[1:]
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument('step')
        p.add_argument('--dom')
        p.add_argument('--keys', default='')
        p.add_argument('--task', type=int, default=0)
        p.add_argument('--ntask', type=int, default=1)
        a = p.parse_args()
        (TR.cmd_codes if a.step == 'codes' else TR.cmd_merge)(a)
    else:
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument('step')
        p.add_argument('--dom', default='mice')
        p.add_argument('--key', default='')
        a = p.parse_args()
        {'cache': EV.cmd_cache, 'tag': EV.cmd_tag, 'prep': EV.cmd_prep, 'report': EV.cmd_report, 'fwdcheck': EV.cmd_fwdcheck}[a.step](a)
