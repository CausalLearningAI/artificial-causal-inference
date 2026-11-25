"""
ECI v2.9 scoring = the v2.5 code with paths redirected to v2.9 (as v27_eval for v2.7; pre-registration
scripts/eci/v2/v29_prereg.md). Rules auc / dist / dist2 (official v2.7 rules, continuity); rule sgap is in v29_sgap.py.
Models: PREDk8 / PREDk16 / PREDk32 (v2.9), PREDk = the v2.7 PREDk s2701-2703, V s1001-1006 and v1 s0 (v2.7 caches and tags,
linked). Ants report: v1 s1-s10 dropped (heavy NFS re-reads in v2.7; v1 s0 kept).
usage: v29_eval.py setup | codes / merge --dom D --keys ... | <v25_eval step> --dom D [--key K]
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v29_train as T9  # noqa: E402

TR, T7 = T9.TR, T9.T7
TR.BEE25, TR.OUT25 = T9.BEE29, T9.OUT29
NEW = ('PREDk16', 'PREDk8', 'PREDk32')
for _a in NEW + ('PREDk',):
    TR.ARMS[_a], TR.DIN[_a] = ('N', 0.0), 1024
TR.SEEDS, TR.MAIN_ARMS = (2901, 2902, 2903), NEW
import v25_eval as EV  # noqa: E402

EV.RULES = ('auc', 'dist', 'dist2')
EV.E.RULES = EV.RULES
OLD = [f'PREDk_s{s}' for s in (2701, 2702, 2703)]
EV.MODELS = {'V': [f'V_s{s}' for s in range(1001, 1007)], 'v1s0': ['v1_s0'], 'PREDk': OLD,
             'PREDk16': [f'PREDk16_s{s}' for s in range(2901, 2907)], 'PREDk8': [f'PREDk8_s{s}' for s in (2901, 2902, 2903)],
             'PREDk32': [f'PREDk32_s{s}' for s in (2901, 2902, 2903)]}
EV.MODELS_ANTS = {k: v for k, v in EV.MODELS.items()}
R3 = EV.RULES
EV.EXTRA_PAIRS_MICE = [(f'{x}|{r}', f'PREDk16|{r}', False) for r in R3 for x in ('PREDk8', 'PREDk32', 'PREDk')]
EV.EXTRA_PAIRS_ANTS = [(x, 'PREDk16', False) for x in ('PREDk8', 'PREDk32', 'PREDk')]
B5 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_5')
B7 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_7')
O7 = T7.OUT27


def setup():
    """rows / counts from v2.5 (as v2.7); caches, tags and prep pickles of V s1001-1006, v1 s0 and the v2.7 PREDk linked."""
    import shutil
    for dom in ('mice', 'ants'):
        od = T9.BEE29 / dom / 'codes'
        od.mkdir(parents=True, exist_ok=True)
        for f in ('rows.parquet', 'rows_all.parquet'):
            shutil.copy(B5 / dom / 'codes' / f, od / f)
        T9.OUT29.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO5 / f'{dom}_count.json', T9.OUT29 / f'{dom}_count.json')
        cd = T9.BEE29 / dom / 'cache'
        cd.mkdir(parents=True, exist_ok=True)
        keys = OLD + ([f'V_s{s}' for s in range(1001, 1007)] + ['v1_s0'] if dom == 'mice' else [])
        for k in keys:
            for suf in ('_raw.npy', '_pon.npy'):
                src = B7 / dom / 'cache' / f'{k}{suf}'
                if src.exists() and not (cd / f'{k}{suf}').exists():
                    (cd / f'{k}{suf}').symlink_to(src.resolve())
        sd = T9.BEE29 / dom / 'sae'
        sd.mkdir(parents=True, exist_ok=True)
        for k in OLD:
            if not (sd / k).exists():
                (sd / k).symlink_to(B7 / dom / 'sae' / k)
        for sub in ('tags',) + (('work',) if dom == 'mice' else ()):
            td = T9.OUT29 / dom / sub
            td.mkdir(parents=True, exist_ok=True)
            for k in keys:
                src = O7 / dom / sub / f'{k}.pkl'
                if src.exists() and not (td / f'{k}.pkl').exists():
                    (td / f'{k}.pkl').symlink_to(src)
    print('setup done', flush=True)


REPO5 = T7.REPO / 'results/vision/eci_v2_5'

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
        if a.step == 'codes':
            TR.cmd_codes(a)
        else:
            TR.cmd_merge(a)
            od = T9.BEE29 / a.dom / 'codes' / 'parts'
            for k in a.keys.split(','):  # merged -> drop the per-task parts (many files)
                for t in range(a.ntask):
                    f = od / f'{k}_t{t}.npy'
                    if f.exists():
                        os.remove(f)
            print('parts removed', flush=True)
    else:
        p = argparse.ArgumentParser()
        p.add_argument('step')
        p.add_argument('--dom', default='mice')
        p.add_argument('--key', default='')
        a = p.parse_args()
        {'cache': EV.cmd_cache, 'tag': EV.cmd_tag, 'prep': EV.cmd_prep, 'report': EV.cmd_report, 'fwdcheck': EV.cmd_fwdcheck}[a.step](a)
