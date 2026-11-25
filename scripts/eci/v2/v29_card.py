"""
ECI v2.9 ADDITION (POST HOC): alignment cards (v28_card.py, unchanged) for v2.9 dictionaries: only the cache paths
(PREDk8 / PREDk16 / PREDk32 s29xx -> eci_codes/v2_9) and the output folder (results/vision/eci_v2_9/figs) change.
usage: same arguments as v28_card.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v28_audit_neurons as A  # noqa: E402
import v28_card as K  # noqa: E402
import v28_sheets as SH  # noqa: E402

_init0 = SH.init
SH.init = lambda ver: _init0('29')  # v28_card calls SH.init('27'); v2.9 arms / paths need v29_eval

BEE9 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_9')
_m0, _a0 = A.mice_paths, A.ants_paths


def _new(key):
    return key.startswith('PREDk') and '_s29' in key


A.mice_paths = lambda key: ((BEE9 / 'mice/cache' / f'{key}_raw.npy', BEE9 / 'mice/cache' / f'{key}_pon.npy') if _new(key) else _m0(key))
A.ants_paths = lambda key: ((BEE9 / 'ants/cache' / f'{key}_raw.npy', BEE9 / 'ants/cache' / f'{key}_pon.npy') if _new(key) else _a0(key))
K.OUT = K.REPO / 'results/vision/eci_v2_9/figs'

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('data', 'plot', 'all'))
    p.add_argument('--dom', choices=('mice', 'ants'))
    p.add_argument('--beh')
    p.add_argument('--rows', default='')
    p.add_argument('--out', required=True)
    p.add_argument('--no-frames', action='store_true')
    p.add_argument('--dpi', type=int, default=80)
    a = p.parse_args()
    if a.step in ('data', 'all'):
        K.cmd_data(a)
    if a.step in ('plot', 'all'):
        K.cmd_plot(a)
