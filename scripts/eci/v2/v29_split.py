"""
ECI v2.9 ADDITION (POST HOC, requested after the pre-registration; not part of its decisions): the v2.8 random-split check
(v28_split_neurons.py, unchanged: same 50 annotator-stratified random pool splits, seed 20261125, leave-one-annotator-out,
rule sgap with the firing cap) run on the v2.9 dictionaries. Only the cache paths and the output folder change.
usage: v29_split.py --key K   -> results/vision/eci_v2_9/split/work/K.pkl
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v28_split_neurons as SN  # noqa: E402

BEE9 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_9/mice/cache')
_paths0 = SN.A.mice_paths
SN.A.mice_paths = lambda key: ((BEE9 / f'{key}_raw.npy', BEE9 / f'{key}_pon.npy') if key.startswith('PREDk') and '_s29' in key else _paths0(key))
SN.WORK = SN.A.REPO / 'results/vision/eci_v2_9/split/work'

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--key', required=True)
    SN.main(p.parse_args())
