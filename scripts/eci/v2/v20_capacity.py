"""
ECI v2.0: label-free behaviour-neuron suite (scripts/eci/v2_behav/behav.py, amendment 2, checks C1-C8, its probe set)
for the pre-registered 5 fps vs 1 fps comparison (scripts/eci/v2/v20_eval.py). behav.py is run UNCHANGED with
  - its own outputs redirected (BEHAV_OUT = results/vision/eci_v2_0/behav, BEHAV_BEE = eci_codes/v2_0/behav; the probe
    tables are copied, the probe token stores symlinked: no file of the original study is written),
  - arms_of() = {V2F_N2 900-904 (mice / ants) | V20_1fps 900-902 (frogs, the v2.0 frogs SAE), V20F5 950-954},
  - deploy_seed() = the highest held-out FVE of each arm (label-free rule).
Frogs: the probe store holds frozen-cue (M0) tokens only, so both frogs arms are read on M0 tokens (stated).
Usage: python -u scripts/eci/v2/v20_capacity.py <behav step> --dom D [--arms ...]
"""
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EC = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes')
OUT = REPO / 'results/vision/eci_v2_0/behav'
BEE = EC / 'v2_0/behav'
os.environ['BEHAV_OUT'], os.environ['BEHAV_BEE'] = str(OUT), str(BEE)
sys.path.insert(0, str(REPO / 'scripts/eci/v2_behav'))
import behav as BH  # noqa: E402

SRC_OUT, SRC_BEE = REPO / 'results/vision/eci_v2_behav', EC / 'v2_behav'


def setup(dom):
    (OUT / dom).mkdir(parents=True, exist_ok=True)
    for f in ('probe.parquet', 'videos.parquet'):
        if not (OUT / dom / f).exists():
            shutil.copyfile(SRC_OUT / dom / f, OUT / dom / f)
    (BEE / dom).mkdir(parents=True, exist_ok=True)
    for fam in ('v2', 'v1'):
        l_ = BEE / dom / fam
        if (SRC_BEE / dom / fam).exists() and not (l_.exists() or l_.is_symlink()):
            os.symlink(SRC_BEE / dom / fam, l_)


def arms_of(dom):
    if dom == 'frogs':
        one = [(s, EC / f'v2_0/frogs/sae/v20_frogs_N2_s{s}/sae_theta.pt', 2048, 'M0', 'v2') for s in (900, 901, 902)]
        out = {'V20_1fps': one}
        m = 'M0'
    else:
        out = {'V2F_N2': [(s, EC / f'v2_diag/{dom}/sae/V2F_N2_s{s}/sae_theta.pt', 2048, 'N2', 'v2') for s in range(900, 905)]}
        m = 'N2'
    out['V20F5'] = [(s, EC / f'v2_0/{dom}/sae5/v20_{dom}_N2f5_s{s}/sae_theta.pt', 2048, m, 'v2') for s in range(950, 955)]
    return out


def deploy_seed(dom, arm):
    if arm == 'V2F_N2':
        fv = {s: json.loads((REPO / f'results/vision/eci_v2_diag/{dom}/train/V2F_N2_s{s}.json').read_text())['eval_fve']
              for s in (900, 901, 902)}
    elif arm == 'V20_1fps':
        fv = {int(s): r['eval_fve'] for s, r in json.loads((REPO / 'results/vision/eci_v2_0/frogs/train.json').read_text())['seeds'].items()}
    else:
        fv = {int(s): r['eval_fve'] for s, r in json.loads((REPO / f'results/vision/eci_v2_0/{dom}/train5.json').read_text())['seeds'].items()}
    return max(fv, key=fv.get)


BH.arms_of, BH.deploy_seed = arms_of, deploy_seed

if __name__ == '__main__':
    dom = sys.argv[sys.argv.index('--dom') + 1]
    setup(dom)
    BH.main()
