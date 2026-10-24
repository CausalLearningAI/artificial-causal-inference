"""
Tail / odour-box discovery scores (scripts/eci/v2_mask2/discovery.py, definitions and code unchanged) for the P1 arms of
scripts/eci/v2_diag/diag.py. The P1 SAEs read the M0 tokens (only their training pool differs), so discovery.py runs
with arm 'M0' token selection and the SAE loaders redirected to the P1 SAEs; outputs go to
results/vision/eci_v2_diag/discovery/<arm>/{lab,mab}/M0_s<seed>.json (file names keep discovery.py's token-arm name).
    python scripts/eci/v2_diag/p1_discovery.py --arm P1_30k --seeds 500-504 [--steps lab,mab]
    python scripts/eci/v2_diag/p1_discovery.py --arm V1M_d3 --seeds 600-604 --steps lab   (v1-mask arms: lab only)
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts/eci/v2_diag'))
sys.path.insert(0, str(REPO / 'scripts/eci/v2_mask2'))
import diag as D  # noqa: E402
import discovery as DI  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', required=True, choices=('P1_30k', 'P1_90k', 'V1M_d3', 'V1M_d2', 'V2F_M0', 'V2F_N2'))
    p.add_argument('--seeds', default='500-504')
    p.add_argument('--steps', default='lab,mab')
    a = p.parse_args()
    arm = a.arm
    DI.OUT = D.OUT / 'discovery' / arm
    T = DI.T
    T.sae_dir = lambda dom, _a, s: D.sae_dir(dom, arm, s)
    T.load_sae = lambda dom, _a, s, dev: D.load_arm_sae(dom, arm, s, dev)
    T.fve_of = lambda dom, _a, s: D.fve_of(dom, arm, s)
    T.DEPLOY_FROM = (500, 501, 502)
    tok_arm = 'M0'
    if arm.startswith('V2F'):  # V2-full: the step-2 M0 / N2 tokens of the scored frames, own SAEs
        T.DEPLOY_FROM = (900, 901, 902)
        tok_arm = arm.split('_')[1]
    if arm.startswith('V1M'):  # v1-mask arms: their own token store (scripts/eci/v2_diag/v1mask.py), selected by flags
        import numpy as np
        import v1mask
        T.BEE = v1mask.BEE
        T.DEPLOY_FROM = (600, 601, 602)
        base_arm = T.Union.arm

        def arm_fn(self, a_):
            if a_.startswith('V1M'):
                fl = np.concatenate([np.load(q['root'] / 'flags.npy') for q in self.parts])
                return (fl & v1mask.FLAG[a_]) > 0
            return base_arm(self, a_)
        T.Union.arm = arm_fn
        base_init = T.Union.__init__
        step2 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_mask2')

        def init_fn(self, dom, kind):  # frame-level pixel stats (app_px, ...) from the step-2 store of the SAME frames
            base_init(self, dom, kind)
            import pandas as pd
            F2 = pd.concat([pd.read_parquet(q / 'frames.parquet') for q in
                            sorted((step2 / dom / kind).glob('s*'), key=lambda q: int(q.name[1:]))], ignore_index=True)
            assert len(F2) == len(self.F) and (F2.obs.values == self.F.obs.values).all() and \
                (F2.k.values == self.F.k.values).all(), 'frame order differs from the step-2 store'
            for c in F2.columns:
                if c not in self.F.columns:
                    self.F[c] = F2[c].values
        T.Union.__init__ = init_fn
        tok_arm = arm
    ns = argparse.Namespace(arm=tok_arm, seeds=a.seeds)
    for st in a.steps.split(','):
        {'lab': DI.cmd_lab, 'mab': DI.cmd_mab}[st](ns)


if __name__ == '__main__':
    main()
