"""
Tail / odour-box discovery scores (scripts/eci/v2_mask2/discovery.py, definitions and code unchanged) for the P1 arms of
scripts/eci/v2_diag/diag.py. The P1 SAEs read the M0 tokens (only their training pool differs), so discovery.py runs
with arm 'M0' token selection and the SAE loaders redirected to the P1 SAEs; outputs go to
results/vision/eci_v2_diag/discovery/<arm>/{lab,mab}/M0_s<seed>.json (file names keep discovery.py's token-arm name).
    python scripts/eci/v2_diag/p1_discovery.py --arm P1_30k --seeds 500-504 [--steps lab,mab]
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
    p.add_argument('--arm', required=True, choices=('P1_30k', 'P1_90k'))
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
    ns = argparse.Namespace(arm='M0', seeds=a.seeds)
    for st in a.steps.split(','):
        {'lab': DI.cmd_lab, 'mab': DI.cmd_mab}[st](ns)


if __name__ == '__main__':
    main()
