"""
MABe22 mouse ECI - EXPERIMENT P2 (rules: eci_p2.sh): one GPU job that (1) re-measures the bf16-vs-fp32 token agreement
of DINOv3-L/16 512 px block 20 on 8 eval clips and aborts if the mean cosine < 0.9996, then (2) encodes the train set
(8 shards) and the eval set (12 shards), kinds t and p60 in one decode per clip, with eci_enc_encode.py unchanged
(only its store / mask / json paths are redirected). Store: MABE_ENC_TOK (eci_p2.sh: BeeGFS pairpush/mabe22/), tag
p2d3l_512; encode / precision JSONs: OUT/p2/.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eci_enc_encode as EE  # noqa: E402
from eci_common import OUT, log  # noqa: E402

EE.ETOK = Path(os.environ['MABE_ENC_TOK'])
EE.R3 = Path(os.environ['MABE_R3_MASK'])
EE.EOUT = OUT / 'p2'
TAG, LAYER, MIN_COS = 'p2d3l_512', 20, 0.9996


def ns(**kw):
    base = dict(mode='encode', bf16=True, orbit=False, workers=6, set='eval', shard=[0], nshard=12, encoder='dinov3l',
                size=512, layers=[LAYER], kinds=['t', 'p60'], tag=TAG, vj_pos='extrap', vj_stride=8, bs=32, smoke=0,
                check_r3=False, bench=[])
    base.update(kw)
    return argparse.Namespace(**base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=int, default=0, help='clips per shard (test only; keys get a _smoke suffix)')
    a = ap.parse_args()
    EE.EOUT.mkdir(parents=True, exist_ok=True)
    pj = EE.EOUT / f'prec_dinov3l_{TAG}.json'
    if not pj.exists():
        EE.cmd_prec(ns(mode='prec', bf16=False, smoke=8, bs=16))
    pr = json.loads(pj.read_text())['layers'][str(LAYER)]
    log(f'precision gate: block {LAYER} bf16 vs fp32 cosine mean {pr["cos_mean"]:.6f} p1 {pr["cos_p1"]:.6f}')
    if pr['cos_mean'] < MIN_COS:
        raise SystemExit(f'ABORT: bf16 cosine mean {pr["cos_mean"]:.6f} < {MIN_COS}')
    EE.cmd_encode(ns(set='train', nshard=8, shard=list(range(8)), smoke=a.smoke))
    EE.cmd_encode(ns(set='eval', nshard=12, shard=list(range(12)), smoke=a.smoke))
    log('P2 encode done')


if __name__ == '__main__':
    main()
