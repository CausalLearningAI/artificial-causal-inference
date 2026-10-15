"""
MABe22 mouse ECI - EXPERIMENT P2 (rules: eci_p2.sh): SAE training, theta*, eval read-out and cross-fitted scoring for
the two P2 arms on the DINOv3-L/16 512 px block-20 store (tag p2d3l_512), with eci_enc_train.py's main() unchanged
(only the store, the output directory and the arm table are redirected):
    P2D  x = [z_t, z_t - z_{t-60}]   (2 s difference; = the encoder study's D3L_d3l_512_L20 recipe)
    P2N  x = z_t only
Outputs: OUT/p2/sae/<run>_s<seed>/{sae.pt, train.json}, OUT/p2/eval/<run>_s<seed>.json, run = <arm>_p2d3l_512_L20.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eci_enc_train as T  # noqa: E402
from eci_common import OUT  # noqa: E402

T.ETOK = Path(os.environ['MABE_ENC_TOK'])
T.RUN = OUT / 'p2'
T.ARMS = ['P2D', 'P2N']


def arm_cfg(arm, tag, layer):
    tag = 'p2d3l_512'
    return dict(store=T.ETOK, suffix=f'_{tag}', t='L20_t', p='L20_p60', lens='n', D=1024, delta=arm == 'P2D',
                run=f'{arm}_{tag}_L20')


T.arm_cfg = arm_cfg

if __name__ == '__main__':
    T.main()
