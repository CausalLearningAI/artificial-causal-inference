"""
Run one existing NES runner (scripts/eci/run_nes.py / run_nes_bouts.py / run_nes_latency.py) on R4 codes, unchanged,
from the code tree ROOT (env NES_CODE_ROOT, default this repo; rollout_r4_nes.sh points it at a clean `git archive`
of HEAD when the working tree of the NES code is dirty).
The ants R4 layouts get two class-attribute patches (no file of the shared tree is edited):
    ANTS_ECI=eci_r4v3*  the 30-min layout's pooled contrast v3 t=2 vs t=6+8 (LAYOUT_CONTRASTS['eci30']), and the
                        null / frame-illustration analysis v3_2_vs_6 instead of v2_1_vs_2 (no v2 videos there)
    ANTS_ECI=eci_r4v2*  nothing (v2_1_vs_2 is the only contrast, as in every ants layout)
    usage: python scripts/eci/rollout_r4_nes.py run_nes.py --domain ants --sae ... (runner arguments)
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(os.environ.get('NES_CODE_ROOT') or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT / 'src'))
import eci.domain as Dm  # noqa: E402

eci = os.environ.get('ANTS_ECI', 'eci')
A = Dm.AntsDomain
if eci.startswith('eci_r4v3'):
    A.LAYOUT_CONTRASTS = {**A.LAYOUT_CONTRASTS, eci: A.LAYOUT_CONTRASTS['eci30']}
    A.null_two = A.frame_analysis = A.frame_key = 'v3_2_vs_6'
    A.text = {**A.text, 'null_two': 'Treatment labels shuffled across videos (v3_2_vs_6', 'frame': 'treatment in v3 (2 vs 6)',
              'null_two_bouts': 'treatment shuffled across videos (v3_2_vs_6)'}
if eci.startswith('eci_r4v4'):  # ants v4 healthy + infested: no v2 videos there
    A.null_two = A.frame_analysis = A.frame_key = 'v4_2_vs_6'
    A.text = {**A.text, 'null_two': 'Treatment labels shuffled across videos (v4_2_vs_6', 'frame': 'treatment in v4 (2 vs 6)',
              'null_two_bouts': 'treatment shuffled across videos (v4_2_vs_6)'}
runner = ROOT / 'scripts/eci' / sys.argv[1]
sys.argv = [str(runner)] + sys.argv[2:]
print(f'[rollout_r4_nes] code root {ROOT}, ANTS_ECI {eci}, runner {runner.name}', flush=True)
runpy.run_path(str(runner), run_name='__main__')
