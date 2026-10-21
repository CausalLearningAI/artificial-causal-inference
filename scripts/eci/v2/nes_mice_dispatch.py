"""
Run one existing NES runner (scripts/eci/run_nes*.py) or the NESS driver (scripts/eci/atlas_v37_methods.py) on mice v1
with an IN-MEMORY patch of src/eci/domain.py MiceDomain (no shared file is edited), from the code tree NES_CODE_ROOT.

V2_MICE_MODE
  S                  family S (sex): female vs male pools, Welch two-sample, unit = pool (one video per pool and
                     stage; pools are single-sex; no per-mouse summaries exist). Analyses S_stage<s> (s = 1..6, both
                     genotypes) and S_<g>_stage<s> (within genotype g = het / wt); tau > 0 = higher in females. They
                     REPLACE the core analyses (families A / B) for this run; null / frame-illustration analyses move
                     to S_stage2; no paired null.
  strata:<line>:<sex>  the core analyses (families A, B) restricted to the pools of one gene line and / or sex ('all' =
                     unrestricted): every Analysis.select first applies contrasts.subset_design (the runners' own
                     --line / --sex subset, identical rows), so the NESS driver tests the same units as the runners.
usage: python nes_mice_dispatch.py run_nes.py --domain mice ... | atlas_v37_methods.py --res ... (argument passthrough)
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(os.environ.get('NES_CODE_ROOT') or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(ROOT / 'src'))
import eci.domain as Dm  # noqa: E402
from eci import contrasts as C  # noqa: E402

MODE = os.environ.get('V2_MICE_MODE', '')
M = Dm.MiceDomain
if MODE == 'S':
    def _analyses(self):
        out = []
        for s in range(1, 7):
            out.append(Dm.Analysis(f'S_stage{s}', 'B', 'pool', {'genotype': 'all', 'stage': s, 'transition': ''},
                                   ('f>m', 'f<m'), where={'stage': s}, arm='sex', control=('m',), treatment=('f',)))
        for g in ('het', 'wt'):
            for s in range(1, 7):
                out.append(Dm.Analysis(f'S_{g}_stage{s}', 'B', 'pool', {'genotype': g, 'stage': s, 'transition': ''},
                                       ('f>m', 'f<m'), where={'stage': s, 'genotype': g}, arm='sex', control=('m',),
                                       treatment=('f',)))
        return out
    M.analyses = property(_analyses)
    M.null_two = M.frame_analysis = 'S_stage2'
    M.frame_key = 'sstage2'
    M.null_paired = None
    M.text = {**M.text, 'families_nes': 'Family S = female vs male pools within stage (and within genotype); unit = '
              'pool, tau > 0 = higher in females.', 'families_bouts': 'Family S = female vs male pools within stage.',
              'null_two': 'Sex labels shuffled across pools (stage 2', 'frame': 'sex at stage 2',
              'null_two_bouts': 'sex shuffled across pools (S stage 2)', 'null_paired_bouts': ''}
elif MODE.startswith('strata:'):
    _, LINE, SEX = MODE.split(':')
    _core = M.analyses.fget

    class StratAnalysis(Dm.Analysis):
        def select(self, design):
            return super().select(C.subset_design(design, LINE, SEX))

        def arm_masks(self, df):
            c, t = super().arm_masks(df)
            m = (df['line'] == LINE).values if LINE != 'all' and 'line' in df else True
            m = m & ((df['sex'] == SEX).values if SEX != 'all' and 'sex' in df else True)
            return c & m, t & m

    def _analyses(self):
        return [StratAnalysis(**a.__dict__) for a in _core(self)]
    M.analyses = property(_analyses)
elif MODE:
    raise SystemExit(f'unknown V2_MICE_MODE {MODE!r}')
runner = ROOT / 'scripts/eci' / sys.argv[1]
sys.argv = [str(runner)] + sys.argv[2:]
print(f'[nes_mice_dispatch] mode {MODE!r}, code root {ROOT}, runner {runner.name}', flush=True)
runpy.run_path(str(runner), run_name='__main__')
