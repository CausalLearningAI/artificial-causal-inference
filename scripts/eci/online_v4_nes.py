"""
Run one existing NES / DINES script (scripts/eci/run_nes.py, run_nes_bouts.py, run_nes_latency.py,
atlas_v37_methods.py) unchanged on the online-model v4 layouts ANTS_ECI=eci_c1v4 / eci_c1v4_noday
(scripts/eci/online_v4.py), from the code tree this file sits in (scripts/eci/online_v4_nes.sh runs it from a
`git archive` of HEAD). Two class-attribute patches of src/eci/domain.py AntsDomain (no shared file is edited):

  1. null / frame-illustration analysis v4_2_vs_6 instead of v2_1_vs_2 (no v2 videos here), as
     scripts/eci/rollout_r4_nes.py does for the new model's eci_r4v4 layout.
  2. Under the runners' --day (Domain.day_nuisance) the infested vs healthy analyses ARE day-matched: they get
     day_col = recording_date, so Analysis.select keeps only the recording days holding both arms (v4: day F only;
     the healthy colony was also filmed on day D, the infested one on day E) and the day indicators are the nuisance
     (one day left -> none). Without --day they stay on every day (day-unmatched), as the domain defines them.
     Analysis ids are unchanged; meta 'confound' says which variant ran (without --day every family B analysis,
     within-colony too, is labelled 'day-unmatched').

    usage: ANTS_ECI=eci_c1v4 python scripts/eci/online_v4_nes.py run_nes.py --domain ants --sae ... --day
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
LAYOUT = os.environ.get('ANTS_ECI', 'eci')
if not LAYOUT.startswith('eci_c1v4'):
    raise SystemExit(f'ANTS_ECI={LAYOUT}: this wrapper is for the eci_c1v4* layouts only')
import eci.domain as Dm  # noqa: E402

A = Dm.AntsDomain
A.null_two = A.frame_analysis = A.frame_key = 'v4_2_vs_6'
A.text = {**A.text, 'null_two': 'Treatment labels shuffled across videos (v4_2_vs_6', 'frame': 'treatment in v4 (2 vs 6)',
          'null_two_bouts': 'treatment shuffled across videos (v4_2_vs_6)'}
_infested = A.infested


def infested(self, aid, ts, e=None):
    an = _infested(self, aid, ts, e)
    if self.day_nuisance:
        an.day_col, an.day_match = self.day_col, True
        if e is not None:
            days = self.day_confound(e, an)[2]
            shared = sorted(set(days.get('control', {})) & set(days.get('treatment', {})))
            n = {v: sum(days.get(v, {}).get(d, 0) for d in shared) for v in ('control', 'treatment')}
            an.meta['confound'] = (f'colony; day-matched: only recording day(s) {"/".join(shared) or "none"} hold both '
                                   f'colonies (healthy n={n["control"]}, infested n={n["treatment"]} videos there)')
        else:
            an.meta['confound'] = 'colony; day-matched (recording days holding both colonies only)'
    else:
        an.meta['confound'] = 'day-unmatched: ' + an.meta['confound']
    return an


_within = A.within


def within(self, e, c, t, conf):
    an = _within(self, e, c, t, conf)
    if not self.day_nuisance:
        an.meta['confound'] = 'day-unmatched' + (f': {conf}' if conf else '')
    return an


A.infested, A.within = infested, within
runner = ROOT / 'scripts/eci' / sys.argv[1]
sys.argv = [str(runner)] + sys.argv[2:]
print(f'[online_v4_nes] code root {ROOT}, ANTS_ECI {LAYOUT}, runner {runner.name}', flush=True)
runpy.run_path(str(runner), run_name='__main__')
