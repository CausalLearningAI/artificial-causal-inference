"""
Run one existing NES runner (scripts/eci/run_nes.py / run_nes_bouts.py / run_nes_latency.py) or the NESS driver
(scripts/eci/atlas_v37_methods.py, NESS = DINES, src/eci/dines.py) on the tadpole r4 codes, from the code tree ROOT
(env NES_CODE_ROOT: a `git archive` of a pinned commit, scripts/eci/tadpole_r4_nes.sh). No file of the shared tree is
edited; two in-memory / local-copy patches:
    TadpolesDomain  TADPOLE_TAG=_r4 (layout dataset/frogs/eci_tadpole_r4, results results/vision/frogs/eci_tadpole_r4/nes)
                    and CONTRASTS = WT vs FoxP1, WT vs En1 (the r4 tadpole layout holds no Scrambled videos)
    NESS driver     its parse_res() knows mice / ants / frogs juveniles only: a patched COPY in the job's code tree
                    adds results/vision/frogs/eci_tadpole*/nes/<sae> -> domain 'tadpoles', key 'frogs_tadpole' (and
                    the same contrasts patch), then runs as __main__.
    usage: python scripts/eci/tadpole_r4_nes.py run_nes.py --domain tadpoles --sae ... (runner arguments)
           python scripts/eci/tadpole_r4_nes.py atlas_v37_methods.py --res ... --out-root ...
"""
import os
import runpy
import sys
from pathlib import Path

os.environ['TADPOLE_TAG'] = '_r4'
ROOT = Path(os.environ.get('NES_CODE_ROOT') or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT / 'src'))
import eci.domain as Dm  # noqa: E402

CONTRASTS = (('WT', 'FoxP1'), ('WT', 'En1'))
Dm.TadpolesDomain.CONTRASTS = CONTRASTS
assert str(Dm.TadpolesDomain.eci_dir).endswith('eci_tadpole_r4'), Dm.TadpolesDomain.eci_dir
runner = ROOT / 'scripts/eci' / sys.argv[1]
if sys.argv[1] == 'atlas_v37_methods.py':
    src = runner.read_text()
    old = "    raise SystemExit(f'{res}: not a mice / ants / frogs NES result set')"
    new = ("    if p[0] == 'frogs' and p[1].startswith('eci_tadpole') and p[2] == 'nes':\n"
           "        return 'tadpoles', None, 'frogs_tadpole'\n" + old)
    assert src.count(old) == 1, 'parse_res changed upstream: re-check the tadpole patch'
    old2 = "    D = Dm.get_domain(dom, 'core')\n"
    assert src.count(old2) == 1, 'get_domain call changed upstream: re-check the tadpole patch'
    src = src.replace(old, new).replace(old2, f"    Dm.TadpolesDomain.CONTRASTS = {CONTRASTS!r}\n" + old2)
    runner = runner.with_name('atlas_v37_methods_tadpole_r4.py')
    runner.write_text(src)
sys.argv = [str(runner)] + sys.argv[2:]
print(f'[tadpole_r4_nes] code root {ROOT}, eci_dir {Dm.TadpolesDomain.eci_dir}, contrasts {CONTRASTS}, runner {runner.name}',
      flush=True)
runpy.run_path(str(runner), run_name='__main__')
