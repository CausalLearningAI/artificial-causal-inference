"""CalMS21 ECI - GPU utilisation / memory per job (nvidia-smi logs, one sample / 60 s) and encode throughput per config
(encode JSONs) -> OUT/res/util.{md,json}. Login safe (small text files). GPU-hours from sacct elapsed time of the jobs."""
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calms_common import OUT  # noqa: E402

LOG = OUT / 'logs'
rows = []
for f in sorted(LOG.glob('gpu_*.csv')):
    m = re.fullmatch(r'gpu_(.+)_(\d+)\.csv', f.name)
    if not m:
        continue
    name, jid = m.groups()
    u, mem = [], []
    with open(f) as fh:
        for r in csv.reader(fh):
            if len(r) < 3 or 'utilization' in r[1]:
                continue
            try:
                u.append(float(r[1].strip().split()[0]))
                mem.append(float(r[2].strip().split()[0]))
            except ValueError:
                pass
    if not u:
        continue
    el = subprocess.run(['sacct', '-j', jid, '-X', '-n', '-o', 'ElapsedRaw,State,NodeList'], capture_output=True,
                        text=True).stdout.split()
    rows.append({'job': name, 'id': int(jid), 'samples': len(u), 'util_mean': float(np.mean(u)),
                 'util_median': float(np.median(u)), 'mem_max_mib': float(max(mem)),
                 'elapsed_h': (int(el[0]) / 3600) if el else None, 'state': el[1] if len(el) > 1 else None,
                 'node': el[2] if len(el) > 2 else None})
kind = lambda j: re.sub(r'(_?(train|eval)\d+|_s\d+)$', '', j)  # noqa: E731
G = {}
for r in rows:
    G.setdefault(kind(r['job']), []).append(r)
L = ['# CalMS21 GPU utilisation (nvidia-smi every 60 s) and GPU-hours', '',
     '| job type | jobs | GPU-hours | mean util % (sample-weighted) | max memory MiB |', '|---|---|---|---|---|']
tot = 0.0
for k, rr in sorted(G.items()):
    gh = sum(r['elapsed_h'] or 0 for r in rr)
    tot += gh
    w = sum(r['samples'] for r in rr)
    L.append(f"| {k} | {len(rr)} | {gh:.2f} | {sum(r['util_mean'] * r['samples'] for r in rr) / w:.1f} | "
             f"{max(r['mem_max_mib'] for r in rr):.0f} |")
L += ['', f'Total GPU-hours (jobs with a utilisation log): {tot:.2f}', '']
E = {}
for f in sorted((OUT / 'encode_json').glob('*.json')):
    if 'smoke' in f.name:
        continue
    j = json.loads(f.read_text())
    for c, v in j['per_config'].items():
        e = E.setdefault(c, {'gpu_s': 0.0, 'inputs': 0, 'grid': v['grid'], 'size': v['size'], 'gpus': set()})
        e['gpu_s'] += v['gpu_s']
        e['inputs'] += v['inputs']
        e['gpus'].add(j['gpu'])
    E.setdefault('_wall', []).append((f.stem, j['frames'], j['wall_s'], j['frames_per_wall_s'], j['gpu'],
                                      j.get('decode_wait_s')))
L += ['## encode throughput per config (summed over shards; inputs = images, or 16-frame clips for V-JEPA)', '',
      '| config | grid | input px | inputs | GPU-s in the encoder | inputs / GPU-s | GPU types |', '|---|---|---|---|---|---|---|']
for c, e in E.items():
    if c == '_wall':
        continue
    L.append(f"| {c} | {e['grid']} | {e['size']} | {e['inputs']:,} | {e['gpu_s']:.0f} | {e['inputs'] / max(e['gpu_s'], 1e-9):.2f} | "
             f"{', '.join(sorted(e['gpus']))} |")
L += ['', '| encode job | frames | wall s | frames / wall-s | GPU | waited for decode s |', '|---|---|---|---|---|---|']
for w in E.get('_wall', []):
    L.append('| ' + ' | '.join(str(x) for x in w) + ' |')
b = OUT / 'bench' / 'bench_bf16.json'
if b.exists():
    L += ['', '## random-input bench (bf16 autocast, incl. canvas pad + resize)', '',
          '| config | grid | px | batch | inputs / s | tokens / s | peak GB | GPU |', '|---|---|---|---|---|---|---|---|']
    for r in json.loads(b.read_text()):
        if 'error' in r:
            L.append(f"| {r['config']} | {r['error']} |")
        else:
            L.append(f"| {r['config']} | {r['grid']} | {r['size']} | {r['batch']} | {r['inputs_per_s']} | "
                     f"{r['tokens_per_s']:,.0f} | {r['peak_mem_gb']} | {r['gpu']} |")
(OUT / 'res').mkdir(parents=True, exist_ok=True)
(OUT / 'res' / 'util.md').write_text('\n'.join(L) + '\n')
(OUT / 'res' / 'util.json').write_text(json.dumps({'jobs': rows}, indent=1))
print('\n'.join(L))
