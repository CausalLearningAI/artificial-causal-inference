"""Download CalMS21 files from the CaltechDATA record s0vdx-0k302 (python requests; the compute nodes have no curl),
256 MB byte ranges fetched by --par threads and written in place, then size + md5 verified against the record API.
A verified file gets a '<name>.ok' marker (with its md5) and is skipped on rerun.
usage: python -I download.py <root> <file names...>"""
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

API = 'https://data.caltech.edu/api/records/s0vdx-0k302/files'
CH = 256 << 20
root, names = sys.argv[1], sys.argv[2:]
raw = os.path.join(root, 'raw')
os.makedirs(raw, exist_ok=True)
rec = requests.get(API, timeout=60).json()
json.dump(rec, open(os.path.join(raw, 'record_files.json'), 'w'), indent=1)
meta = {e['key']: (int(e['size']), e['checksum'].split(':')[1]) for e in rec['entries']}


def fetch(url, p, a, b):
    for k in range(8):
        try:
            r = requests.get(url, headers={'Range': f'bytes={a}-{b}'}, timeout=120)
            if r.status_code == 206 and len(r.content) == b - a + 1:
                with open(p, 'r+b') as f:
                    f.seek(a)
                    f.write(r.content)
                return
            print('retry', a, r.status_code, len(r.content), flush=True)
        except Exception as e:  # network hiccup: retry the whole range
            print('retry', a, repr(e)[:200], flush=True)
        time.sleep(10 * (k + 1))
    raise RuntimeError(f'range {a}-{b} failed')


for f in names:
    size, md5 = meta[f]
    p = os.path.join(raw, f)
    if os.path.exists(p + '.ok'):
        print('already verified', f, flush=True)
        continue
    t0 = time.time()
    url = f'{API}/{f}/content'
    with open(p, 'wb') as fh:
        fh.truncate(size)
    rng = [(a, min(a + CH, size) - 1) for a in range(0, size, CH)]
    with ThreadPoolExecutor(int(os.environ.get('NPAR', 8))) as ex:
        list(ex.map(lambda ab: fetch(url, p, *ab), rng))
    dt = time.time() - t0
    h = hashlib.md5()
    with open(p, 'rb') as fh:
        for blk in iter(lambda: fh.read(64 << 20), b''):
            h.update(blk)
    got = h.hexdigest()
    ok = got == md5 and os.path.getsize(p) == size
    print(('VERIFIED' if ok else 'FAILED'), f, size, got, 'vs', md5, f'{dt:.0f}s {size / dt / 1e6:.1f} MB/s', flush=True)
    if not ok:
        sys.exit(1)
    open(p + '.ok', 'w').write(f'{size} {md5}\n')
