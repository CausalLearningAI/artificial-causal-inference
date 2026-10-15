"""Download files of a CaltechDATA record with resume + retries, then verify size + md5 against the record API.

Used by both scripts/benchmark/flyvsfly/download.sh and scripts/benchmark/crim13/download.sh.
- The record's content URL 302-redirects to an S3 URL that is signed for 60 s, so every byte range goes through the
  API URL again (a fresh redirect per request).
- A file is fetched as 128 MB byte ranges by NPAR threads, written in place; each finished range is appended to
  '<name>.parts', so a killed job resumes where it stopped (range-level resume, like `curl -C -`).
- Each range is retried up to 10 times with backoff; a range that still fails stops the job loudly (rerun resumes).
- After all ranges: size + md5 of the whole file vs the record. Match -> '<name>.ok' (size md5); mismatch -> exit 1
  and the .parts file is removed so a rerun starts clean.
usage: python -I caltechdata_download.py <record id> <dest dir> <file names...>"""
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

rec_id, dest, names = sys.argv[1], sys.argv[2], sys.argv[3:]
API = f'https://data.caltech.edu/api/records/{rec_id}/files'
CH = 128 << 20
os.makedirs(dest, exist_ok=True)
rec = requests.get(API, timeout=60).json()
json.dump(rec, open(os.path.join(dest, f'record_files_{rec_id}.json'), 'w'), indent=1)
meta = {e['key']: (int(e['size']), e['checksum'].split(':')[1]) for e in rec['entries']}
lock = threading.Lock()


def fetch(url, p, parts, a, b):
    for k in range(10):
        try:
            r = requests.get(url, headers={'Range': f'bytes={a}-{b}'}, timeout=300)
            if r.status_code == 206 and len(r.content) == b - a + 1:
                with open(p, 'r+b') as f:
                    f.seek(a)
                    f.write(r.content)
                    f.flush()
                    os.fsync(f.fileno())
                with lock, open(parts, 'a') as g:
                    g.write(f'{a}\n')
                return
            print('retry', os.path.basename(p), a, r.status_code, len(r.content), flush=True)
        except Exception as e:  # network hiccup: retry the whole range
            print('retry', os.path.basename(p), a, repr(e)[:200], flush=True)
        time.sleep(15 * (k + 1))
    raise RuntimeError(f'{p}: range {a}-{b} failed 10 times')


for f in names:
    size, md5 = meta[f]
    p = os.path.join(dest, f)
    parts = p + '.parts'
    if os.path.exists(p + '.ok'):
        print('already verified', f, flush=True)
        continue
    if not os.path.exists(p) or os.path.getsize(p) != size:
        with open(p, 'wb') as fh:
            fh.truncate(size)
        open(parts, 'w').close()
    done = {int(x) for x in open(parts).read().split()} if os.path.exists(parts) else set()
    rng = [(a, min(a + CH, size) - 1) for a in range(0, size, CH) if a not in done]
    print(f'{f}: {size} bytes, {len(done)} ranges already done, {len(rng)} to fetch', flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(int(os.environ.get('NPAR', 8))) as ex:
        list(ex.map(lambda ab: fetch(url=f'{API}/{f}/content', p=p, parts=parts, a=ab[0], b=ab[1]), rng))
    dt = max(time.time() - t0, 1e-3)
    h = hashlib.md5()
    with open(p, 'rb') as fh:
        for blk in iter(lambda: fh.read(64 << 20), b''):
            h.update(blk)
    got = h.hexdigest()
    ok = got == md5 and os.path.getsize(p) == size
    print(('VERIFIED' if ok else 'FAILED'), f, size, got, 'vs', md5,
          f'{dt:.0f}s {sum(b - a + 1 for a, b in rng) / dt / 1e6:.1f} MB/s', flush=True)
    if not ok:
        os.remove(parts)
        sys.exit(1)
    open(p + '.ok', 'w').write(f'{size} {md5}\n')
    os.remove(parts)
