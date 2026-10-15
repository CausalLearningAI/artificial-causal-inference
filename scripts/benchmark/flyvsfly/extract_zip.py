"""Extract a (md5-verified) zip with python zipfile, member by member, skipping members that match --skip (regex on
the member path). zipfile checks every member's CRC32 while reading; after writing, each file's size is checked
against the central directory. If every selected member is good, the zip is deleted (unless --keep).
A JSON report (extracted, skipped, bad) goes next to the output dir. Used by flyvsfly/ and crim13/ download jobs.
usage: python -I extract_zip.py <zip> <out dir> <report.json> [--skip REGEX] [--strip N] [--keep]
  --strip N drops the first N path components of each member (like tar --strip-components)."""
import argparse
import json
import os
import re
import shutil
import zipfile

ap = argparse.ArgumentParser()
ap.add_argument('zip')
ap.add_argument('out')
ap.add_argument('report')
ap.add_argument('--skip', default=None)
ap.add_argument('--strip', type=int, default=0)
ap.add_argument('--keep', action='store_true')
a = ap.parse_args()
skip = re.compile(a.skip) if a.skip else None
zf = zipfile.ZipFile(a.zip)
files = [i for i in zf.infolist() if not i.is_dir()]
done, skipped, bad = [], [], []
for i in files:
    if skip and skip.search(i.filename):
        skipped.append([i.filename, i.file_size, f'{i.CRC:08x}'])
        continue
    parts = i.filename.split('/')[a.strip:]
    if not parts or '..' in parts:
        bad.append([i.filename, 'unsafe or empty path after strip'])
        continue
    dst = os.path.join(a.out, *parts)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        with zf.open(i) as src, open(dst, 'wb') as fo:  # CRC is checked when the member is read to the end
            shutil.copyfileobj(src, fo, 64 << 20)
        if os.path.getsize(dst) != i.file_size:
            raise IOError(f'size {os.path.getsize(dst)} != {i.file_size}')
        done.append([i.filename, os.path.relpath(dst, a.out), i.file_size, f'{i.CRC:08x}'])
    except Exception as e:  # CRC error, short write: report, drop the partial file, keep going
        bad.append([i.filename, repr(e)])
        if os.path.exists(dst):
            os.remove(dst)
rep = dict(zip=os.path.basename(a.zip), zip_bytes=os.path.getsize(a.zip), members=len(files), extracted=done,
           skipped=skipped, bad=bad, skip_regex=a.skip, deleted_zip=False)
print(rep['zip'], len(files), 'members:', len(done), 'extracted', len(skipped), 'skipped', len(bad), 'BAD', bad, flush=True)
if not bad and not a.keep:
    os.remove(a.zip)
    rep['deleted_zip'] = True
    print('removed', a.zip, flush=True)
json.dump(rep, open(a.report, 'w'), indent=1)
if bad:
    raise SystemExit(f'EXTRACTION FAILED for {len(bad)} members of {a.zip}')
