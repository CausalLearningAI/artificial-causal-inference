"""Per-128-MB-chunk md5 of a file (the download range size of caltechdata_download.py), to locate the corrupted
ranges of a download whose whole-file md5 failed, by comparing against a good re-download.
usage: python -I chunk_md5.py <file> <out.json>      |      python -I chunk_md5.py --compare a.json b.json"""
import hashlib
import json
import sys

CH = 128 << 20
if sys.argv[1] == '--compare':
    a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    bad = [i for i, (x, y) in enumerate(zip(a['md5'], b['md5'])) if x != y]
    print(json.dumps(dict(chunks=len(a['md5']), differing=len(bad), chunk_ids=bad, byte_offsets=[i * CH for i in bad])))
    sys.exit(0)
out = []
with open(sys.argv[1], 'rb') as f:
    for blk in iter(lambda: f.read(CH), b''):
        out.append(hashlib.md5(blk).hexdigest())
json.dump(dict(file=sys.argv[1], chunk_bytes=CH, md5=out), open(sys.argv[2], 'w'))
print(len(out), 'chunks')
