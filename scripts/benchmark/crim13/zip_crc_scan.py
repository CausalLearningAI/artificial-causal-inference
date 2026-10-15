"""Locate corruption in a zip whose whole-file md5 failed: read every member (zipfile checks its CRC32) and report
the bad members with their byte ranges and the 128 MB download ranges (caltechdata_download.py) they touch.
usage: python -I zip_crc_scan.py <zip> <out.json>"""
import json
import sys
import zipfile

CH = 128 << 20
zf = zipfile.ZipFile(sys.argv[1])
inf = sorted(zf.infolist(), key=lambda i: i.header_offset)
bad = []
for k, i in enumerate(inf):
    try:
        with zf.open(i) as f:
            while f.read(64 << 20):
                pass
    except Exception as e:  # BadZipFile (CRC), zlib.error
        a = i.header_offset
        b = inf[k + 1].header_offset if k + 1 < len(inf) else a + i.compress_size
        bad.append(dict(member=i.filename, error=repr(e)[:200], bytes=[a, b], chunks=list(range(a // CH, (b - 1) // CH + 1))))
        print('BAD', bad[-1], flush=True)
json.dump(dict(zip=sys.argv[1], members=len(inf), bad=bad), open(sys.argv[2], 'w'), indent=1)
print('members', len(inf), 'bad', len(bad))
