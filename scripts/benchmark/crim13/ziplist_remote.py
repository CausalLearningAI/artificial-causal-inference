"""List the members of a zip on CaltechDATA without downloading it: HTTP range reads feed zipfile, which only needs
the end-of-central-directory and the central directory (a few MB at most).
usage: python -I ziplist_remote.py <record id> <zip name> [out.json]
       python -I ziplist_remote.py <record id> <zip name> --get <member> <out path>   (fetch one small member)"""
import io
import json
import sys
import zipfile

import requests


class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        # the content URL 302-redirects to an S3 URL signed for GET only (HEAD is 403) and valid 60 s, so every range
        # read goes through the API URL again; the size comes from the record's file listing
        rec, name = url.split('/records/')[1].split('/files/')
        name = name.split('/content')[0]
        ents = requests.get(f'https://data.caltech.edu/api/records/{rec.split("/")[0]}/files', timeout=60).json()
        self.size = {e['key']: int(e['size']) for e in ents['entries']}[name]
        self.nreq = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b''
        b = min(self.pos + n, self.size) - 1
        r = requests.get(self.url, headers={'Range': f'bytes={self.pos}-{b}'}, timeout=120)
        assert r.status_code == 206, r.status_code
        self.nreq += 1
        self.pos += len(r.content)
        return r.content


rec, name = sys.argv[1], sys.argv[2]
f = HttpFile(f'https://data.caltech.edu/api/records/{rec}/files/{name}/content')
zf = zipfile.ZipFile(f)
mem = [[i.filename, i.file_size, i.compress_size, i.header_offset, i.compress_type] for i in zf.infolist()]
print(name, f.size, 'bytes', len(mem), 'members', f.nreq, 'range requests', file=sys.stderr)
out = dict(zip=name, zip_bytes=f.size, members=mem)
if len(sys.argv) > 3 and sys.argv[3] == '--get':
    open(sys.argv[5], 'wb').write(zf.read(sys.argv[4]))  # zipfile checks the member CRC
    print('wrote', sys.argv[5], file=sys.stderr)
elif len(sys.argv) > 3:
    json.dump(out, open(sys.argv[3], 'w'), indent=0)
else:
    for m in mem:
        print(*m, sep='\t')
