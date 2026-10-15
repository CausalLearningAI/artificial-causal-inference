"""Extract the downloaded CalMS21 zips with python zipfile (no unzip binary on the nodes), member by member; zipfile
checks every member's CRC on extraction. A zip whose members all extract cleanly is deleted (its md5 was verified
against the record before); a report goes to <root>/raw/unzip_report.json.
usage: python -I unzip.py <root> [zip names]"""
import json
import os
import sys
import zipfile

root = sys.argv[1]
WANT = set(sys.argv[2:])
TARGET = {'task1_videos_mp4.zip': 'videos', 'task1_classic_classification.zip': 'annotations',
          'task2_annotation_styles.zip': 'annotations', 'task3_new_behaviors.zip': 'annotations'}
report = {}
for z, sub in TARGET.items():
    if WANT and z not in WANT:
        continue
    zp = os.path.join(root, 'raw', z)
    if not os.path.exists(zp):
        report[z] = 'missing (already extracted?)'
        continue
    zf = zipfile.ZipFile(zp)
    out = os.path.join(root, sub)
    names = [i for i in zf.infolist() if not i.is_dir()]
    bad = []
    for i in names:
        try:
            zf.extract(i, out)
        except Exception as e:  # CRC error etc.: report, remove the partial file, keep going
            bad.append((i.filename, repr(e)))
            q = os.path.join(out, i.filename)
            if os.path.exists(q):
                os.remove(q)
    report[z] = dict(members=len(names), uncompressed_bytes=sum(i.file_size for i in names),
                     compress_types=sorted({i.compress_type for i in names}), bad=bad,
                     files=[[i.filename, i.file_size] for i in names])
    print(z, len(names), 'members', 'bad', bad, flush=True)
    if not bad:
        os.remove(zp)
        print('removed', zp, flush=True)
json.dump(report, open(os.path.join(root, 'raw', f"unzip_report_{os.environ.get('SLURM_JOB_ID', 'local')}.json"), 'w'), indent=1)
