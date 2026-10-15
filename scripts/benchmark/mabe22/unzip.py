"""Unzip the four MABe22 video zips with python zipfile (no unzip binary on nodes).
Extracts member by member; members whose CRC fails are reported (and the partial file removed) instead of aborting."""
import zipfile, os, json
D = "dataset/benchmark/mabe22"
report = {}
for z in ["mouse_userTrain", "mouse_submission", "beetle_userTrain", "beetle_submission"]:
    zf = zipfile.ZipFile(f"{D}/raw/{z}_videos_resized_224.zip")
    out = f"{D}/videos/{z}"
    names = [i for i in zf.infolist() if not i.is_dir()]
    bad = []
    for i in names:
        try:
            zf.extract(i, out)
        except Exception as e:
            bad.append((i.filename, repr(e)))
            p = os.path.join(out, i.filename)
            if os.path.exists(p): os.remove(p)
    report[z] = dict(members=len(names), uncompressed_bytes=sum(i.file_size for i in names),
                     compress_types=sorted({i.compress_type for i in names}), bad=bad,
                     exts=sorted({os.path.splitext(i.filename)[1] for i in names}))
    print(z, json.dumps(report[z]), flush=True)
json.dump(report, open("results/vision/benchmark_mabe22/inventory/unzip_report.json", "w"), indent=1)
