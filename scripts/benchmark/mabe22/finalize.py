"""Save the 4 CRC-failed zip members' raw bytes under videos/crc_failed/, verify every other member was extracted with the right size, then delete the (md5-verified) zips."""
import zipfile, struct, os, json
D = "dataset/benchmark/mabe22"; bad = {"mouse_submission": ["QO9ONQTPOYAI26MYHPHY"], "beetle_userTrain": ["eb0e794d7f6e703d6d7a"], "beetle_submission": ["5411da61501cc4d42ad4", "868aea506ef41a301f3b"]}
ok = True
for z in ["mouse_userTrain", "mouse_submission", "beetle_userTrain", "beetle_submission"]:
    p = f"{D}/raw/{z}_videos_resized_224.zip"; zf = zipfile.ZipFile(p)
    miss = 0; wrong = 0; n = 0
    for info in zf.infolist():
        if info.is_dir(): continue
        n += 1; name = info.filename; stem = name[:-4]
        if stem in bad.get(z, []):
            with open(p, "rb") as fh:
                fh.seek(info.header_offset); h = fh.read(30); nl, el = struct.unpack("<HH", h[26:30]); fh.seek(info.header_offset + 30 + nl + el); data = fh.read(info.file_size)
            os.makedirs(f"{D}/videos/crc_failed/{z}", exist_ok=True); open(f"{D}/videos/crc_failed/{z}/{name}", "wb").write(data)
            continue
        f = f"{D}/videos/{z}/{name}"
        if not os.path.exists(f): miss += 1
        elif os.path.getsize(f) != info.file_size: wrong += 1
    print(z, "members", n, "missing", miss, "wrong_size", wrong, flush=True); ok &= (miss == 0 and wrong == 0)
if ok:
    for z in ["mouse_userTrain", "mouse_submission", "beetle_userTrain", "beetle_submission"]: os.remove(f"{D}/raw/{z}_videos_resized_224.zip")
    print("zips deleted")
else: print("NOT deleting zips")
